package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Label;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;

/**
 * Emits carrier-, type-, family-, and access-specialized overlap-fold loops directly into
 * generated CPU classes. Each owned output starts at represented positive zero, scans logical
 * input occurrences in canonical row-major order, and is stored exactly once. FLOAT64/FLOAT32
 * additions retain their represented precision, BFLOAT16 rounds after each contribution, and
 * INT32/INT64 additions remain modular. Direct segment access uses the predefined native-order
 * unaligned layouts. One guarded mixed-carrier FLOAT32 FOLD2D form accepts only the frozen
 * padded/dilated geometry after complete cold proof. It uses an output-cell, kernel-position,
 * then column loop, starts each result at {@code +0.0f}, preserves canonical occurrence order and
 * sequential FLOAT32 additions, and stores once. Only singleton dimensions proved by that guard
 * are removed. Its semantic algorithm and hot-loop dataflow match the optimal clean Java form;
 * the guard only embeds proved constants and removes general coordinate machinery. Arbitrary
 * legal output subranges remain valid, and every unproved case uses the typed general-long body.
 * All coordinates, addresses, and arithmetic values are invocation-
 * local primitives; concrete geometry remains in the invocation-private packed array.
 */
public final class CpuFoldEmitter {
    private static final ClassDesc FLOAT_CLASS = ClassDesc.of(Float.class.getName());
    private static final ClassDesc SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout");
    private static final ClassDesc DOUBLE_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfDouble");
    private static final ClassDesc FLOAT_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfFloat");
    private static final ClassDesc SHORT_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfShort");
    private static final ClassDesc INT_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfInt");
    private static final ClassDesc LONG_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfLong");

    /** Creates a stateless fold emitter that retains no generated or invocation state. */
    public CpuFoldEmitter() { }

    /**
     * Emits one two-boundary, workspace-free, structurally specialized fold body, using the
     * guarded frozen-shape form only after its complete geometry proof and otherwise retaining
     * the applicable typed fallback.
     *
     * @param code non-null generated method body
     * @param specialization non-null matching scalar fold specialization
     * @param ir non-null instruction-free structural fold encoding
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the specialization or IR is not a supported exact fold
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        FoldEncoding p = FoldEncoding.parse(ir);
        if (specialization.carrierPattern().size() != 2 || specialization.scratchParameter()) {
            throw new IllegalArgumentException("fold requires two boundaries and no scratch");
        }
        boolean ints = specialization.loopAddressing(ir)
                == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT;
        if (ints && p.denseLinear && p.family.equals("FOLD_AXIS")
                && p.inputRank == 2 && p.outputRank == 1) {
            emitDenseAxisRankOne(code, specialization, p);
        } else if (!ints && p.family.equals("FOLD2D") && p.type == DataType.FLOAT32
                && p.inputRank == 3 && p.outputRank == 4
                && specialization.carrierPattern().getFirst() == CarrierAccess.FLOAT_ARRAY
                && specialization.carrierPattern().getLast() == CarrierAccess.MEMORY_SEGMENT) {
            emitBoundedTargetFold2dOrGeneralLong(code, specialization, p);
        } else {
            emitTyped(code, specialization, p, ints);
        }
    }

    private static void emitBoundedTargetFold2dOrGeneralLong(CodeBuilder code,
            CpuKernelSpecialization specialization, FoldEncoding p) {
        var fallback = code.newLabel();
        var complete = code.newLabel();
        requireGeometryAtMost(code, 4, 512, fallback);
        requireGeometryAtMost(code, 5, 512, fallback);
        geometry(code, 5);
        geometry(code, 4);
        code.lcmp().branch(Opcode.IFLT, fallback);
        requireGeometry(code, 6, 4_608, fallback);
        requireGeometry(code, p.inputLayout, 3, fallback);
        requireGeometry(code, p.outputLayout, 4, fallback);
        requireGeometrySequence(code, p.inputLayout + 2,
                new long[]{1, 9, 512, 4_608, 512, 1}, fallback);
        requireGeometrySequence(code, p.outputLayout + 2,
                new long[]{1, 1, 16, 32, 1_024, 1_024, 64, 2}, fallback);
        requireGeometrySequence(code, p.mapping,
                new long[]{3, 3, 1, 1, 2, 2, 2, 2, 16, 32}, fallback);
        requireBoundedBase(code, p.inputLayout + 1, 4_608, fallback);
        requireBoundedBase(code, p.outputLayout + 1, 1_023, fallback);
        for (int axis = 0; axis < p.outputRank; axis++) {
            geometry(code, p.outputSeed + axis).loadConstant(0L).lcmp()
                    .branch(Opcode.IFLT, fallback);
            geometry(code, p.outputSeed + axis).loadConstant(
                    new long[]{1, 1, 16, 32}[axis]).lcmp().branch(Opcode.IFGE, fallback);
        }
        emitBoundedTargetFold2d(code, specialization, p);
        code.branch(Opcode.GOTO, complete).labelBinding(fallback);
        emitTyped(code, specialization, p, false);
        code.labelBinding(complete);
    }

    private static void emitBoundedTargetFold2d(CodeBuilder code,
            CpuKernelSpecialization specialization, FoldEncoding p) {
        int logical = code.allocateLocal(TypeKind.INT);
        int end = code.allocateLocal(TypeKind.INT);
        int inputBase = code.allocateLocal(TypeKind.INT);
        int outputBase = code.allocateLocal(TypeKind.INT);
        int q = code.allocateLocal(TypeKind.INT);
        int column = code.allocateLocal(TypeKind.INT);
        int kernelY = code.allocateLocal(TypeKind.INT);
        int kernelX = code.allocateLocal(TypeKind.INT);
        int outputY = code.allocateLocal(TypeKind.INT);
        int outputX = code.allocateLocal(TypeKind.INT);
        int inputAddress = code.allocateLocal(TypeKind.INT);
        int outputAddress = code.allocateLocal(TypeKind.INT);
        int sum = code.allocateLocal(TypeKind.FLOAT);
        int right = code.allocateLocal(TypeKind.FLOAT);
        code.lload(3).l2i().istore(logical).lload(5).l2i().istore(end);
        geometry(code, p.inputLayout + 1).l2i().istore(inputBase);
        geometry(code, p.outputLayout + 1).l2i().istore(outputBase);
        var carriers = new CpuCarrierEmitter(code);
        var outer = code.newLabel();
        var done = code.newLabel();
        var qLoop = code.newLabel();
        var columnLoop = code.newLabel();
        var nextQ = code.newLabel();
        var noMatch = code.newLabel();
        code.labelBinding(outer).iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        positiveZero(code, DataType.FLOAT32, sum);
        code.iload(logical).loadConstant(32).idiv().istore(outputY);
        code.iload(logical).iload(outputY).loadConstant(32).imul().isub().istore(outputX);
        code.loadConstant(0).istore(q).labelBinding(qLoop);
        code.iload(q).loadConstant(9).branch(Opcode.IF_ICMPGE, nextQ);
        code.iload(q).loadConstant(3).idiv().istore(kernelY);
        code.iload(q).iload(kernelY).loadConstant(3).imul().isub().istore(kernelX);
        code.loadConstant(0).istore(column).labelBinding(columnLoop);
        code.iload(column).loadConstant(512).branch(Opcode.IF_ICMPGE, nextQ);
        code.iload(column).loadConstant(32).idiv().iload(kernelY).loadConstant(2).imul()
                .iadd().loadConstant(2).isub().iload(outputY)
                .branch(Opcode.IF_ICMPNE, noMatch);
        code.iload(column).iload(column).loadConstant(32).idiv().loadConstant(32).imul()
                .isub().iload(kernelX).loadConstant(2).imul().iadd().loadConstant(2).isub()
                .iload(outputX).branch(Opcode.IF_ICMPNE, noMatch);
        code.iload(inputBase).iload(q).loadConstant(512).imul().iadd().iload(column).iadd()
                .istore(inputAddress);
        loadCarrier(code, carriers, DataType.FLOAT32, specialization.carrierPattern().getFirst(),
                0, inputAddress, true);
        code.fstore(right).fload(sum).fload(right).fadd().fstore(sum);
        code.labelBinding(noMatch).iinc(column, 1).branch(Opcode.GOTO, columnLoop);
        code.labelBinding(nextQ).iinc(q, 1);
        code.iload(q).loadConstant(9).branch(Opcode.IF_ICMPLT, qLoop);
        code.iload(outputBase).iload(logical).loadConstant(2).imul().iadd()
                .istore(outputAddress);
        storeCarrier(code, carriers, DataType.FLOAT32, specialization.carrierPattern().getLast(),
                1, outputAddress, sum, true);
        code.iinc(logical, 1).branch(Opcode.GOTO, outer).labelBinding(done);
    }

    private static void requireGeometrySequence(CodeBuilder code, int first, long[] expected,
            Label fallback) {
        for (int index = 0; index < expected.length; index++) {
            requireGeometry(code, first + index, expected[index], fallback);
        }
    }

    private static void requireGeometry(CodeBuilder code, int index, long expected,
            Label fallback) {
        geometry(code, index).loadConstant(expected).lcmp().branch(Opcode.IFNE, fallback);
    }

    private static void requireGeometryAtMost(CodeBuilder code, int index, long maximum,
            Label fallback) {
        geometry(code, index).loadConstant(0L).lcmp().branch(Opcode.IFLT, fallback);
        geometry(code, index).loadConstant(maximum).lcmp().branch(Opcode.IFGT, fallback);
    }

    private static void requireBoundedBase(CodeBuilder code, int index, long span,
            Label fallback) {
        geometry(code, index).loadConstant(0L).lcmp().branch(Opcode.IFLT, fallback);
        geometry(code, index).loadConstant((long) Integer.MAX_VALUE - span).lcmp()
                .branch(Opcode.IFGT, fallback);
    }

    private static void emitDenseAxisRankOne(CodeBuilder code, CpuKernelSpecialization s,
            FoldEncoding p) {
        int logical = code.allocateLocal(TypeKind.INT);
        int end = code.allocateLocal(TypeKind.INT);
        int inputOrdinal = code.allocateLocal(TypeKind.INT);
        int inputCount = code.allocateLocal(TypeKind.INT);
        int inputAddress = code.allocateLocal(TypeKind.INT);
        int outputAddress = code.allocateLocal(TypeKind.INT);
        int inputBase = code.allocateLocal(TypeKind.INT);
        int outputBase = code.allocateLocal(TypeKind.INT);
        int within = code.allocateLocal(TypeKind.INT);
        int target = code.allocateLocal(TypeKind.INT);
        int step = code.allocateLocal(TypeKind.INT);
        int windowSize = code.allocateLocal(TypeKind.INT);
        int sum = code.allocateLocal(localKind(p.type));
        int right = code.allocateLocal(localKind(p.type));
        code.lload(3).l2i().istore(logical).lload(5).l2i().istore(end);
        geometry(code, p.inputLayout + 1).l2i().istore(inputBase);
        geometry(code, p.outputLayout + 1).l2i().istore(outputBase);
        geometry(code, 6).l2i().istore(inputCount);
        geometry(code, p.mapping + 2).l2i().istore(step);
        geometry(code, p.mapping + 3).l2i().istore(windowSize);
        var zeroBase = code.newLabel();
        var complete = code.newLabel();
        code.iload(inputBase).branch(Opcode.IFNE, zeroBase);
        code.iload(outputBase).branch(Opcode.IFNE, zeroBase);
        emitDenseAxisRankOneBody(code, s, p, logical, end, inputOrdinal, inputCount,
                inputAddress, outputAddress, inputBase, outputBase, within, target, step,
                windowSize, sum, right, true);
        code.branch(Opcode.GOTO, complete).labelBinding(zeroBase);
        emitDenseAxisRankOneBody(code, s, p, logical, end, inputOrdinal, inputCount,
                inputAddress, outputAddress, inputBase, outputBase, within, target, step,
                windowSize, sum, right, false);
        code.labelBinding(complete);
    }

    private static void emitDenseAxisRankOneBody(CodeBuilder code, CpuKernelSpecialization s,
            FoldEncoding p, int logical, int end, int inputOrdinal, int inputCount,
            int inputAddress, int outputAddress, int inputBase, int outputBase, int within,
            int target, int step, int windowSize, int sum, int right, boolean zeroBase) {
        var carriers = new CpuCarrierEmitter(code);
        var outer = code.newLabel();
        var done = code.newLabel();
        var inner = code.newLabel();
        var write = code.newLabel();
        code.labelBinding(outer).iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        positiveZero(code, p.type, sum);
        code.loadConstant(0).istore(inputOrdinal).loadConstant(0).istore(within)
                .loadConstant(0).istore(target).labelBinding(inner);
        code.iload(inputOrdinal).iload(inputCount).branch(Opcode.IF_ICMPGE, write);
        var noMatch = code.newLabel();
        code.iload(target).iload(logical).branch(Opcode.IF_ICMPNE, noMatch);
        if (zeroBase) code.iload(inputOrdinal).istore(inputAddress);
        else code.iload(inputBase).iload(inputOrdinal).iadd().istore(inputAddress);
        loadCarrier(code, carriers, p.type, s.carrierPattern().getFirst(), 0, inputAddress, true);
        storeValue(code, p.type, right);
        add(code, p.type, sum, right);
        code.labelBinding(noMatch).iinc(inputOrdinal, 1).iinc(within, 1).iinc(target, 1);
        var carried = code.newLabel();
        code.iload(within).iload(windowSize).branch(Opcode.IF_ICMPGE, carried);
        code.branch(Opcode.GOTO, inner).labelBinding(carried).loadConstant(0).istore(within);
        code.iload(target).iload(windowSize).isub().iload(step).iadd().istore(target);
        code.branch(Opcode.GOTO, inner);
        code.labelBinding(write);
        if (zeroBase) code.iload(logical).istore(outputAddress);
        else code.iload(outputBase).iload(logical).iadd().istore(outputAddress);
        storeCarrier(code, carriers, p.type, s.carrierPattern().getLast(), 1, outputAddress, sum,
                true);
        code.iinc(logical, 1).branch(Opcode.GOTO, outer).labelBinding(done);
    }

    private static void emitTyped(CodeBuilder code, CpuKernelSpecialization s, FoldEncoding p,
            boolean ints) {
        TypeKind addressKind = ints ? TypeKind.INT : TypeKind.LONG;
        int logical = code.allocateLocal(addressKind);
        int end = code.allocateLocal(addressKind);
        int inputOrdinal = code.allocateLocal(addressKind);
        int inputCount = code.allocateLocal(addressKind);
        int inputAddress = code.allocateLocal(addressKind);
        int outputAddress = code.allocateLocal(addressKind);
        int sum = code.allocateLocal(localKind(p.type));
        int right = code.allocateLocal(localKind(p.type));
        int match = code.allocateLocal(TypeKind.INT);
        int[] input = coordinates(code, p.inputRank, ints);
        int[] output = coordinates(code, p.outputRank, ints);
        LayoutLocals inputLayout = loadLayout(code, p.inputLayout, p.inputRank, ints);
        LayoutLocals outputLayout = loadLayout(code, p.outputLayout, p.outputRank, ints);
        AxisMappingLocals axisMapping = p.family.equals("FOLD_AXIS")
                ? loadAxisMapping(code, p.mapping, ints) : null;
        int[] twoDimensionalMapping = p.family.equals("FOLD2D")
                ? loadMapping(code, p.mapping, 10, ints) : null;
        if (ints) {
            code.lload(3).l2i().istore(logical).lload(5).l2i().istore(end);
            geometry(code, 6).l2i().istore(inputCount);
        } else {
            code.lload(3).lstore(logical).lload(5).lstore(end);
            geometry(code, 6).lstore(inputCount);
        }
        for (int axis = 0; axis < p.outputRank; axis++) {
            geometry(code, p.outputSeed + axis);
            if (ints) code.l2i().istore(output[axis]); else code.lstore(output[axis]);
        }
        var carriers = new CpuCarrierEmitter(code);
        var outer = code.newLabel();
        var done = code.newLabel();
        var inner = code.newLabel();
        var write = code.newLabel();
        code.labelBinding(outer);
        compare(code, logical, end, ints, Opcode.IF_ICMPGE, Opcode.IFGE, done);
        positiveZero(code, p.type, sum);
        for (int coordinate : input) zero(code, coordinate, ints);
        zero(code, inputOrdinal, ints);
        code.labelBinding(inner);
        compare(code, inputOrdinal, inputCount, ints, Opcode.IF_ICMPGE, Opcode.IFGE, write);
        code.loadConstant(1).istore(match);
        if (p.family.equals("FOLD_AXIS")) {
            emitAxisMatch(code, p, input, output, match, ints, axisMapping);
        } else {
            emitTwoDimensionalMatch(code, input, output, match, ints, twoDimensionalMapping);
        }
        var noMatch = code.newLabel();
        code.iload(match).branch(Opcode.IFEQ, noMatch);
        address(code, inputLayout, input, inputAddress, ints);
        loadCarrier(code, carriers, p.type, s.carrierPattern().getFirst(), 0, inputAddress, ints);
        storeValue(code, p.type, right);
        add(code, p.type, sum, right);
        code.labelBinding(noMatch);
        advance(code, inputLayout, input, ints);
        increment(code, inputOrdinal, ints);
        code.branch(Opcode.GOTO, inner).labelBinding(write);
        address(code, outputLayout, output, outputAddress, ints);
        storeCarrier(code, carriers, p.type, s.carrierPattern().getLast(), 1, outputAddress, sum,
                ints);
        advance(code, outputLayout, output, ints);
        increment(code, logical, ints);
        code.branch(Opcode.GOTO, outer).labelBinding(done);
    }

    private static void emitAxisMatch(CodeBuilder code, FoldEncoding p, int[] input,
            int[] output, int match, boolean ints, AxisMappingLocals mapping) {
        int target = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        for (int axis = 0; axis < p.outputRank; axis++) {
            var selected = code.newLabel();
            var compare = code.newLabel();
            code.loadConstant(axis);
            code.iload(mapping.axis()).branch(Opcode.IF_ICMPEQ, selected);
            load(code, input[axis], ints);
            code.branch(Opcode.GOTO, compare).labelBinding(selected);
            load(code, input[axis], ints);
            load(code, mapping.step(), ints);
            multiply(code, ints);
            load(code, input[p.inputRank - 1], ints);
            if (ints) code.iadd(); else code.ladd();
            code.labelBinding(compare);
            store(code, target, ints);
            var equal = code.newLabel();
            compare(code, target, output[axis], ints, Opcode.IF_ICMPEQ, Opcode.IFEQ, equal);
            code.loadConstant(0).istore(match).labelBinding(equal);
        }
    }

    private static void emitTwoDimensionalMatch(CodeBuilder code, int[] input, int[] output,
            int match, boolean ints, int[] mapping) {
        int q = valueLocal(code, ints), column = valueLocal(code, ints);
        int kernelArea = valueLocal(code, ints), channel = valueLocal(code, ints);
        int kernel = valueLocal(code, ints), kh = valueLocal(code, ints);
        int kw = valueLocal(code, ints), oh = valueLocal(code, ints);
        int ow = valueLocal(code, ints), height = valueLocal(code, ints);
        int width = valueLocal(code, ints);
        copy(code, input[1], q, ints); copy(code, input[2], column, ints);
        load(code, mapping[0], ints);
        load(code, mapping[1], ints);
        multiply(code, ints); store(code, kernelArea, ints);
        load(code, q, ints); load(code, kernelArea, ints); divide(code, ints);
        store(code, channel, ints);
        load(code, q, ints); load(code, channel, ints); load(code, kernelArea, ints);
        multiply(code, ints); subtract(code, ints); store(code, kernel, ints);
        load(code, kernel, ints); load(code, mapping[1], ints);
        divide(code, ints); store(code, kh, ints);
        load(code, kernel, ints); load(code, kh, ints); load(code, mapping[1], ints);
        multiply(code, ints); subtract(code, ints); store(code, kw, ints);
        load(code, column, ints); load(code, mapping[9], ints);
        divide(code, ints); store(code, oh, ints);
        load(code, column, ints); load(code, oh, ints); load(code, mapping[9], ints);
        multiply(code, ints); subtract(code, ints); store(code, ow, ints);
        mappedCoordinate(code, oh, kh, mapping[2], mapping[4], mapping[6],
                height, ints);
        mappedCoordinate(code, ow, kw, mapping[3], mapping[5], mapping[7],
                width, ints);
        requireEqual(code, input[0], output[0], match, ints);
        requireEqual(code, channel, output[1], match, ints);
        requireEqual(code, height, output[2], match, ints);
        requireEqual(code, width, output[3], match, ints);
    }

    private static void mappedCoordinate(CodeBuilder code, int outputColumn, int kernel,
            int stride, int padding, int dilation, int target, boolean ints) {
        load(code, outputColumn, ints); load(code, stride, ints);
        multiply(code, ints); load(code, padding, ints);
        subtract(code, ints); load(code, kernel, ints); load(code, dilation, ints);
        multiply(code, ints); addPrimitive(code, ints); store(code, target, ints);
    }

    private static void requireEqual(CodeBuilder code, int left, int right, int match,
            boolean ints) {
        var equal = code.newLabel();
        compare(code, left, right, ints, Opcode.IF_ICMPEQ, Opcode.IFEQ, equal);
        code.loadConstant(0).istore(match).labelBinding(equal);
    }

    private static void address(CodeBuilder code, LayoutLocals layout, int[] coordinates, int target,
            boolean ints) {
        load(code, layout.base(), ints); store(code, target, ints);
        for (int axis = 0; axis < coordinates.length; axis++) {
            load(code, target, ints); load(code, coordinates[axis], ints);
            load(code, layout.strides()[axis], ints);
            multiply(code, ints); addPrimitive(code, ints); store(code, target, ints);
        }
    }

    private static void advance(CodeBuilder code, LayoutLocals layout, int[] coordinates,
            boolean ints) {
        Label finished = code.newLabel();
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            increment(code, coordinates[axis], ints);
            Label carry = code.newLabel();
            load(code, coordinates[axis], ints); load(code, layout.extents()[axis], ints);
            if (ints) code.branch(Opcode.IF_ICMPGE, carry);
            else code.lcmp().branch(Opcode.IFGE, carry);
            code.branch(Opcode.GOTO, finished).labelBinding(carry);
            zero(code, coordinates[axis], ints);
        }
        code.labelBinding(finished);
    }

    private static void add(CodeBuilder code, DataType type, int left, int right) {
        if (type == DataType.BFLOAT16) {
            emitBfloatAdd(code, left, right);
            code.istore(left);
            return;
        }
        loadValue(code, type, left); loadValue(code, type, right);
        switch (type) {
            case FLOAT64 -> code.dadd();
            case FLOAT32 -> code.fadd();
            case INT32 -> code.iadd();
            case INT64 -> code.ladd();
            case BFLOAT16 -> throw new AssertionError("handled above");
            case BOOL -> throw new IllegalArgumentException("BOOL fold is unsupported");
        }
        storeValue(code, type, left);
    }

    private static void loadCarrier(CodeBuilder code, CpuCarrierEmitter carriers, DataType type,
            CarrierAccess access, int parameterSlot, int addressLocal, boolean intAddress) {
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            carriers.load(type, access, parameterSlot, addressLocal, intAddress);
            return;
        }
        code.aload(parameterSlot).getstatic(VALUE_LAYOUT, layoutField(type), layoutClass(type));
        byteOffset(code, type, addressLocal, intAddress);
        code.invokeinterface(SEGMENT, "get", MethodTypeDesc.of(primitive(type), layoutClass(type),
                ConstantDescs.CD_long));
    }

    private static void storeCarrier(CodeBuilder code, CpuCarrierEmitter carriers, DataType type,
            CarrierAccess access, int parameterSlot, int addressLocal, int valueLocal,
            boolean intAddress) {
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            carriers.store(type, access, parameterSlot, addressLocal, valueLocal, intAddress);
            return;
        }
        code.aload(parameterSlot).getstatic(VALUE_LAYOUT, layoutField(type), layoutClass(type));
        byteOffset(code, type, addressLocal, intAddress);
        loadValue(code, type, valueLocal);
        code.invokeinterface(SEGMENT, "set", MethodTypeDesc.of(ConstantDescs.CD_void,
                layoutClass(type), ConstantDescs.CD_long, primitive(type)));
    }

    private static void byteOffset(CodeBuilder code, DataType type, int addressLocal,
            boolean intAddress) {
        if (intAddress) code.iload(addressLocal).i2l();
        else code.lload(addressLocal);
        code.loadConstant((long) type.byteWidth()).lmul();
    }

    private static String layoutField(DataType type) {
        return switch (type) {
            case FLOAT64 -> "JAVA_DOUBLE_UNALIGNED";
            case FLOAT32 -> "JAVA_FLOAT_UNALIGNED";
            case BFLOAT16 -> "JAVA_SHORT_UNALIGNED";
            case INT32 -> "JAVA_INT_UNALIGNED";
            case INT64 -> "JAVA_LONG_UNALIGNED";
            case BOOL -> throw new IllegalArgumentException("BOOL fold is unsupported");
        };
    }

    private static ClassDesc layoutClass(DataType type) {
        return switch (type) {
            case FLOAT64 -> DOUBLE_LAYOUT;
            case FLOAT32 -> FLOAT_LAYOUT;
            case BFLOAT16 -> SHORT_LAYOUT;
            case INT32 -> INT_LAYOUT;
            case INT64 -> LONG_LAYOUT;
            case BOOL -> throw new IllegalArgumentException("BOOL fold is unsupported");
        };
    }

    private static ClassDesc primitive(DataType type) {
        return switch (type) {
            case FLOAT64 -> ConstantDescs.CD_double;
            case FLOAT32 -> ConstantDescs.CD_float;
            case BFLOAT16 -> ConstantDescs.CD_short;
            case INT32 -> ConstantDescs.CD_int;
            case INT64 -> ConstantDescs.CD_long;
            case BOOL -> throw new IllegalArgumentException("BOOL fold is unsupported");
        };
    }

    private static void emitBfloatAdd(CodeBuilder code, int left, int right) {
        int resultBits = code.allocateLocal(TypeKind.INT);
        int upper = code.allocateLocal(TypeKind.INT);
        int lower = code.allocateLocal(TypeKind.INT);
        code.iload(left).loadConstant(16).ishl()
                .invokestatic(FLOAT_CLASS, "intBitsToFloat",
                        MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int));
        code.iload(right).loadConstant(16).ishl()
                .invokestatic(FLOAT_CLASS, "intBitsToFloat",
                        MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int));
        code.fadd().invokestatic(FLOAT_CLASS, "floatToRawIntBits",
                MethodTypeDesc.of(ConstantDescs.CD_int, ConstantDescs.CD_float)).istore(resultBits);
        var finite = code.newLabel(); var round = code.newLabel(); var complete = code.newLabel();
        code.iload(resultBits).loadConstant(0x7f800000).iand().loadConstant(0x7f800000)
                .branch(Opcode.IF_ICMPNE, finite);
        code.iload(resultBits).loadConstant(0x007fffff).iand().branch(Opcode.IFEQ, finite);
        code.iload(resultBits).loadConstant(16).iushr().loadConstant(0x40).ior()
                .branch(Opcode.GOTO, complete);
        code.labelBinding(finite).iload(resultBits).loadConstant(16).iushr().istore(upper);
        code.iload(resultBits).loadConstant(0xffff).iand().istore(lower);
        code.iload(lower).loadConstant(0x8000).branch(Opcode.IF_ICMPGT, round);
        var tie = code.newLabel();
        code.iload(lower).loadConstant(0x8000).branch(Opcode.IF_ICMPEQ, tie);
        code.iload(upper).branch(Opcode.GOTO, complete);
        code.labelBinding(tie).iload(upper).loadConstant(1).iand().branch(Opcode.IFNE, round);
        code.iload(upper).branch(Opcode.GOTO, complete);
        code.labelBinding(round).iinc(upper, 1).iload(upper).labelBinding(complete);
    }

    private static void positiveZero(CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.loadConstant(0.0d).dstore(local);
            case FLOAT32 -> code.loadConstant(0.0f).fstore(local);
            case INT64 -> code.loadConstant(0L).lstore(local);
            case BFLOAT16, INT32 -> code.loadConstant(0).istore(local);
            case BOOL -> throw new IllegalArgumentException("BOOL fold is unsupported");
        }
    }

    private static int[] coordinates(CodeBuilder code, int rank, boolean ints) {
        int[] locals = new int[rank];
        for (int i = 0; i < rank; i++) locals[i] = valueLocal(code, ints);
        return locals;
    }

    private static LayoutLocals loadLayout(CodeBuilder code, int geometryIndex, int rank,
            boolean ints) {
        int base = valueLocal(code, ints);
        geometry(code, geometryIndex + 1);
        if (ints) code.l2i().istore(base); else code.lstore(base);
        int[] extents = loadMapping(code, geometryIndex + 2, rank, ints);
        int[] strides = loadMapping(code, geometryIndex + 2 + rank, rank, ints);
        return new LayoutLocals(base, extents, strides);
    }

    private static AxisMappingLocals loadAxisMapping(CodeBuilder code, int geometryIndex,
            boolean ints) {
        int axis = code.allocateLocal(TypeKind.INT);
        geometry(code, geometryIndex).l2i().istore(axis);
        int step = valueLocal(code, ints);
        geometry(code, geometryIndex + 2);
        if (ints) code.l2i().istore(step); else code.lstore(step);
        return new AxisMappingLocals(axis, step);
    }

    private static int[] loadMapping(CodeBuilder code, int geometryIndex, int count,
            boolean ints) {
        int[] locals = coordinates(code, count, ints);
        for (int i = 0; i < count; i++) {
            geometry(code, geometryIndex + i);
            if (ints) code.l2i().istore(locals[i]); else code.lstore(locals[i]);
        }
        return locals;
    }

    private static int valueLocal(CodeBuilder code, boolean ints) {
        return code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
    }

    private static CodeBuilder geometry(CodeBuilder code, int index) {
        return code.aload(2).loadConstant(index).laload();
    }

    private static void compare(CodeBuilder code, int left, int right, boolean ints,
            Opcode intOpcode, Opcode longOpcode, Label target) {
        load(code, left, ints); load(code, right, ints);
        if (ints) code.branch(intOpcode, target); else code.lcmp().branch(longOpcode, target);
    }

    private static void copy(CodeBuilder code, int source, int target, boolean ints) {
        load(code, source, ints); store(code, target, ints);
    }

    private static void load(CodeBuilder code, int local, boolean ints) {
        if (ints) code.iload(local); else code.lload(local);
    }

    private static void store(CodeBuilder code, int local, boolean ints) {
        if (ints) code.istore(local); else code.lstore(local);
    }

    private static void zero(CodeBuilder code, int local, boolean ints) {
        if (ints) code.loadConstant(0).istore(local); else code.loadConstant(0L).lstore(local);
    }

    private static void increment(CodeBuilder code, int local, boolean ints) {
        if (ints) code.iinc(local, 1);
        else code.lload(local).loadConstant(1L).ladd().lstore(local);
    }

    private static void multiply(CodeBuilder code, boolean ints) {
        if (ints) code.imul(); else code.lmul();
    }

    private static void divide(CodeBuilder code, boolean ints) {
        if (ints) code.idiv(); else code.ldiv();
    }

    private static void subtract(CodeBuilder code, boolean ints) {
        if (ints) code.isub(); else code.lsub();
    }

    private static void addPrimitive(CodeBuilder code, boolean ints) {
        if (ints) code.iadd(); else code.ladd();
    }

    private static void loadValue(CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dload(local);
            case FLOAT32 -> code.fload(local);
            case INT64 -> code.lload(local);
            case BFLOAT16, INT32, BOOL -> code.iload(local);
        }
    }

    private static void storeValue(CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dstore(local);
            case FLOAT32 -> code.fstore(local);
            case INT64 -> code.lstore(local);
            case BFLOAT16, INT32, BOOL -> code.istore(local);
        }
    }

    private static TypeKind localKind(DataType type) {
        return switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE;
            case FLOAT32 -> TypeKind.FLOAT;
            case INT64 -> TypeKind.LONG;
            case BFLOAT16, INT32, BOOL -> TypeKind.INT;
        };
    }

    private record FoldEncoding(String family, DataType type, int inputRank, int outputRank,
            int outputSeed, int inputLayout, int outputLayout, int mapping, boolean denseLinear) {
        static FoldEncoding parse(CpuKernelIr ir) {
            String identity = ir.familyIdentity();
            if (!identity.startsWith("fold:") || ir.values().size() != 2
                    || !ir.instructions().isEmpty()) {
                throw new IllegalArgumentException("unsupported fold encoding");
            }
            int familyEnd = identity.indexOf(':', 5);
            String family = identity.substring(5, familyEnd);
            if (!family.equals("FOLD_AXIS") && !family.equals("FOLD2D")) {
                throw new IllegalArgumentException("unsupported fold family");
            }
            int inputRank = ir.values().getFirst().accessPlan().iterationRank();
            int outputRank = ir.values().getLast().accessPlan().iterationRank();
            int outputSeed = 8 + inputRank + outputRank;
            int inputLayout = outputSeed + outputRank;
            int outputLayout = inputLayout + 2 + 2 * inputRank;
            int mapping = outputLayout + 2 + 2 * outputRank;
            boolean denseLinear = ir.values().stream().allMatch(value -> value.accessPlan().regime()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime
                            .DENSE_LINEAR);
            return new FoldEncoding(family, ir.values().getFirst().dataType(), inputRank,
                    outputRank, outputSeed, inputLayout, outputLayout, mapping, denseLinear);
        }
    }

    private record LayoutLocals(int base, int[] extents, int[] strides) { }

    private record AxisMappingLocals(int axis, int step) { }
}
