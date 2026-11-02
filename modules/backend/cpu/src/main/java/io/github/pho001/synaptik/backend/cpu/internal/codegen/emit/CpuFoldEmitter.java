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
 * additions retain their represented precision, BFLOAT16/FLOAT16 accumulate exact decoded
 * contributions in FLOAT32 and narrow once at the output store, and
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
 * Selected F32/F64 compact-input FOLD2D instead inverse-gathers at most one stored position per
 * kernel coordinate in ascending ky/kx order, with no expanded-input scan or allocation. A
 * complete invocation-entry int proof includes inverse numerators and physical carrier bases;
 * failed int proofs retain the checked long algorithm, not ordinary all-columns scanning.
 * Ordinary folds retain the pre-existing shared geometry-generic traversal; repairing low
 * arithmetic does not introduce a dtype-specific mapping algorithm. That traversal can contain
 * hoistable repeated coordinate decoding. Its retention preserves mapping/order and nonlow
 * ClassFiles; only the changed decode/add/final-conversion dataflow is qualified against the
 * primitive Java arithmetic oracle, not whole-loop optimality or performance parity.
 *
 * <p>Controlled folds guard output cells by their full logical boundary ordinal and contributors
 * by the input boundary's own proved ROW provenance. Inactive outputs retain positive zero;
 * inactive contributors are skipped before matching, operand loads, and ordered additions.
 * Compact inverse gathering reconstructs the compact input ordinal from logical coordinates,
 * never from a physical carrier address.</p>
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
        if (ir.familyIdentity().endsWith(":kernel-broadcast-input-v1")) {
            if (!p.family.equals("FOLD2D") || p.inputRank != 3 || p.outputRank != 4
                    || p.type != DataType.FLOAT32 && p.type != DataType.FLOAT64) {
                throw new IllegalArgumentException("unsupported compact fold encoding");
            }
            emitKernelBroadcastGuarded(code, specialization, p);
        } else if (ints && p.denseLinear && p.family.equals("FOLD_AXIS")
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

    /**
     * Emits output-cell inverse gathering from stored [N,C,P], in ascending ky/kx order.
     * Coordinate/address arithmetic uses the entry-proved width; cold admission proves long extrema.
     * No expanded cursor, all-position scan or arithmetic helper survives into generated code.
     * @param code generated body
     * @param s authenticated typed carriers
     * @param p checked F32/F64 FOLD2D encoding with compact physical input
     * @param ints whether the complete entry guard proved all arithmetic fits int
     */
    private static void emitKernelBroadcast(CodeBuilder code, CpuKernelSpecialization s, FoldEncoding p, boolean ints) {
        int logical = valueLocal(code, ints), end = valueLocal(code, ints);
        int ky = valueLocal(code, ints), kx = valueLocal(code, ints);
        int a = valueLocal(code, ints), b = valueLocal(code, ints);
        int oh = valueLocal(code, ints), ow = valueLocal(code, ints);
        int row = valueLocal(code, ints);
        int base = valueLocal(code, ints), inputAddress = valueLocal(code, ints), outputAddress = valueLocal(code, ints);
        int sum = code.allocateLocal(localKind(p.type)), right = code.allocateLocal(localKind(p.type));
        int[] output = coordinates(code, 4, ints);
        LayoutLocals inputLayout = loadLayout(code, p.inputLayout, 3, ints);
        LayoutLocals outputLayout = loadLayout(code, p.outputLayout, 4, ints);
        int[] m = loadMapping(code, p.mapping, 10, ints);
        code.lload(3); if (ints) code.l2i(); store(code, logical, ints);
        code.lload(5); if (ints) code.l2i(); store(code, end, ints);
        for (int axis = 0; axis < 4; axis++) {
            geometry(code, p.outputSeed + axis); if (ints) code.l2i(); store(code, output[axis], ints);
        }
        var carriers = new CpuCarrierEmitter(code);
        var outer = code.newLabel(); var done = code.newLabel(); var yLoop = code.newLabel();
        var xLoop = code.newLabel(); var nextY = code.newLabel(); var nextX = code.newLabel(); var write = code.newLabel();
        compare(code, logical, end, ints, Opcode.IF_ICMPGE, Opcode.IFGE, done);
        load(code, inputLayout.base, ints); load(code, output[0], ints); load(code, inputLayout.strides[0], ints);
        multiply(code, ints); addPrimitive(code, ints); load(code, output[1], ints); load(code, inputLayout.strides[1], ints);
        multiply(code, ints); addPrimitive(code, ints); store(code, base, ints);
        address(code, outputLayout, output, outputAddress, ints);
        code.labelBinding(outer);
        positiveZero(code, p.type, sum);
        CpuLoopEmitter.emitActiveOrdinalGuard(code, 1, logical,
                ints ? TypeKind.INT : TypeKind.LONG, write);
        zero(code, ky, ints);
        code.labelBinding(yLoop); compare(code, ky, m[0], ints, Opcode.IF_ICMPGE, Opcode.IFGE, write);
        inverseCoordinate(code, output[2], m[4], ky, m[6], m[2], m[8], a, oh, ints, nextY);
        load(code, base, ints); load(code, oh, ints); load(code, m[9], ints); multiply(code, ints);
        addPrimitive(code, ints); store(code, row, ints);
        zero(code, kx, ints);
        code.labelBinding(xLoop); compare(code, kx, m[1], ints, Opcode.IF_ICMPGE, Opcode.IFGE, nextY);
        inverseCoordinate(code, output[3], m[5], kx, m[7], m[3], m[9], b, ow, ints, nextX);
        load(code, row, ints); load(code, ow, ints); addPrimitive(code, ints); store(code, inputAddress, ints);
        if (CpuCarrierEmitter.controlFacts()
                .map(control -> control.boundaryRows().get(0).axis() >= 0).orElse(false)) {
            int inputOrdinal = code.allocateLocal(TypeKind.LONG);
            load(code, output[0], ints); if (ints) code.i2l();
            load(code, inputLayout.extents[1], ints); if (ints) code.i2l();
            code.lmul();
            load(code, output[1], ints); if (ints) code.i2l();
            code.ladd();
            load(code, inputLayout.extents[2], ints); if (ints) code.i2l();
            code.lmul();
            load(code, oh, ints); if (ints) code.i2l();
            load(code, m[9], ints); if (ints) code.i2l();
            code.lmul();
            load(code, ow, ints); if (ints) code.i2l();
            code.ladd().ladd().lstore(inputOrdinal);
            CpuLoopEmitter.emitActiveOrdinalGuard(code, 0, inputOrdinal, TypeKind.LONG, nextX);
        }
        loadCarrier(code, carriers, p.type, s.carrierPattern().getFirst(), 0, inputAddress, ints);
        storeValue(code, p.type, right); add(code, p.type, sum, right);
        code.labelBinding(nextX); increment(code, kx, ints); code.branch(Opcode.GOTO, xLoop);
        code.labelBinding(nextY); increment(code, ky, ints); code.branch(Opcode.GOTO, yLoop);
        code.labelBinding(write);
        storeCarrier(code, carriers, p.type, s.carrierPattern().getLast(), 1, outputAddress, sum, ints);
        increment(code, logical, ints);
        compare(code, logical, end, ints, Opcode.IF_ICMPGE, Opcode.IFGE, done);
        advanceCompactOutput(code, outputLayout, inputLayout, output, outputAddress, base, ints);
        code.branch(Opcode.GOTO, outer).labelBinding(done);
    }

    private static void advanceCompactOutput(CodeBuilder code, LayoutLocals out, LayoutLocals in,
            int[] coordinates, int outputAddress, int sourceBase, boolean ints) {
        var complete = code.newLabel();
        for (int axis = 3; axis >= 0; axis--) {
            var carry = code.newLabel(); increment(code, coordinates[axis], ints);
            compare(code, coordinates[axis], out.extents[axis], ints, Opcode.IF_ICMPGE, Opcode.IFGE, carry);
            load(code, outputAddress, ints); load(code, out.strides[axis], ints); addPrimitive(code, ints); store(code, outputAddress, ints);
            if (axis < 2) {
                load(code, sourceBase, ints); load(code, in.strides[axis], ints); addPrimitive(code, ints); store(code, sourceBase, ints);
            }
            code.branch(Opcode.GOTO, complete).labelBinding(carry);
            zero(code, coordinates[axis], ints);
            // Subtract the old coordinate's contribution before carrying: no transient
            // one-past-span address (including at MAX_VALUE) is formed.
            load(code, outputAddress, ints); load(code, out.extents[axis], ints);
            if (ints) code.loadConstant(1).isub(); else code.loadConstant(1L).lsub();
            load(code, out.strides[axis], ints); multiply(code, ints); subtract(code, ints); store(code, outputAddress, ints);
            if (axis < 2) {
                load(code, sourceBase, ints); load(code, out.extents[axis], ints);
                if (ints) code.loadConstant(1).isub(); else code.loadConstant(1L).lsub();
                load(code, in.strides[axis], ints); multiply(code, ints); subtract(code, ints); store(code, sourceBase, ints);
            }
        }
        code.labelBinding(complete);
    }

    private static void inverseCoordinate(CodeBuilder code, int coordinate, int pad, int kernel,
            int dilation, int stride, int limit, int numerator, int quotient, boolean ints, Label skip) {
        load(code, coordinate, ints); load(code, pad, ints); addPrimitive(code, ints);
        load(code, kernel, ints); load(code, dilation, ints); multiply(code, ints); subtract(code, ints);
        store(code, numerator, ints); load(code, numerator, ints);
        if (ints) code.branch(Opcode.IFLT, skip);
        else code.loadConstant(0L).lcmp().branch(Opcode.IFLT, skip);
        load(code, numerator, ints); load(code, stride, ints);
        if (ints) code.irem().branch(Opcode.IFNE, skip);
        else code.lrem().loadConstant(0L).lcmp().branch(Opcode.IFNE, skip);
        load(code, numerator, ints); load(code, stride, ints); divide(code, ints); store(code, quotient, ints);
        compare(code, quotient, limit, ints, Opcode.IF_ICMPGE, Opcode.IFGE, skip);
    }

    private static void emitKernelBroadcastGuarded(CodeBuilder code, CpuKernelSpecialization s, FoldEncoding p) {
        var longs = code.newLabel(); var complete = code.newLabel();
        code.lload(3).lload(5).lcmp().branch(Opcode.IFGE, complete);
        // Every individual cold fact and both entry bounds must be nonnegative/int-valued.
        for (int i = 4; i < p.mapping + 10; i++) requireGeometryAtMost(code, i, Integer.MAX_VALUE, longs);
        code.lload(3).loadConstant(0L).lcmp().branch(Opcode.IFLT, longs);
        code.lload(5).loadConstant((long)Integer.MAX_VALUE).lcmp().branch(Opcode.IFGT, longs);
        // Prove extrema of inverse numerators, not just carrier spans. Long products/sums below
        // are safe by cold geometry proof; their int bounds dominate both emitted inner loops.
        for (int axis = 0; axis < 2; axis++) {
            geometry(code, p.outputLayout + 4 + axis); code.loadConstant(1L).lsub(); geometry(code, p.mapping + 4 + axis);
            code.ladd().loadConstant((long)Integer.MAX_VALUE).lcmp().branch(Opcode.IFGT, longs);
            geometry(code, p.mapping + axis).loadConstant(1L).lsub();
            geometry(code, p.mapping + 6 + axis); code.lmul().loadConstant((long)Integer.MAX_VALUE).lcmp().branch(Opcode.IFGT, longs);
        }
        // Check complete maximum physical addresses, including invocation carrier bases.
        for (int rank : new int[]{3, 4}) {
            int layout = rank == 3 ? p.inputLayout : p.outputLayout;
            geometry(code, layout + 1);
            for (int axis = 0; axis < rank; axis++) {
                geometry(code, layout + 2 + axis).loadConstant(1L).lsub();
                geometry(code, layout + 2 + rank + axis); code.lmul().ladd();
            }
            code.loadConstant((long)Integer.MAX_VALUE).lcmp().branch(Opcode.IFGT, longs);
        }
        emitKernelBroadcast(code, s, p, true);
        code.branch(Opcode.GOTO, complete).labelBinding(longs);
        emitKernelBroadcast(code, s, p, false);
        code.labelBinding(complete);
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
        var write = code.newLabel();
        code.labelBinding(outer).iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        positiveZero(code, DataType.FLOAT32, sum);
        CpuLoopEmitter.emitActiveOrdinalGuard(code, 1, logical, TypeKind.INT, write);
        code.iload(logical).loadConstant(32).idiv().istore(outputY);
        code.iload(logical).iload(outputY).loadConstant(32).imul().isub().istore(outputX);
        code.loadConstant(0).istore(q).labelBinding(qLoop);
        code.iload(q).loadConstant(9).branch(Opcode.IF_ICMPGE, nextQ);
        code.iload(q).loadConstant(3).idiv().istore(kernelY);
        code.iload(q).iload(kernelY).loadConstant(3).imul().isub().istore(kernelX);
        code.loadConstant(0).istore(column).labelBinding(columnLoop);
        code.iload(column).loadConstant(512).branch(Opcode.IF_ICMPGE, nextQ);
        if (CpuCarrierEmitter.controlFacts()
                .map(control -> control.boundaryRows().get(0).axis() >= 0).orElse(false)) {
            int inputOrdinal = code.allocateLocal(TypeKind.INT);
            code.iload(q).loadConstant(512).imul().iload(column).iadd().istore(inputOrdinal);
            CpuLoopEmitter.emitActiveOrdinalGuard(code, 0, inputOrdinal, TypeKind.INT, noMatch);
        }
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
        code.labelBinding(write);
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
        CpuLoopEmitter.emitActiveOrdinalGuard(code, 1, logical, TypeKind.INT, write);
        code.loadConstant(0).istore(inputOrdinal).loadConstant(0).istore(within)
                .loadConstant(0).istore(target).labelBinding(inner);
        code.iload(inputOrdinal).iload(inputCount).branch(Opcode.IF_ICMPGE, write);
        var noMatch = code.newLabel();
        CpuLoopEmitter.emitActiveOrdinalGuard(code, 0, inputOrdinal, TypeKind.INT, noMatch);
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
        CpuLoopEmitter.emitActiveOrdinalGuard(code, 1, logical, addressKind, write);
        for (int coordinate : input) zero(code, coordinate, ints);
        zero(code, inputOrdinal, ints);
        code.labelBinding(inner);
        compare(code, inputOrdinal, inputCount, ints, Opcode.IF_ICMPGE, Opcode.IFGE, write);
        var noMatch = code.newLabel();
        CpuLoopEmitter.emitActiveOrdinalGuard(code, 0, inputOrdinal, addressKind, noMatch);
        code.loadConstant(1).istore(match);
        if (p.family.equals("FOLD_AXIS")) {
            emitAxisMatch(code, p, input, output, match, ints, axisMapping);
        } else {
            emitTwoDimensionalMatch(code, input, output, match, ints, twoDimensionalMapping);
        }
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

    /**
     * Emits one sequential addition in the working type; low operands are already decoded floats.
     * @param code non-null generation-time body builder
     * @param type non-null supported fold boundary type
     * @param left local containing the accumulator, overwritten with the working sum
     * @param right local containing the working contributor, not modified
     * @throws IllegalArgumentException if type is BOOL
     */
    private static void add(CodeBuilder code, DataType type, int left, int right) {
        loadValue(code, type, left); loadValue(code, type, right);
        switch (type) {
            case FLOAT64 -> code.dadd();
            case FLOAT32, BFLOAT16, FLOAT16 -> code.fadd();
            case INT32 -> code.iadd();
            case INT64 -> code.ladd();
            case BOOL -> throw new IllegalArgumentException("BOOL fold is unsupported");
        }
        storeValue(code, type, left);
    }

    /**
     * Loads storage and exactly decodes either low format to FLOAT32 on the operand stack.
     * @param code non-null generation-time body builder
     * @param carriers non-null direct carrier emitter
     * @param type non-null admitted boundary type; storage width remains unchanged
     * @param access non-null prepared carrier form
     * @param parameterSlot local holding the input carrier
     * @param addressLocal local holding the checked element address, not a byte offset
     * @param intAddress whether the address local is int rather than long
     */
    private static void loadCarrier(CodeBuilder code, CpuCarrierEmitter carriers, DataType type,
            CarrierAccess access, int parameterSlot, int addressLocal, boolean intAddress) {
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            carriers.load(type, access, parameterSlot, addressLocal, intAddress);
        } else {
            code.aload(parameterSlot).getstatic(VALUE_LAYOUT, layoutField(type), layoutClass(type));
            byteOffset(code, type, addressLocal, intAddress);
            code.invokeinterface(SEGMENT, "get", MethodTypeDesc.of(primitive(type), layoutClass(type),
                    ConstantDescs.CD_long));
        }
        if (type == DataType.BFLOAT16) {
            code.loadConstant(16).ishl().invokestatic(FLOAT_CLASS, "intBitsToFloat",
                    MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int));
        } else if (type == DataType.FLOAT16) {
            code.i2s().invokestatic(FLOAT_CLASS, "float16ToFloat",
                    MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_short));
        }
    }

    /**
     * Stores one completed result, narrowing a low FLOAT32 accumulator once with ties-to-even.
     * The existing cast emitter is a generation-time collaborator, not a generated helper call.
     * @param code non-null generation-time body builder
     * @param carriers non-null direct carrier emitter
     * @param type non-null admitted output storage type
     * @param access non-null prepared carrier form
     * @param parameterSlot local holding the output carrier
     * @param addressLocal local holding the checked element address, not a byte offset
     * @param valueLocal local containing the completed working accumulator; float for low types
     * @param intAddress whether the address local is int rather than long
     */
    private static void storeCarrier(CodeBuilder code, CpuCarrierEmitter carriers, DataType type,
            CarrierAccess access, int parameterSlot, int addressLocal, int valueLocal,
            boolean intAddress) {
        boolean low = type == DataType.BFLOAT16 || type == DataType.FLOAT16;
        if (low) {
            int represented = code.allocateLocal(TypeKind.INT);
            new CpuCastEmitter(code).emit(DataType.FLOAT32, type, valueLocal);
            code.istore(represented);
            valueLocal = represented;
        }
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            carriers.store(type, access, parameterSlot, addressLocal, valueLocal, intAddress);
            return;
        }
        code.aload(parameterSlot).getstatic(VALUE_LAYOUT, layoutField(type), layoutClass(type));
        byteOffset(code, type, addressLocal, intAddress);
        if (low) code.iload(valueLocal); else loadValue(code, type, valueLocal);
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
            case BFLOAT16, FLOAT16 -> "JAVA_SHORT_UNALIGNED";
            case INT32 -> "JAVA_INT_UNALIGNED";
            case INT64 -> "JAVA_LONG_UNALIGNED";
            case BOOL -> throw new IllegalArgumentException("BOOL fold is unsupported");
        };
    }

    private static ClassDesc layoutClass(DataType type) {
        return switch (type) {
            case FLOAT64 -> DOUBLE_LAYOUT;
            case FLOAT32 -> FLOAT_LAYOUT;
            case BFLOAT16, FLOAT16 -> SHORT_LAYOUT;
            case INT32 -> INT_LAYOUT;
            case INT64 -> LONG_LAYOUT;
            case BOOL -> throw new IllegalArgumentException("BOOL fold is unsupported");
        };
    }

    private static ClassDesc primitive(DataType type) {
        return switch (type) {
            case FLOAT64 -> ConstantDescs.CD_double;
            case FLOAT32 -> ConstantDescs.CD_float;
            case BFLOAT16, FLOAT16 -> ConstantDescs.CD_short;
            case INT32 -> ConstantDescs.CD_int;
            case INT64 -> ConstantDescs.CD_long;
            case BOOL -> throw new IllegalArgumentException("BOOL fold is unsupported");
        };
    }

    /**
     * Initializes the owned cell's working accumulator, including gaps with no contributors.
     * @param code non-null generation-time body builder
     * @param type non-null supported fold boundary type
     * @param local accumulator local; float for either low format
     * @throws IllegalArgumentException if type is BOOL
     */
    private static void positiveZero(CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.loadConstant(0.0d).dstore(local);
            case FLOAT32, BFLOAT16, FLOAT16 -> code.loadConstant(0.0f).fstore(local);
            case INT64 -> code.loadConstant(0L).lstore(local);
            case INT32 -> code.loadConstant(0).istore(local);
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

    /**
     * Reloads a working value without a storage conversion.
     * @param code non-null generation-time body builder
     * @param type non-null boundary type selecting its working local kind
     * @param local working value local; float for either low format
     */
    private static void loadValue(CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dload(local);
            case FLOAT32, BFLOAT16, FLOAT16 -> code.fload(local);
            case INT64 -> code.lload(local);
            case INT32, BOOL -> code.iload(local);
        }
    }

    /**
     * Saves a working value without narrowing a low contribution or accumulator.
     * @param code non-null generation-time body builder
     * @param type non-null boundary type selecting its working local kind
     * @param local destination working local; float for either low format
     */
    private static void storeValue(CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dstore(local);
            case FLOAT32, BFLOAT16, FLOAT16 -> code.fstore(local);
            case INT64 -> code.lstore(local);
            case INT32, BOOL -> code.istore(local);
        }
    }

    /**
     * Distinguishes arithmetic locals from the low formats' short storage carriers.
     * @param type non-null fold boundary type
     * @return non-null primitive working kind; FLOAT for FLOAT32, BFLOAT16 and FLOAT16
     */
    private static TypeKind localKind(DataType type) {
        return switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE;
            case FLOAT32, BFLOAT16, FLOAT16 -> TypeKind.FLOAT;
            case INT64 -> TypeKind.LONG;
            case INT32, BOOL -> TypeKind.INT;
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
