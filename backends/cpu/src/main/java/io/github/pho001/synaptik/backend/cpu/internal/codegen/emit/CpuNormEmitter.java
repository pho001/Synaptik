package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;

/** Emits direct L1/L2 norm bodies and exposes only shared primitive traversal mechanics. */
public final class CpuNormEmitter {
    private static final ClassDesc MATH = ClassDesc.of(Math.class.getName());
    private static final ClassDesc FLOAT = ClassDesc.of(Float.class.getName());
    private static final ClassDesc DOUBLE = ClassDesc.of(Double.class.getName());

    /** Creates a stateless generation-time emitter. */
    public CpuNormEmitter() { }

    /**
     * Emits one direct typed norm entry body.
     * @param code non-null Class-File method builder to mutate
     * @param specialization non-null exact carrier and optional scratch specialization
     * @param ir non-null canonical L1 or L2 identity
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        boolean l1 = ir.familyIdentity().startsWith("advanced-reduction:L1_NORM:");
        if (!l1 && !ir.familyIdentity().startsWith("advanced-reduction:L2_NORM:")) {
            throw new IllegalArgumentException("norm emitter requires L1_NORM or L2_NORM");
        }
        if (specialization.carrierPattern().size() != 2
                || specialization.boundaryDataTypes().size() != 2
                || specialization.boundaryDataTypes().getFirst()
                    != specialization.boundaryDataTypes().getLast()
                || !ir.familyIdentity().startsWith("advanced-reduction:")) {
            throw new IllegalArgumentException("advanced reduction requires two matching boundaries");
        }
        DataType type = specialization.boundaryDataTypes().getFirst();
        int inputRank = ir.values().getFirst().accessPlan().iterationRank();
        int outputRank = ir.values().getLast().accessPlan().iterationRank();
        boolean keep = ir.familyIdentity().contains(":keep=true:");
        boolean[] selected = selected(ir.familyIdentity(), inputRank);
        boolean contiguousSelected = contiguousSelected(selected,
                ir.values().getFirst().accessPlan().contiguousSuffix());
        int scratch = specialization.scratchParameter() ? 2 : -1;
        int geometry = specialization.scratchParameter() ? 3 : 2;
        int start = geometry + 1, end = start + 2;
        var carriers = new CpuCarrierEmitter(code);
        int cell = code.allocateLocal(TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.LONG);
        int coordinate = code.allocateLocal(TypeKind.LONG);
        int inputBase = code.allocateLocal(TypeKind.LONG);
        int outputAddress = code.allocateLocal(TypeKind.LONG);
        int domain = code.allocateLocal(TypeKind.LONG);
        int address = code.allocateLocal(TypeKind.LONG);
        int value = code.allocateLocal(TypeKind.DOUBLE);
        int result = code.allocateLocal(TypeKind.DOUBLE);
        int temporary = code.allocateLocal(TypeKind.DOUBLE);
        int scale = code.allocateLocal(TypeKind.DOUBLE);
        int squares = code.allocateLocal(TypeKind.DOUBLE);
        int nan = code.allocateLocal(TypeKind.INT);
        int anyInfinity = code.allocateLocal(TypeKind.INT);
        int inputOffsetIndex = 0;
        int inputExtentsIndex = 11;
        int inputStridesIndex = inputExtentsIndex + inputRank;
        int outputOffsetIndex = 1;
        int outputExtentsIndex = inputStridesIndex + inputRank;
        int outputStridesIndex = outputExtentsIndex + outputRank;
        TypeKind representedKind = type == DataType.FLOAT64 ? TypeKind.DOUBLE
                : type == DataType.FLOAT32 ? TypeKind.FLOAT : TypeKind.INT;
        int represented = code.allocateLocal(representedKind);
        CpuExactSumEmitter exact = l1 ? new CpuExactSumEmitter(code, type, false, scratch, geometry,
                Math.toIntExact(longAfter(ir.familyIdentity(), ":slice=")
                        / Long.BYTES - 1)) : null;

        code.lload(start).lstore(cell);
        var cells = code.newLabel(); var done = code.newLabel();
        code.labelBinding(cells).lload(cell).lload(end).lcmp().branch(Opcode.IFGE, done);
        geometry(code, geometry, inputOffsetIndex).lstore(inputBase);
        geometry(code, geometry, outputOffsetIndex).lstore(outputAddress);
        code.lload(cell).lstore(remaining);
        int[] outputCoordinates = new int[outputRank];
        for (int axis = outputRank - 1; axis >= 0; axis--) {
            int local = code.allocateLocal(TypeKind.LONG); outputCoordinates[axis] = local;
            if (axis == 0) code.lload(remaining).lstore(local);
            else {
                code.lload(remaining); geometry(code, geometry, outputExtentsIndex + axis)
                        .lrem().lstore(local);
                code.lload(remaining); geometry(code, geometry, outputExtentsIndex + axis)
                        .ldiv().lstore(remaining);
            }
            code.lload(outputAddress).lload(local);
            geometry(code, geometry, outputStridesIndex + axis).lmul().ladd()
                    .lstore(outputAddress);
        }
        for (int inputAxis = 0, outputAxis = 0; inputAxis < inputRank; inputAxis++) {
            if (selected[inputAxis]) { if (keep) outputAxis++; continue; }
            code.lload(inputBase).lload(outputCoordinates[outputAxis++]);
            geometry(code, geometry, inputStridesIndex + inputAxis).lmul().ladd()
                    .lstore(inputBase);
        }
        int[] selectedCoordinates = new int[inputRank];
        for (int axis = 0; axis < inputRank; axis++) if (selected[axis])
            selectedCoordinates[axis] = code.allocateLocal(TypeKind.LONG);
        code.loadConstant(0).istore(nan).loadConstant(0).istore(anyInfinity);
        code.loadConstant(0.0).dstore(squares).loadConstant(0.0).dstore(scale);
        if (exact != null) exact.emitReset();

        emitPass(code, carriers, specialization, type, selected, inputRank, geometry,
                inputExtentsIndex, inputStridesIndex, inputBase, domain, remaining, coordinate,
                address, selectedCoordinates, contiguousSelected, represented, value, () -> {
                    classify(code, value, nan, -1, anyInfinity);
                    if (l1) {
                        emitRepresentedAbs(code, type, represented);
                        exact.emitFactor(represented);
                    } else {
                        emitScaledSquares(code, value, scale, squares, temporary);
                    }
                });

        if (l1) {
            exact.emitFinish(represented);
            decodeRepresented(code, type, represented, result);
        } else code.dload(scale).dload(squares)
                .invokestatic(MATH, "sqrt", doubleUnary()).dmul().dstore(result);
        var noNan = code.newLabel(); var noInfinity = code.newLabel(); var specialsDone = code.newLabel();
        code.iload(nan).branch(Opcode.IFEQ, noNan).loadConstant(Double.longBitsToDouble(
                0x7ff8000000000000L)).dstore(result).branch(Opcode.GOTO, specialsDone)
                .labelBinding(noNan).iload(anyInfinity).branch(Opcode.IFEQ, noInfinity)
                .loadConstant(Double.POSITIVE_INFINITY).dstore(result)
                .branch(Opcode.GOTO, specialsDone).labelBinding(noInfinity)
                .labelBinding(specialsDone);
        emitStore(code, carriers, specialization, type, outputAddress, result);
        code.lload(cell).loadConstant(1L).ladd().lstore(cell)
                .branch(Opcode.GOTO, cells).labelBinding(done);
    }

    /**
     * Emits the common primitive selected-domain traversal and invokes a generation-time body.
     * @param code non-null method builder
     * @param carriers non-null typed carrier emitter
     * @param specialization non-null boundary specialization
     * @param type represented floating type
     * @param selected selected input-axis membership
     * @param rank input rank
     * @param geometry geometry-argument local
     * @param extents packed input-extents offset
     * @param strides packed input-strides offset
     * @param inputBase input-cell base local
     * @param domain domain-ordinal local
     * @param remaining coordinate-work local
     * @param coordinate reserved primitive coordinate local
     * @param address current input-address local
     * @param coordinates selected-axis coordinate locals
     * @param contiguousSelected whether the selected domain is one contiguous suffix
     * @param represented represented-value local
     * @param value binary64 decoded-value local
     * @param body non-null generation-time numerical body emitter
     */
    static void emitPass(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, boolean[] selected,
            int rank, int geometry, int extents, int strides, int inputBase, int domain,
            int remaining, int coordinate, int address, int[] coordinates,
            boolean contiguousSelected, int represented, int value, Runnable body) {
        if (contiguousSelected && specialization.carrierPattern().getFirst()
                != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) {
            int intAddress = code.allocateLocal(TypeKind.INT);
            int intEnd = code.allocateLocal(TypeKind.INT);
            code.lload(inputBase).l2i().istore(intAddress).iload(intAddress);
            geometry(code, geometry, 2).l2i().iadd().istore(intEnd);
            var intLoop = code.newLabel(); var intFinish = code.newLabel();
            code.labelBinding(intLoop).iload(intAddress).iload(intEnd)
                    .branch(Opcode.IF_ICMPGE, intFinish);
            carriers.load(type, specialization.carrierPattern().getFirst(), 0, intAddress, true);
            if (type == DataType.FLOAT64) code.dstore(represented);
            else if (type == DataType.FLOAT32) code.fstore(represented);
            else code.istore(represented);
            decodeRepresented(code, type, represented, value);
            body.run();
            code.iinc(intAddress, 1).branch(Opcode.GOTO, intLoop).labelBinding(intFinish);
            return;
        }
        code.loadConstant(0L).lstore(domain).lload(inputBase).lstore(address);
        for (int axis = 0; axis < rank; axis++) if (selected[axis])
            code.loadConstant(0L).lstore(coordinates[axis]);
        var loop = code.newLabel(); var finish = code.newLabel();
        code.labelBinding(loop).lload(domain); geometry(code, geometry, 2).lcmp()
                .branch(Opcode.IFGE, finish);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, address, false);
        if (type == DataType.FLOAT64) code.dstore(represented);
        else if (type == DataType.FLOAT32) code.fstore(represented);
        else code.istore(represented);
        decodeRepresented(code, type, represented, value);
        body.run();
        code.lload(domain).loadConstant(1L).ladd().lstore(domain);
        if (contiguousSelected) {
            code.lload(address).loadConstant(1L).ladd().lstore(address)
                    .branch(Opcode.GOTO, loop).labelBinding(finish);
            return;
        }
        var advanced = code.newLabel();
        for (int axis = rank - 1; axis >= 0; axis--) if (selected[axis]) {
            code.lload(coordinates[axis]).loadConstant(1L).ladd().lstore(coordinates[axis]);
            code.lload(address); geometry(code, geometry, strides + axis).ladd().lstore(address);
            code.lload(coordinates[axis]); geometry(code, geometry, extents + axis).lcmp()
                    .branch(Opcode.IFLT, advanced);
            code.loadConstant(0L).lstore(coordinates[axis]);
            code.lload(address); geometry(code, geometry, extents + axis);
            geometry(code, geometry, strides + axis).lmul().lsub().lstore(address);
        }
        code.labelBinding(advanced).branch(Opcode.GOTO, loop).labelBinding(finish);
    }

    /**
     * Emits primitive NaN and infinity classification without changing the represented value.
     * @param code non-null method builder
     * @param value binary64 value local
     * @param nan NaN flag local
     * @param positive positive-infinity flag local, or negative to omit that flag
     * @param infinity either-infinity flag local
     */
    static void classify(CodeBuilder code, int value, int nan, int positive, int infinity) {
        var notNan = code.newLabel(); var notPositive = code.newLabel(); var finite = code.newLabel();
        code.dload(value).invokestatic(DOUBLE, "isNaN", MethodTypeDesc.of(
                TypeKind.BOOLEAN.upperBound(), TypeKind.DOUBLE.upperBound()))
                .branch(Opcode.IFEQ, notNan).loadConstant(1).istore(nan)
                .branch(Opcode.GOTO, finite).labelBinding(notNan);
        code.dload(value).loadConstant(Double.POSITIVE_INFINITY).dcmpl()
                .branch(Opcode.IFNE, notPositive);
        if (positive >= 0) code.loadConstant(1).istore(positive);
        code
                .loadConstant(1).istore(infinity).branch(Opcode.GOTO, finite)
                .labelBinding(notPositive);
        code.dload(value).loadConstant(Double.NEGATIVE_INFINITY).dcmpl()
                .branch(Opcode.IFNE, finite).loadConstant(1).istore(infinity)
                .labelBinding(finite);
    }

    /**
     * Emits one compensated binary64 addition step.
     * @param code non-null method builder
     * @param value addend local, reused for the compensated addend
     * @param sum running-sum local
     * @param compensation running-compensation local
     * @param temporary temporary sum local
     */
    static void kahan(CodeBuilder code, int value, int sum, int compensation, int temporary) {
        code.dload(value).dload(compensation).dsub().dstore(value);
        code.dload(sum).dload(value).dadd().dstore(temporary);
        code.dload(temporary).dload(sum).dsub().dload(value).dsub().dstore(compensation);
        code.dload(temporary).dstore(sum);
    }

    /**
     * Emits exact represented FLOAT64/FLOAT32/BFLOAT16 decoding to binary64.
     * @param code non-null method builder
     * @param type represented floating type
     * @param represented source local in its carrier-compatible primitive kind
     * @param target binary64 destination local
     */
    static void decodeRepresented(CodeBuilder code, DataType type, int represented,
            int target) {
        if (type == DataType.FLOAT64) code.dload(represented).dstore(target);
        else if (type == DataType.FLOAT32) code.fload(represented).f2d().dstore(target);
        else code.iload(represented).loadConstant(16).ishl().invokestatic(FLOAT, "intBitsToFloat",
                MethodTypeDesc.of(TypeKind.FLOAT.upperBound(), TypeKind.INT.upperBound()))
                .f2d().dstore(target);
    }

    private static void emitRepresentedAbs(CodeBuilder code, DataType type, int represented) {
        if (type == DataType.FLOAT64) code.dload(represented).invokestatic(MATH, "abs",
                doubleUnary()).dstore(represented);
        else if (type == DataType.FLOAT32) code.fload(represented).invokestatic(MATH, "abs",
                MethodTypeDesc.of(TypeKind.FLOAT.upperBound(), TypeKind.FLOAT.upperBound()))
                .fstore(represented);
        else code.iload(represented).loadConstant(0x7fff).iand().istore(represented);
    }

    private static void emitScaledSquares(CodeBuilder code, int value, int scale, int squares,
            int temporary) {
        code.dload(value).invokestatic(MATH, "abs", doubleUnary()).dstore(value);
        var zero = code.newLabel(); var smaller = code.newLabel(); var complete = code.newLabel();
        code.dload(value).loadConstant(0.0).dcmpl().branch(Opcode.IFEQ, zero);
        code.dload(scale).dload(value).dcmpl().branch(Opcode.IFGE, smaller);
        code.dload(scale).dload(value).ddiv().dstore(temporary);
        code.loadConstant(1.0).dload(squares).dload(temporary).dload(temporary).dmul()
                .dmul().dadd().dstore(squares).dload(value).dstore(scale)
                .branch(Opcode.GOTO, complete).labelBinding(smaller);
        code.dload(value).dload(scale).ddiv().dstore(temporary);
        code.dload(squares).dload(temporary).dload(temporary).dmul().dadd().dstore(squares)
                .branch(Opcode.GOTO, complete).labelBinding(zero).labelBinding(complete);
    }

    /**
     * Emits the single final result-format narrowing and typed store.
     * @param code non-null method builder
     * @param carriers non-null typed carrier emitter
     * @param specialization non-null output-carrier specialization
     * @param type represented result type
     * @param address output element-address local
     * @param result binary64 result local
     */
    static void emitStore(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int address, int result) {
        int represented = result;
        if (type == DataType.FLOAT32) {
            represented = code.allocateLocal(TypeKind.FLOAT);
            code.dload(result).d2f().fstore(represented);
        } else if (type == DataType.BFLOAT16) {
            represented = code.allocateLocal(TypeKind.INT);
            int bits = code.allocateLocal(TypeKind.INT);
            code.dload(result).d2f().invokestatic(FLOAT, "floatToRawIntBits",
                    MethodTypeDesc.of(TypeKind.INT.upperBound(), TypeKind.FLOAT.upperBound()))
                    .istore(bits);
            var finite = code.newLabel(); var rounded = code.newLabel();
            code.iload(bits).loadConstant(0x7fffffff).iand().loadConstant(0x7f800000)
                    .branch(Opcode.IF_ICMPLE, finite).loadConstant(0x7fc0)
                    .istore(represented).branch(Opcode.GOTO, rounded).labelBinding(finite)
                    .iload(bits).loadConstant(0x7fff).iadd().iload(bits).loadConstant(16).iushr()
                    .loadConstant(1).iand().iadd().loadConstant(16).iushr()
                    .istore(represented).labelBinding(rounded);
        }
        carriers.store(type, specialization.carrierPattern().getLast(), 1, address, represented, false);
    }

    /**
     * Decodes immutable selected-axis flags from the canonical structural identity.
     * @param identity non-null canonical advanced-reduction identity
     * @param rank non-negative input rank
     * @return a new membership array of length {@code rank}
     */
    static boolean[] selected(String identity, int rank) {
        int begin = identity.indexOf(":selected=[") + 11;
        int end = identity.indexOf("]:keep=", begin);
        String[] values = identity.substring(begin, end).split(", ");
        boolean[] result = new boolean[rank];
        for (int i = 0; i < rank; i++) result[i] = Boolean.parseBoolean(values[i]);
        return result;
    }

    /**
     * Determines whether selected axes form a physically contiguous input suffix.
     * @param selected non-null input-axis membership
     * @param contiguousSuffixRank number of structurally contiguous trailing input axes
     * @return {@code true} when direct unit-stride selected-domain traversal is valid
     */
    static boolean contiguousSelected(boolean[] selected, int contiguousSuffixRank) {
        int first = selected.length;
        while (first > 0 && selected[first - 1]) first--;
        for (int axis = 0; axis < first; axis++) if (selected[axis]) return false;
        return selected.length - first <= contiguousSuffixRank;
    }

    /**
     * Reads one generation-time long fact from the canonical structural identity.
     * @param identity non-null canonical identity
     * @param marker non-null field marker including its leading separator
     * @return the parsed structural value
     * @throws NumberFormatException if the encoded value is not a long
     */
    static long longAfter(String identity, String marker) {
        int begin = identity.indexOf(marker) + marker.length();
        int end = identity.indexOf(':', begin);
        if (end < 0) end = identity.length();
        return Long.parseLong(identity.substring(begin, end));
    }

    /**
     * Emits one packed long-geometry load.
     * @param code non-null method builder
     * @param slot geometry-array local
     * @param index non-negative packed index
     * @return the same builder with the long value on its operand stack
     */
    static CodeBuilder geometry(CodeBuilder code, int slot, int index) {
        return code.aload(slot).loadConstant(index).laload();
    }

    /**
     * Returns the primitive binary64 unary method descriptor used by {@link Math} calls.
     * @return non-null {@code (double)double} descriptor
     */
    static MethodTypeDesc doubleUnary() {
        return MethodTypeDesc.of(TypeKind.DOUBLE.upperBound(), TypeKind.DOUBLE.upperBound());
    }
}
