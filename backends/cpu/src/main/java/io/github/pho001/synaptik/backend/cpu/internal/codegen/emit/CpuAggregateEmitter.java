package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ConstantDescs;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Emits typed ordinary aggregate folds directly into generated CPU entries.
 *
 * <p>The static body reduces complete output cells only. It traverses every selected domain in
 * logical input row-major order. Numerical floating rows update an exact primitive-limb state and
 * round once; extrema retain first-NaN and signed-zero selection. Extrema and Boolean combination
 * are emitted directly, so the generated per-element body calls no Synaptik runtime helper. The
 * body allocates no per-cell or per-element object. Dense heap-array
 * reductions use one typed integer-address fold; other forms embed a typed long-address fallback
 * that decodes output and selected-domain coordinates without runtime type, kind, form, or
 * carrier dispatch.</p>
 */
public final class CpuAggregateEmitter {
    private static final DataType[] TYPES = DataType.values();
    /** Creates a stateless typed-body emitter. */
    public CpuAggregateEmitter() { }

    /**
     * Emits a typed aggregate body whose hot loops contain no runtime semantic dispatch.
     *
     * @param code non-null Class-File method builder mutated with the generated fold
     * @param specialization non-null two-boundary carrier/type specialization, scratch-bearing
     *     exactly for floating numerical aggregates
     * @param ir non-null canonical aggregate IR supplying kind, form, axes, rank, and access facts
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the specialization has another boundary shape
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        if (specialization.carrierPattern().size() != 2)
            throw new IllegalArgumentException("aggregate requires two boundaries");
        String identity = ir.familyIdentity();
        DataType type = specialization.boundaryDataTypes().getFirst();
        int kind = identity.startsWith("aggregate:MIN:") ? 0
                : identity.startsWith("aggregate:MAX:") ? 1
                : identity.startsWith("aggregate:ALL:") ? 2
                : identity.startsWith("aggregate:ANY:") ? 3
                : identity.startsWith("aggregate:SUM:") ? 4
                : identity.startsWith("aggregate:MEAN:") ? 5 : 6;
        boolean exactFloating = kind >= 4 && (type == DataType.FLOAT64
                || type == DataType.FLOAT32 || type == DataType.BFLOAT16);
        int exactLimbs = exactFloating ? identityNumber(identity, ":limbs=") : 0;
        if (specialization.scratchParameter() != exactFloating)
            throw new IllegalArgumentException("aggregate scratch shape disagrees with numerical kind");
        int geometrySlot = exactFloating ? 3 : 2;
        int startSlot = exactFloating ? 4 : 3;
        int endSlot = exactFloating ? 6 : 5;
        int inRank = ir.values().getFirst().accessPlan().iterationRank();
        int outRank = ir.values().getLast().accessPlan().iterationRank();
        boolean fullDenseArrays = identity.contains(":FULL:")
                && specialization.carrierPattern().stream().noneMatch(
                    access -> access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                && ir.values().getFirst().accessPlan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR;
        boolean denseTrailingArrays = !identity.contains(":FULL:")
                && specialization.carrierPattern().stream().noneMatch(
                    access -> access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                && ir.values().getFirst().accessPlan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR
                && ir.values().getLast().accessPlan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR
                && selectedSuffix(identity, inRank);
        if (fullDenseArrays) emitFullDense(code, specialization, type, kind, inRank,
                exactFloating, exactLimbs, geometrySlot, startSlot, endSlot);
        else if (denseTrailingArrays) emitDenseCells(code, specialization, type, kind, inRank,
                outRank, exactFloating, exactLimbs, geometrySlot, startSlot, endSlot);
        else emitGeneral(code, specialization, type, kind, identity, inRank, outRank,
                exactFloating, exactLimbs, geometrySlot, startSlot, endSlot);
    }

    private static boolean selectedSuffix(String identity, int rank) {
        boolean[] axes = selectedAxes(identity, rank); boolean selected = false;
        for (boolean axis : axes) { if (axis) selected = true; else if (selected) return false; }
        return selected;
    }

    private static void emitDenseCells(CodeBuilder code, CpuKernelSpecialization specialization,
            DataType type, int kind, int inRank, int outRank, boolean exactFloating,
            int exactLimbs, int geometrySlot, int startSlot, int endSlot) {
        int inputLayout = 11 + 2 * inRank + outRank;
        int outputLayout = inputLayout + 2 + 2 * inRank;
        int inputBase = code.allocateLocal(TypeKind.INT), outputBase = code.allocateLocal(TypeKind.INT);
        code.aload(geometrySlot).loadConstant(inputLayout + 1).laload().l2i().istore(inputBase);
        code.aload(geometrySlot).loadConstant(outputLayout + 1).laload().l2i().istore(outputBase);
        int cell = code.allocateLocal(TypeKind.INT); code.lload(startSlot).l2i().istore(cell);
        var done = code.newLabel(); var cells = code.newLabel();
        code.iload(cell).lload(endSlot).l2i().branch(Opcode.IF_ICMPGE, done).labelBinding(cells);
        int accumulator = code.allocateLocal(localKind(type));
        CpuExactSumEmitter exactSum = exactFloating && kind != 6
                ? new CpuExactSumEmitter(code, type, kind == 5, 2, geometrySlot, exactLimbs) : null;
        CpuExactProductEmitter exactProduct = exactFloating && kind == 6
                ? new CpuExactProductEmitter(code, type, 2, geometrySlot, 10, 9, true) : null;
        if (exactFloating) { emitFloatingZero(code, type); store(code, type, accumulator); }
        if (exactSum != null) exactSum.emitReset();
        else if (exactProduct != null) exactProduct.emitReset();
        else { emitIdentity(code, type, kind); store(code, type, accumulator); }
        int domain = code.allocateLocal(TypeKind.INT); code.loadConstant(0).istore(domain);
        var write = code.newLabel();
        code.aload(geometrySlot).loadConstant(7).laload().loadConstant(0L).lcmp()
                .branch(Opcode.IFEQ, write);
        var domains = code.newLabel(); code.labelBinding(domains);
        int address = code.allocateLocal(TypeKind.INT);
        code.iload(inputBase).iload(cell).aload(geometrySlot).loadConstant(7).laload().l2i()
                .imul().iadd().iload(domain).iadd().istore(address);
        int value = code.allocateLocal(localKind(type)); var carriers = new CpuCarrierEmitter(code);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, address, true);
        store(code, type, value);
        if (exactSum != null) exactSum.emitFactor(value);
        else if (exactProduct != null) exactProduct.emitFactor(value);
        else emitApply(code, type, kind, accumulator, value);
        if (type == DataType.BOOL) { load(code, type, accumulator);
            code.branch(kind == 2 ? Opcode.IFEQ : Opcode.IFNE, write); }
        code.iinc(domain, 1); code.iload(domain).aload(geometrySlot).loadConstant(7).laload().l2i()
                .branch(Opcode.IF_ICMPLT, domains).labelBinding(write);
        if (exactSum != null) exactSum.emitFinish(accumulator);
        else if (exactProduct != null) { int found = code.allocateLocal(TypeKind.INT);
            code.loadConstant(1).istore(found); exactProduct.emitFinish(accumulator, found); }
        int output = code.allocateLocal(TypeKind.INT); code.iload(outputBase).iload(cell).iadd()
                .istore(output);
        carriers.store(type, specialization.carrierPattern().getLast(), 1, output, accumulator, true);
        code.iinc(cell, 1); code.iload(cell).lload(endSlot).l2i().branch(Opcode.IF_ICMPLT, cells)
                .labelBinding(done);
    }

    private static void emitFullDense(CodeBuilder code, CpuKernelSpecialization specialization,
            DataType type, int kind, int inRank, boolean exactFloating, int exactLimbs,
            int geometrySlot, int startSlot, int endSlot) {
        var done = code.newLabel();
        code.lload(startSlot).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        int inputLayout = 11 + 2 * inRank;
        int outputLayout = inputLayout + 2 + 2 * inRank;
        int input = code.allocateLocal(TypeKind.INT), output = code.allocateLocal(TypeKind.INT);
        code.aload(geometrySlot).loadConstant(inputLayout + 1).laload().l2i().istore(input);
        code.aload(geometrySlot).loadConstant(outputLayout + 1).laload().l2i().istore(output);
        int accumulator = code.allocateLocal(localKind(type));
        CpuExactSumEmitter exactSum = exactFloating && kind != 6
                ? new CpuExactSumEmitter(code, type, kind == 5, 2, geometrySlot, exactLimbs) : null;
        CpuExactProductEmitter exactProduct = exactFloating && kind == 6
                ? new CpuExactProductEmitter(code, type, 2, geometrySlot, 10, 9, true) : null;
        if (exactFloating) { emitFloatingZero(code, type); store(code, type, accumulator); }
        if (exactSum != null) exactSum.emitReset();
        else if (exactProduct != null) exactProduct.emitReset();
        else { emitIdentity(code, type, kind); store(code, type, accumulator); }
        int domain = code.allocateLocal(TypeKind.INT); code.loadConstant(0).istore(domain);
        var store = code.newLabel();
        code.aload(geometrySlot).loadConstant(7).laload().loadConstant(0L).lcmp()
                .branch(Opcode.IFEQ, store);
        var loop = code.newLabel(); code.labelBinding(loop);
        int address = code.allocateLocal(TypeKind.INT);
        code.iload(input).iload(domain).iadd().istore(address);
        int value = code.allocateLocal(localKind(type));
        var carriers = new CpuCarrierEmitter(code);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, address, true);
        store(code, type, value);
        if (exactSum != null) exactSum.emitFactor(value);
        else if (exactProduct != null) exactProduct.emitFactor(value);
        else emitApply(code, type, kind, accumulator, value);
        if (type == DataType.BOOL) {
            load(code, type, accumulator);
            code.branch(kind == 2 ? Opcode.IFEQ : Opcode.IFNE, store);
        }
        code.iinc(domain, 1); code.iload(domain).aload(geometrySlot).loadConstant(7).laload().l2i()
                .branch(Opcode.IF_ICMPLT, loop);
        code.labelBinding(store);
        if (exactSum != null) exactSum.emitFinish(accumulator);
        else if (exactProduct != null) {
            int found = code.allocateLocal(TypeKind.INT); code.loadConstant(1).istore(found);
            exactProduct.emitFinish(accumulator, found);
        }
        carriers.store(type, specialization.carrierPattern().getLast(), 1, output, accumulator, true);
        code.labelBinding(done);
    }

    private static void emitGeneral(CodeBuilder code, CpuKernelSpecialization specialization,
            DataType type, int kind, String identity, int inRank, int outRank,
            boolean exactFloating, int exactLimbs, int geometrySlot, int startSlot, int endSlot) {
        boolean keep = identity.contains(":keep=true:");
        boolean[] selectedAxes = selectedAxes(identity, inRank);
        int selected = 11, inputCoordinates = selected + inRank;
        int outputCoordinates = inputCoordinates + inRank;
        int inputLayout = outputCoordinates + outRank;
        int outputLayout = inputLayout + 2 + 2 * inRank;
        int cell = code.allocateLocal(TypeKind.LONG);
        code.lload(startSlot).lstore(cell);
        var done = code.newLabel();
        code.lload(cell).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        var cells = code.newLabel(); code.labelBinding(cells);
        decode(code, geometrySlot, cell, outputCoordinates, outputLayout, outRank);
        int outAxis = 0;
        for (int axis = 0; axis < inRank; axis++) {
            code.aload(geometrySlot).loadConstant(inputCoordinates + axis);
            if (selectedAxes[axis]) code.loadConstant(0L);
            else code.aload(geometrySlot).loadConstant(
                    outputCoordinates + (keep ? axis : outAxis++)).laload();
            code.lastore();
        }
        int accumulator = code.allocateLocal(localKind(type));
        CpuExactSumEmitter exactSum = exactFloating && kind != 6
                ? new CpuExactSumEmitter(code, type, kind == 5, 2, geometrySlot, exactLimbs) : null;
        CpuExactProductEmitter exactProduct = exactFloating && kind == 6
                ? new CpuExactProductEmitter(code, type, 2, geometrySlot, 10, 9, true) : null;
        if (exactFloating) { emitFloatingZero(code, type); store(code, type, accumulator); }
        if (exactSum != null) exactSum.emitReset();
        else if (exactProduct != null) exactProduct.emitReset();
        else { emitIdentity(code, type, kind); store(code, type, accumulator); }
        int domain = code.allocateLocal(TypeKind.LONG); code.loadConstant(0L).lstore(domain);
        var write = code.newLabel();
        code.aload(geometrySlot).loadConstant(7).laload().loadConstant(0L).lcmp()
                .branch(Opcode.IFEQ, write);
        var domains = code.newLabel(); code.labelBinding(domains);
        int remaining = code.allocateLocal(TypeKind.LONG); code.lload(domain).lstore(remaining);
        for (int axis = inRank - 1; axis >= 0; axis--) if (selectedAxes[axis]) {
            code.aload(geometrySlot).loadConstant(inputCoordinates + axis).lload(remaining)
                    .aload(geometrySlot).loadConstant(inputLayout + 2 + axis).laload().lrem().lastore();
            code.lload(remaining).aload(geometrySlot).loadConstant(inputLayout + 2 + axis).laload()
                    .ldiv().lstore(remaining);
        }
        int inputAddress = address(code, geometrySlot, inRank, inputCoordinates, inputLayout);
        int value = code.allocateLocal(localKind(type));
        var carriers = new CpuCarrierEmitter(code);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, inputAddress);
        store(code, type, value);
        if (exactSum != null) exactSum.emitFactor(value);
        else if (exactProduct != null) exactProduct.emitFactor(value);
        else emitApply(code, type, kind, accumulator, value);
        if (type == DataType.BOOL) {
            load(code, type, accumulator);
            code.branch(kind == 2 ? Opcode.IFEQ : Opcode.IFNE, write);
        }
        code.lload(domain).loadConstant(1L).ladd().lstore(domain);
        code.lload(domain).aload(geometrySlot).loadConstant(7).laload().lcmp()
                .branch(Opcode.IFLT, domains);
        code.labelBinding(write);
        if (exactSum != null) exactSum.emitFinish(accumulator);
        else if (exactProduct != null) {
            int found = code.allocateLocal(TypeKind.INT); code.loadConstant(1).istore(found);
            exactProduct.emitFinish(accumulator, found);
        }
        int outputAddress = address(code, geometrySlot, outRank, outputCoordinates, outputLayout);
        carriers.store(type, specialization.carrierPattern().getLast(), 1, outputAddress, accumulator);
        code.lload(cell).loadConstant(1L).ladd().lstore(cell);
        code.lload(cell).lload(endSlot).lcmp().branch(Opcode.IFLT, cells);
        code.labelBinding(done);
    }

    private static boolean[] selectedAxes(String identity, int rank) {
        boolean[] result = new boolean[rank];
        int start = identity.indexOf(":axes=[") + 7, end = identity.indexOf(']', start);
        String body = identity.substring(start, end).trim();
        if (!body.isEmpty()) for (String axis : body.split(", ")) result[Integer.parseInt(axis)] = true;
        return result;
    }

    private static int identityNumber(String identity, String marker) {
        int start = identity.indexOf(marker) + marker.length();
        int end = identity.indexOf(':', start);
        return Integer.parseInt(identity.substring(start, end));
    }

    private static void decode(CodeBuilder code, int geometry, int logical, int coordinates,
            int layout, int rank) {
        int remaining = code.allocateLocal(TypeKind.LONG); code.lload(logical).lstore(remaining);
        for (int axis = rank - 1; axis >= 0; axis--) {
            code.aload(geometry).loadConstant(coordinates + axis).lload(remaining)
                    .aload(geometry).loadConstant(layout + 2 + axis).laload().lrem().lastore();
            code.lload(remaining).aload(geometry).loadConstant(layout + 2 + axis).laload()
                    .ldiv().lstore(remaining);
        }
    }

    private static int address(CodeBuilder code, int geometry, int rank, int coordinates, int layout) {
        int address = code.allocateLocal(TypeKind.LONG);
        code.aload(geometry).loadConstant(layout + 1).laload().lstore(address);
        for (int axis = 0; axis < rank; axis++) code.lload(address)
                .aload(geometry).loadConstant(coordinates + axis).laload()
                .aload(geometry).loadConstant(layout + 2 + rank + axis).laload().lmul().ladd()
                .lstore(address);
        return address;
    }

    private static TypeKind localKind(DataType type) { return switch (type) {
        case FLOAT64 -> TypeKind.DOUBLE; case FLOAT32 -> TypeKind.FLOAT;
        case INT64 -> TypeKind.LONG; case BFLOAT16, INT32, BOOL -> TypeKind.INT;
    }; }
    private static void emitFloatingZero(CodeBuilder code, DataType type) { switch (type) {
        case FLOAT64 -> code.loadConstant(0.0d); case FLOAT32 -> code.loadConstant(0.0f);
        case BFLOAT16 -> code.loadConstant(0); default -> throw new IllegalArgumentException();
    } }
    private static void store(CodeBuilder code, DataType type, int local) { switch (type) {
        case FLOAT64 -> code.dstore(local); case FLOAT32 -> code.fstore(local);
        case INT64 -> code.lstore(local); case BFLOAT16, INT32, BOOL -> code.istore(local);
    } }
    private static void load(CodeBuilder code, DataType type, int local) { switch (type) {
        case FLOAT64 -> code.dload(local); case FLOAT32 -> code.fload(local);
        case INT64 -> code.lload(local); case BFLOAT16, INT32, BOOL -> code.iload(local);
    } }
    private static void emitIdentity(CodeBuilder code, DataType type, int kind) { switch (type) {
        case FLOAT64 -> code.loadConstant(kind == 0 ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY);
        case FLOAT32 -> code.loadConstant(kind == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY);
        case BFLOAT16 -> code.loadConstant(kind == 0 ? 0x7f80 : 0xff80);
        case INT32 -> code.loadConstant(kind == 4 ? 0 : kind == 6 ? 1
                : kind == 0 ? Integer.MAX_VALUE : Integer.MIN_VALUE);
        case INT64 -> code.loadConstant(kind == 4 ? 0L : kind == 6 ? 1L
                : kind == 0 ? Long.MAX_VALUE : Long.MIN_VALUE);
        case BOOL -> code.loadConstant(kind == 2 ? 1 : 0);
    } }
    private static void emitApply(CodeBuilder code, DataType type, int kind,
            int accumulator, int value) {
        if ((type == DataType.INT32 || type == DataType.INT64) && (kind == 4 || kind == 6)) {
            load(code, type, accumulator); load(code, type, value);
            if (type == DataType.INT32) {
                if (kind == 4) code.iadd(); else code.imul();
            } else if (kind == 4) code.ladd(); else code.lmul();
            store(code, type, accumulator);
            return;
        }
        switch (type) {
            case FLOAT64 -> emitFloatingSelection(code, true, kind == 0, accumulator, value);
            case FLOAT32 -> emitFloatingSelection(code, false, kind == 0, accumulator, value);
            case BFLOAT16 -> emitBfloatSelection(code, kind == 0, accumulator, value);
            case INT32 -> emitIntSelection(code, kind == 0, accumulator, value);
            case INT64 -> emitLongSelection(code, kind == 0, accumulator, value);
            case BOOL -> emitBooleanCombination(code, kind == 2, accumulator, value);
        }
    }

    private static void emitIntSelection(CodeBuilder code, boolean minimum, int left, int right) {
        code.iload(left).iload(right).invokestatic(ClassDescHolder.MATH,
                minimum ? "min" : "max", java.lang.constant.MethodTypeDesc.of(
                        ConstantDescs.CD_int, ConstantDescs.CD_int, ConstantDescs.CD_int))
                .istore(left);
    }

    private static void emitLongSelection(CodeBuilder code, boolean minimum, int left, int right) {
        code.lload(left).lload(right).invokestatic(ClassDescHolder.MATH,
                minimum ? "min" : "max", java.lang.constant.MethodTypeDesc.of(
                        ConstantDescs.CD_long, ConstantDescs.CD_long, ConstantDescs.CD_long))
                .lstore(left);
    }

    private static void emitBooleanCombination(CodeBuilder code, boolean all, int left, int right) {
        var falseResult = code.newLabel(); var trueResult = code.newLabel(); var done = code.newLabel();
        if (all) {
            code.iload(left).branch(Opcode.IFEQ, falseResult);
            code.iload(right).branch(Opcode.IFEQ, falseResult);
            code.labelBinding(trueResult).loadConstant(1).branch(Opcode.GOTO, done);
            code.labelBinding(falseResult).loadConstant(0);
        } else {
            code.iload(left).branch(Opcode.IFNE, trueResult);
            code.iload(right).branch(Opcode.IFNE, trueResult);
            code.loadConstant(0).branch(Opcode.GOTO, done);
            code.labelBinding(trueResult).loadConstant(1);
        }
        code.labelBinding(done).istore(left);
    }

    private static void emitFloatingSelection(CodeBuilder code, boolean binary64,
            boolean minimum, int left, int right) {
        var leftNumber = code.newLabel(); var rightNumber = code.newLabel();
        var ordinary = code.newLabel(); var chooseRight = code.newLabel(); var chooseLeft = code.newLabel();
        var done = code.newLabel();
        if (binary64) {
            code.dload(left).invokestatic(ClassDescHolder.DOUBLE, "isNaN",
                    java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_boolean,
                            ConstantDescs.CD_double)).branch(Opcode.IFEQ, leftNumber);
            code.dload(left).branch(Opcode.GOTO, done).labelBinding(leftNumber);
            code.dload(right).invokestatic(ClassDescHolder.DOUBLE, "isNaN",
                    java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_boolean,
                            ConstantDescs.CD_double)).branch(Opcode.IFEQ, rightNumber);
            code.dload(right).branch(Opcode.GOTO, done).labelBinding(rightNumber);
            code.dload(left).loadConstant(0.0d).dcmpl().branch(Opcode.IFNE, ordinary);
            code.dload(right).loadConstant(0.0d).dcmpl().branch(Opcode.IFNE, ordinary);
            code.dload(left).invokestatic(ClassDescHolder.DOUBLE, "doubleToRawLongBits",
                    java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_long,
                            ConstantDescs.CD_double)).loadConstant(0L).lcmp()
                    .branch(minimum ? Opcode.IFLT : Opcode.IFGE, chooseLeft);
            code.branch(Opcode.GOTO, chooseRight).labelBinding(ordinary);
            code.dload(right).dload(left).dcmpg().branch(
                    minimum ? Opcode.IFLT : Opcode.IFGT, chooseRight);
            code.labelBinding(chooseLeft).dload(left).branch(Opcode.GOTO, done);
            code.labelBinding(chooseRight).dload(right);
            code.labelBinding(done).dstore(left);
        } else {
            code.fload(left).invokestatic(ClassDescHolder.FLOAT, "isNaN",
                    java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_boolean,
                            ConstantDescs.CD_float)).branch(Opcode.IFEQ, leftNumber);
            code.fload(left).branch(Opcode.GOTO, done).labelBinding(leftNumber);
            code.fload(right).invokestatic(ClassDescHolder.FLOAT, "isNaN",
                    java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_boolean,
                            ConstantDescs.CD_float)).branch(Opcode.IFEQ, rightNumber);
            code.fload(right).branch(Opcode.GOTO, done).labelBinding(rightNumber);
            code.fload(left).loadConstant(0.0f).fcmpl().branch(Opcode.IFNE, ordinary);
            code.fload(right).loadConstant(0.0f).fcmpl().branch(Opcode.IFNE, ordinary);
            code.fload(left).invokestatic(ClassDescHolder.FLOAT, "floatToRawIntBits",
                    java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_int,
                            ConstantDescs.CD_float)).branch(minimum ? Opcode.IFLT : Opcode.IFGE,
                            chooseLeft);
            code.branch(Opcode.GOTO, chooseRight).labelBinding(ordinary);
            code.fload(right).fload(left).fcmpg().branch(
                    minimum ? Opcode.IFLT : Opcode.IFGT, chooseRight);
            code.labelBinding(chooseLeft).fload(left).branch(Opcode.GOTO, done);
            code.labelBinding(chooseRight).fload(right);
            code.labelBinding(done).fstore(left);
        }
    }

    private static void emitBfloatSelection(CodeBuilder code, boolean minimum, int left, int right) {
        var leftNumber = code.newLabel(); var rightNumber = code.newLabel();
        var ordinary = code.newLabel(); var chooseRight = code.newLabel(); var chooseLeft = code.newLabel();
        var done = code.newLabel();
        code.iload(left).loadConstant(0xffff).iand().istore(left);
        code.iload(right).loadConstant(0xffff).iand().istore(right);
        code.iload(left).loadConstant(0x7f80).iand().loadConstant(0x7f80)
                .branch(Opcode.IF_ICMPNE, leftNumber);
        code.iload(left).loadConstant(0x7f).iand().branch(Opcode.IFEQ, leftNumber);
        code.iload(left).branch(Opcode.GOTO, done).labelBinding(leftNumber);
        code.iload(right).loadConstant(0x7f80).iand().loadConstant(0x7f80)
                .branch(Opcode.IF_ICMPNE, rightNumber);
        code.iload(right).loadConstant(0x7f).iand().branch(Opcode.IFEQ, rightNumber);
        code.iload(right).branch(Opcode.GOTO, done).labelBinding(rightNumber);
        code.iload(left).loadConstant(0x7fff).iand().branch(Opcode.IFNE, ordinary);
        code.iload(right).loadConstant(0x7fff).iand().branch(Opcode.IFNE, ordinary);
        code.iload(left).i2s().branch(minimum ? Opcode.IFLT : Opcode.IFGE, chooseLeft);
        code.branch(Opcode.GOTO, chooseRight).labelBinding(ordinary);
        code.iload(right).loadConstant(16).ishl().invokestatic(ClassDescHolder.FLOAT,
                "intBitsToFloat", java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_float,
                        ConstantDescs.CD_int));
        code.iload(left).loadConstant(16).ishl().invokestatic(ClassDescHolder.FLOAT,
                "intBitsToFloat", java.lang.constant.MethodTypeDesc.of(ConstantDescs.CD_float,
                        ConstantDescs.CD_int));
        code.fcmpg().branch(minimum ? Opcode.IFLT : Opcode.IFGT, chooseRight);
        code.labelBinding(chooseLeft).iload(left).branch(Opcode.GOTO, done);
        code.labelBinding(chooseRight).iload(right);
        code.labelBinding(done).istore(left);
    }

    private static final class ClassDescHolder {
        private static final java.lang.constant.ClassDesc DOUBLE =
                java.lang.constant.ClassDesc.of(Double.class.getName());
        private static final java.lang.constant.ClassDesc FLOAT =
                java.lang.constant.ClassDesc.of(Float.class.getName());
        private static final java.lang.constant.ClassDesc MATH =
                java.lang.constant.ClassDesc.of(Math.class.getName());
        private ClassDescHolder() { }
    }

    /**
     * Selects the FLOAT64 minimum using CPU aggregate NaN and signed-zero rules.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the selected represented value
     */
    static double minDouble(double left, double right) { return selectDouble(left, right, true); }

    /**
     * Selects the FLOAT64 maximum using CPU aggregate NaN and signed-zero rules.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the selected represented value
     */
    static double maxDouble(double left, double right) { return selectDouble(left, right, false); }
    private static double selectDouble(double left, double right, boolean minimum) {
        if (Double.isNaN(left)) return left; if (Double.isNaN(right)) return right;
        if (left == 0.0 && right == 0.0) return minimum
                ? (Double.doubleToRawLongBits(left) < 0 ? left : right)
                : (Double.doubleToRawLongBits(left) >= 0 ? left : right);
        return minimum ? (right < left ? right : left) : (right > left ? right : left);
    }
    /**
     * Selects the FLOAT32 minimum using CPU aggregate NaN and signed-zero rules.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the selected represented value
     */
    static float minFloat(float left, float right) { return selectFloat(left, right, true); }

    /**
     * Selects the FLOAT32 maximum using CPU aggregate NaN and signed-zero rules.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the selected represented value
     */
    static float maxFloat(float left, float right) { return selectFloat(left, right, false); }
    private static float selectFloat(float left, float right, boolean minimum) {
        if (Float.isNaN(left)) return left; if (Float.isNaN(right)) return right;
        if (left == 0.0f && right == 0.0f) return minimum
                ? (Float.floatToRawIntBits(left) < 0 ? left : right)
                : (Float.floatToRawIntBits(left) >= 0 ? left : right);
        return minimum ? (right < left ? right : left) : (right > left ? right : left);
    }
    /**
     * Selects the BFLOAT16 minimum using unsigned represented-bit inputs and aggregate
     * NaN/signed-zero rules.
     *
     * @param left unsigned 16-bit BFLOAT16 bits in the low half of the integer
     * @param right unsigned 16-bit BFLOAT16 bits in the low half of the integer
     * @return unsigned 16-bit BFLOAT16 bits for the selected represented value
     */
    static int minBfloat(int left, int right) { return selectBfloat(left, right, true); }

    /**
     * Selects the BFLOAT16 maximum using unsigned represented-bit inputs and aggregate
     * NaN/signed-zero rules.
     *
     * @param left unsigned 16-bit BFLOAT16 bits in the low half of the integer
     * @param right unsigned 16-bit BFLOAT16 bits in the low half of the integer
     * @return unsigned 16-bit BFLOAT16 bits for the selected represented value
     */
    static int maxBfloat(int left, int right) { return selectBfloat(left, right, false); }
    private static int selectBfloat(int left, int right, boolean minimum) {
        int le = left & 0xffff, ri = right & 0xffff;
        if ((le & 0x7f80) == 0x7f80 && (le & 0x7f) != 0) return le;
        if ((ri & 0x7f80) == 0x7f80 && (ri & 0x7f) != 0) return ri;
        float l = Float.intBitsToFloat(le << 16), r = Float.intBitsToFloat(ri << 16);
        if (l == 0.0f && r == 0.0f) return minimum ? ((short) le < 0 ? le : ri)
                : ((short) le >= 0 ? le : ri);
        return minimum ? (r < l ? ri : le) : (r > l ? ri : le);
    }
    /**
     * Selects the smaller INT32 value.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the smaller represented value
     */
    static int minInt(int left, int right) { return Math.min(left, right); }

    /**
     * Selects the larger INT32 value.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the larger represented value
     */
    static int maxInt(int left, int right) { return Math.max(left, right); }

    /**
     * Selects the smaller INT64 value.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the smaller represented value
     */
    static long minLong(long left, long right) { return Math.min(left, right); }

    /**
     * Selects the larger INT64 value.
     *
     * @param left first represented value
     * @param right second represented value
     * @return the larger represented value
     */
    static long maxLong(long left, long right) { return Math.max(left, right); }

    /**
     * Applies represented BOOL conjunction.
     *
     * @param left first represented boolean, where zero is false and non-zero is true
     * @param right second represented boolean, where zero is false and non-zero is true
     * @return {@code 1} when both inputs are true, otherwise {@code 0}
     */
    static int all(int left, int right) { return left != 0 && right != 0 ? 1 : 0; }

    /**
     * Applies represented BOOL disjunction.
     *
     * @param left first represented boolean, where zero is false and non-zero is true
     * @param right second represented boolean, where zero is false and non-zero is true
     * @return {@code 1} when either input is true, otherwise {@code 0}
     */
    static int any(int left, int right) { return left != 0 || right != 0 ? 1 : 0; }

    /**
     * Reduces complete flattened output cells in {@code [start,end)}.
     * @param input non-null readable primitive-array or native-segment carrier
     * @param output non-null writable non-overlapping carrier
     * @param packed non-null invocation-owned geometry and mutable coordinate state
     * @param start non-negative inclusive output-cell ordinal
     * @param end exclusive output-cell ordinal no greater than the output count
     * @throws NullPointerException if a carrier or {@code packed} is {@code null}
     * @throws ClassCastException if a carrier does not match the represented type selected during
     *     cold specialization
     * @throws ArithmeticException if an array address cannot be represented as {@code int}
     * @throws IndexOutOfBoundsException if a carrier does not cover a packed address
     * @throws IllegalStateException if a supplied memory segment is inaccessible
     */
    public static void execute(Object input, Object output, long[] packed, long start, long end) {
        int kind = (int) packed[0]; DataType type = TYPES[(int) packed[1]];
        boolean keep = packed[3] != 0; int inRank = (int) packed[4], outRank = (int) packed[5];
        long domainCount = packed[7]; int selected = 11;
        int inputCoordinates = selected + inRank;
        int outputCoordinates = inputCoordinates + inRank;
        int inputLayout = outputCoordinates + outRank;
        int outputLayout = inputLayout + 2 + 2 * inRank;
        for (long cell = start; cell < end; cell++) {
            decode(cell, packed, outputCoordinates, packed, outputLayout);
            int outAxis = 0;
            for (int axis = 0; axis < inRank; axis++) {
                if (packed[selected + axis] != 0) packed[inputCoordinates + axis] = 0;
                else packed[inputCoordinates + axis] = packed[outputCoordinates
                        + (keep ? axis : outAxis++)];
            }
            long accumulator = identity(kind, type);
            for (long domain = 0; domain < domainCount; domain++) {
                long remaining = domain;
                for (int axis = inRank - 1; axis >= 0; axis--) if (packed[selected + axis] != 0) {
                    long extent = packed[inputLayout + 2 + axis];
                    packed[inputCoordinates + axis] = remaining % extent; remaining /= extent;
                }
                long value = readBits(input, address(packed, inputLayout, inputCoordinates), type);
                accumulator = apply(kind, accumulator, value, type);
            }
            writeBits(output, address(packed, outputLayout, outputCoordinates), type, accumulator);
        }
    }

    private static void decode(long logical, long[] coordinatesOwner, int coordinates,
            long[] layoutOwner, int layout) {
        int rank = (int) layoutOwner[layout];
        for (int axis = rank - 1; axis >= 0; axis--) {
            long extent = layoutOwner[layout + 2 + axis];
            coordinatesOwner[coordinates + axis] = logical % extent; logical /= extent;
        }
    }
    private static long identity(int kind, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits(kind == 0
                    ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY);
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(kind == 0
                    ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY));
            case BFLOAT16 -> kind == 0 ? 0x7f80L : 0xff80L;
            case INT32 -> kind == 0 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            case INT64 -> kind == 0 ? Long.MAX_VALUE : Long.MIN_VALUE;
            case BOOL -> kind == 2 ? 1 : 0;
        };
    }
    private static long apply(int kind, long left, long right, DataType type) {
        if (type == DataType.BOOL) return kind == 2
                ? ((left != 0 && right != 0) ? 1 : 0) : ((left != 0 || right != 0) ? 1 : 0);
        if (type == DataType.INT32) return kind == 0
                ? Math.min((int) left, (int) right) : Math.max((int) left, (int) right);
        if (type == DataType.INT64) return kind == 0 ? Math.min(left, right) : Math.max(left, right);
        if (isNaN(left, type)) return left;
        if (isNaN(right, type)) return right;
        double l = floatingValue(left, type), r = floatingValue(right, type);
        if (l == 0.0 && r == 0.0) {
            boolean leftNegative = negative(left, type), rightNegative = negative(right, type);
            if (kind == 0) return leftNegative ? left : rightNegative ? right : left;
            return !leftNegative ? left : !rightNegative ? right : left;
        }
        return kind == 0 ? (r < l ? right : left) : (r > l ? right : left);
    }
    private static boolean isNaN(long bits, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.isNaN(Double.longBitsToDouble(bits));
            case FLOAT32 -> Float.isNaN(Float.intBitsToFloat((int) bits));
            case BFLOAT16 -> ((bits & 0x7f80L) == 0x7f80L) && (bits & 0x7fL) != 0;
            default -> false;
        };
    }
    private static double floatingValue(long bits, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.longBitsToDouble(bits);
            case FLOAT32 -> Float.intBitsToFloat((int) bits);
            case BFLOAT16 -> Float.intBitsToFloat((int) bits << 16);
            default -> throw new AssertionError("non-floating aggregate type");
        };
    }
    private static boolean negative(long bits, DataType type) {
        return switch (type) {
            case FLOAT64 -> bits < 0;
            case FLOAT32 -> ((int) bits) < 0;
            case BFLOAT16 -> ((short) bits) < 0;
            default -> false;
        };
    }
    private static long address(long[] p, int layout, int coordinates) {
        long result = p[layout + 1]; int rank = (int) p[layout];
        for (int axis = 0; axis < rank; axis++)
            result += p[coordinates + axis] * p[layout + 2 + rank + axis];
        return result;
    }
    private static long readBits(Object carrier, long address, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits(carrier instanceof double[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_DOUBLE, address * 8));
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(carrier instanceof float[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_FLOAT, address * 4)));
            case BFLOAT16 -> Short.toUnsignedLong(carrier instanceof short[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_SHORT, address * 2));
            case INT32 -> carrier instanceof int[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_INT, address * 4);
            case INT64 -> carrier instanceof long[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_LONG, address * 8);
            case BOOL -> carrier instanceof byte[] a ? Byte.toUnsignedLong(a[Math.toIntExact(address)]) : Byte.toUnsignedLong(((MemorySegment) carrier).get(ValueLayout.JAVA_BYTE, address));
        };
    }
    private static void writeBits(Object carrier, long address, DataType type, long bits) {
        switch (type) {
            case FLOAT64 -> { double v = Double.longBitsToDouble(bits); if (carrier instanceof double[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_DOUBLE, address * 8, v); }
            case FLOAT32 -> { float v = Float.intBitsToFloat((int) bits); if (carrier instanceof float[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_FLOAT, address * 4, v); }
            case BFLOAT16 -> { short v = (short) bits; if (carrier instanceof short[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_SHORT, address * 2, v); }
            case INT32 -> { int v = (int) bits; if (carrier instanceof int[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_INT, address * 4, v); }
            case INT64 -> { if (carrier instanceof long[] a) a[Math.toIntExact(address)] = bits; else ((MemorySegment) carrier).set(ValueLayout.JAVA_LONG, address * 8, bits); }
            case BOOL -> { byte v = (byte) (bits == 0 ? 0 : 1); if (carrier instanceof byte[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_BYTE, address, v); }
        }
    }
}
