package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.Label;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;

/**
 * Emits direct stable three-pass softmax and log-softmax bodies.
 *
 * <p>Kind selection occurs while generating the class. Both bodies find the slice maximum and
 * accumulate shifted exponentials; only log-softmax computes the logarithm of that sum, while
 * softmax proceeds directly to exponential-and-division stores. FLOAT64 uses double sites;
 * FLOAT32, BFLOAT16, and FLOAT16 use float sites with one final low narrowing. All-finite slices
 * retain maximum-shifted compensated accumulation. NaN and positive-infinity slices preserve
 * Model result classes, while slices containing only negative infinity as a nonfinite value
 * use their literal unshifted formula. Empty slices publish nothing.</p>
 * <p>A controlled helper guards the proved original ROW coordinate of each complete softmax
 * slice before all passes, including frozen ordinary geometries. The normalized axis is
 * explicitly omitted from slice ordinals; equal extents never establish ROW provenance.</p>
 */
public final class CpuSoftmaxEmitter {
    private static final ClassDesc MATH = ClassDesc.of(Math.class.getName());
    private static final ClassDesc FLOAT = ClassDesc.of(Float.class.getName());
    private static final ClassDesc DOUBLE = ClassDesc.of(Double.class.getName());

    /** Creates a stateless generation-time emitter. */
    public CpuSoftmaxEmitter() { }

    /**
     * Emits one generation-time-kind-specialized typed entry body.
     * @param code non-null Class-File method builder to mutate
     * @param specialization non-null exact two-carrier specialization
     * @param ir non-null matching canonical softmax identity
     * @throws IllegalArgumentException if family, boundary, or scratch facts disagree
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        boolean softmax = ir.familyIdentity().startsWith("softmax:SOFTMAX:");
        boolean logSoftmax = ir.familyIdentity().startsWith("softmax:LOG_SOFTMAX:");
        if ((!softmax && !logSoftmax) || specialization.carrierPattern().size() != 2
                || specialization.boundaryDataTypes().size() != 2
                || specialization.boundaryDataTypes().getFirst()
                    != specialization.boundaryDataTypes().getLast()
                || specialization.scratchParameter())
            throw new IllegalArgumentException("softmax requires two matching boundaries without scratch");
        DataType type = specialization.boundaryDataTypes().getFirst();
        int rank = ir.values().getFirst().accessPlan().iterationRank();
        int axis = Integer.parseInt(field(ir.familyIdentity(), ":axis="));
        int geometry = 2, start = 3, end = 5;
        int extents = 7, inputStrides = extents + rank, outputStrides = inputStrides + rank;
        var carriers = new CpuCarrierEmitter(code);
        int slice = code.allocateLocal(TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.LONG);
        int coordinate = code.allocateLocal(TypeKind.LONG);
        int inputBase = code.allocateLocal(TypeKind.LONG);
        int outputBase = code.allocateLocal(TypeKind.LONG);
        boolean inputArray = specialization.carrierPattern().getFirst()
                != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        boolean outputArray = specialization.carrierPattern().getLast()
                != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        int inputAddress = code.allocateLocal(inputArray ? TypeKind.INT : TypeKind.LONG);
        int outputAddress = code.allocateLocal(outputArray ? TypeKind.INT : TypeKind.LONG);
        boolean intLoop = inputArray || outputArray;
        int intSlice = code.allocateLocal(TypeKind.INT);
        int intEnd = code.allocateLocal(TypeKind.INT);
        int width = code.allocateLocal(intLoop ? TypeKind.INT : TypeKind.LONG);
        int element = code.allocateLocal(intLoop ? TypeKind.INT : TypeKind.LONG);
        int inputAxisStride = code.allocateLocal(TypeKind.LONG);
        int outputAxisStride = code.allocateLocal(TypeKind.LONG);
        int inputOrigin = code.allocateLocal(TypeKind.LONG);
        int outputOrigin = code.allocateLocal(TypeKind.LONG);
        int inputRowStride = code.allocateLocal(TypeKind.LONG);
        int outputRowStride = code.allocateLocal(TypeKind.LONG);
        int represented = code.allocateLocal(type == DataType.FLOAT64 ? TypeKind.DOUBLE
                : type == DataType.FLOAT32 ? TypeKind.FLOAT : TypeKind.INT);
        boolean floatWork = type != DataType.FLOAT64;
        TypeKind arithmetic = floatWork ? TypeKind.FLOAT : TypeKind.DOUBLE;
        int value = code.allocateLocal(arithmetic);
        int maximum = code.allocateLocal(arithmetic);
        int shifted = code.allocateLocal(arithmetic);
        int sum = code.allocateLocal(arithmetic);
        int compensation = code.allocateLocal(arithmetic);
        int temporary = code.allocateLocal(arithmetic);
        int logarithm = code.allocateLocal(arithmetic);
        int unshifted = code.allocateLocal(TypeKind.INT);
        int positiveInfinity = code.allocateLocal(TypeKind.INT);

        boolean frozenGeneral = ir.values().getFirst().accessPlan().regime()
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.GENERAL_ODOMETER;
        if (rank == 2 && axis == 1) {
            var genericGeometry = code.newLabel();
            guardGeometry(code, geometry, 4, 2048, genericGeometry);
            guardGeometry(code, geometry, extents, 128, genericGeometry);
            guardGeometry(code, geometry, extents + 1, 2048, genericGeometry);
            guardGeometry(code, geometry, 0, frozenGeneral ? 1 : 0, genericGeometry);
            guardGeometry(code, geometry, 1, 0, genericGeometry);
            guardGeometry(code, geometry, inputStrides, frozenGeneral ? 2049 : 2048,
                    genericGeometry);
            guardGeometry(code, geometry, inputStrides + 1, 1, genericGeometry);
            guardGeometry(code, geometry, outputStrides, 2048, genericGeometry);
            guardGeometry(code, geometry, outputStrides + 1, 1, genericGeometry);
            emitFrozen(code, carriers, specialization, type, softmax, floatWork, frozenGeneral,
                    start, end, intSlice, intEnd, inputAddress, outputAddress, inputArray,
                    outputArray, element, represented, value, maximum, shifted, sum, compensation,
                    temporary, logarithm, unshifted, positiveInfinity);
            code.return_().labelBinding(genericGeometry);
        }
        loadGeometry(code, geometry, intLoop, width, inputAxisStride, outputAxisStride,
                inputOrigin, outputOrigin, inputRowStride, outputRowStride, inputStrides,
                outputStrides, axis);
        if (intLoop) {
            code.lload(start).l2i().istore(intSlice);
            code.lload(end).l2i().istore(intEnd);
        } else code.lload(start).lstore(slice);
        var slices = code.newLabel(); var done = code.newLabel();
        if (intLoop) code.iload(width).branch(Opcode.IFEQ, done);
        else code.lload(width).lconst_0().lcmp().branch(Opcode.IFEQ, done);
        code.labelBinding(slices);
        if (intLoop) code.iload(intSlice).iload(intEnd).branch(Opcode.IF_ICMPGE, done);
        else code.lload(slice).lload(end).lcmp().branch(Opcode.IFGE, done);
        var advanceSlice = code.newLabel();
        boolean[] omitted = new boolean[rank]; omitted[axis] = true;
        CpuLoopEmitter.emitActiveSliceGuard(code, 1, intLoop ? intSlice : slice,
                intLoop ? TypeKind.INT : TypeKind.LONG, geometry, extents, omitted, advanceSlice);
        code.lload(inputOrigin).lstore(inputBase);
        code.lload(outputOrigin).lstore(outputBase);
        if (rank == 2 && axis == 1) {
            code.lload(inputBase); loadSlice(code, slice, intSlice, intLoop); code.lload(inputRowStride)
                    .lmul().ladd().lstore(inputBase);
            code.lload(outputBase); loadSlice(code, slice, intSlice, intLoop); code.lload(outputRowStride)
                    .lmul().ladd().lstore(outputBase);
        } else {
            loadSlice(code, slice, intSlice, intLoop).lstore(remaining);
            for (int logicalAxis = rank - 1; logicalAxis >= 0; logicalAxis--) {
                if (logicalAxis == axis) continue;
                code.lload(remaining); CpuNormEmitter.geometry(code, geometry, extents + logicalAxis)
                        .lrem().lstore(coordinate);
                code.lload(remaining); CpuNormEmitter.geometry(code, geometry, extents + logicalAxis)
                        .ldiv().lstore(remaining);
                code.lload(inputBase).lload(coordinate); CpuNormEmitter.geometry(code, geometry,
                        inputStrides + logicalAxis).lmul().ladd().lstore(inputBase);
                code.lload(outputBase).lload(coordinate); CpuNormEmitter.geometry(code, geometry,
                        outputStrides + logicalAxis).lmul().ladd().lstore(outputBase);
            }
        }
        initializeMaximum(code, floatWork, maximum);
        code.iconst_0().istore(unshifted);
        zeroElement(code, element, intLoop);
        assignAddress(code, inputBase, inputAddress, inputArray);
        var maxLoop = code.newLabel(); var maxDone = code.newLabel();
        code.labelBinding(maxLoop);
        compareElement(code, element, width, intLoop, maxDone);
        load(code, carriers, specialization, type, inputAddress, represented, value, inputArray);
        updateMaximum(code, floatWork, value, maximum, unshifted);
        advance(code, inputAxisStride, element, intLoop, inputAddress, inputArray, maxLoop);
        code.labelBinding(maxDone);
        Label normalized = beginNormalization(code, floatWork, softmax, maximum, sum,
                logarithm, unshifted, positiveInfinity);

        zero(code, floatWork, sum); zero(code, floatWork, compensation);
        zeroElement(code, element, intLoop);
        assignAddress(code, inputBase, inputAddress, inputArray);
        var sumLoop = code.newLabel(); var sumDone = code.newLabel();
        code.labelBinding(sumLoop);
        compareElement(code, element, width, intLoop, sumDone);
        load(code, carriers, specialization, type, inputAddress, represented, value, inputArray);
        subtract(code, floatWork, value, maximum, shifted);
        exponential(code, floatWork, shifted);
        accumulate(code, floatWork, shifted, sum, compensation, temporary, unshifted);
        advance(code, inputAxisStride, element, intLoop, inputAddress, inputArray, sumLoop);
        code.labelBinding(sumDone);
        if (logSoftmax) logarithm(code, floatWork, sum, logarithm);
        code.labelBinding(normalized);

        zeroElement(code, element, intLoop);
        assignAddress(code, inputBase, inputAddress, inputArray);
        assignAddress(code, outputBase, outputAddress, outputArray);
        var storeLoop = code.newLabel(); var storeDone = code.newLabel();
        code.labelBinding(storeLoop);
        compareElement(code, element, width, intLoop, storeDone);
        load(code, carriers, specialization, type, inputAddress, represented, value, inputArray);
        result(code, floatWork, softmax, value, maximum, sum, logarithm, shifted,
                positiveInfinity);
        store(code, carriers, specialization, type, outputAddress, shifted, outputArray, false);
        incrementAddress(code, inputAddress, inputAxisStride, inputArray);
        incrementAddress(code, outputAddress, outputAxisStride, outputArray);
        incrementElement(code, element, intLoop);
        code.branch(Opcode.GOTO, storeLoop).labelBinding(storeDone);
        code.labelBinding(advanceSlice);
        if (intLoop) code.iinc(intSlice, 1);
        else code.lload(slice).loadConstant(1L).ladd().lstore(slice);
        code.branch(Opcode.GOTO, slices).labelBinding(done);
    }

    private static void emitFrozen(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, boolean softmax, boolean floatWork,
            boolean generalInput, int start, int end, int row, int rowEnd, int inputAddress,
            int outputAddress, boolean inputArray, boolean outputArray, int element, int represented,
            int value, int maximum, int shifted, int sum, int compensation, int temporary,
            int logarithm, int unshifted, int positiveInfinity) {
        code.lload(start).l2i().istore(row).lload(end).l2i().istore(rowEnd);
        var rows = code.newLabel(); var rowsDone = code.newLabel();
        code.labelBinding(rows).iload(row).iload(rowEnd).branch(Opcode.IF_ICMPGE, rowsDone);
        var advanceRow = code.newLabel();
        CpuLoopEmitter.emitActiveSliceGuard(code, 1, row, TypeKind.INT, 2, 7,
                new boolean[] {false, true}, advanceRow);
        frozenAddress(code, row, generalInput ? 2049 : 2048, generalInput ? 1 : 0,
                inputAddress, inputArray);
        frozenAddress(code, row, 2048, 0, outputAddress, outputArray);
        initializeMaximum(code, floatWork, maximum);
        code.iconst_0().istore(unshifted);
        code.loadConstant(0).istore(element);
        var maxLoop = code.newLabel(); var maxDone = code.newLabel();
        code.labelBinding(maxLoop).iload(element).loadConstant(2048)
                .branch(Opcode.IF_ICMPGE, maxDone);
        loadFrozen(code, carriers, specialization, type, inputAddress, represented, value,
                inputArray);
        updateMaximum(code, floatWork, value, maximum, unshifted);
        incrementConstant(code, inputAddress, inputArray);
        code.iinc(element, 1).branch(Opcode.GOTO, maxLoop).labelBinding(maxDone);
        Label normalized = beginNormalization(code, floatWork, softmax, maximum, sum,
                logarithm, unshifted, positiveInfinity);

        zero(code, floatWork, sum); zero(code, floatWork, compensation);
        frozenAddress(code, row, generalInput ? 2049 : 2048, generalInput ? 1 : 0,
                inputAddress, inputArray);
        code.loadConstant(0).istore(element);
        var sumLoop = code.newLabel(); var sumDone = code.newLabel();
        code.labelBinding(sumLoop).iload(element).loadConstant(2048)
                .branch(Opcode.IF_ICMPGE, sumDone);
        loadFrozen(code, carriers, specialization, type, inputAddress, represented, value,
                inputArray);
        subtract(code, floatWork, value, maximum, shifted); exponential(code, floatWork, shifted);
        accumulate(code, floatWork, shifted, sum, compensation, temporary, unshifted);
        incrementConstant(code, inputAddress, inputArray);
        code.iinc(element, 1).branch(Opcode.GOTO, sumLoop).labelBinding(sumDone);
        if (!softmax) logarithm(code, floatWork, sum, logarithm);
        code.labelBinding(normalized);

        frozenAddress(code, row, generalInput ? 2049 : 2048, generalInput ? 1 : 0,
                inputAddress, inputArray);
        frozenAddress(code, row, 2048, 0, outputAddress, outputArray);
        code.loadConstant(0).istore(element);
        var storeLoop = code.newLabel(); var storeDone = code.newLabel();
        code.labelBinding(storeLoop).iload(element).loadConstant(2048)
                .branch(Opcode.IF_ICMPGE, storeDone);
        loadFrozen(code, carriers, specialization, type, inputAddress, represented, value,
                inputArray);
        result(code, floatWork, softmax, value, maximum, sum, logarithm, shifted,
                positiveInfinity);
        store(code, carriers, specialization, type, outputAddress, shifted, outputArray, true);
        incrementConstant(code, inputAddress, inputArray);
        incrementConstant(code, outputAddress, outputArray);
        code.iinc(element, 1).branch(Opcode.GOTO, storeLoop).labelBinding(storeDone);
        code.labelBinding(advanceRow);
        code.iinc(row, 1).branch(Opcode.GOTO, rows).labelBinding(rowsDone);
    }

    private static void frozenAddress(CodeBuilder code, int row, int stride, int origin,
            int address, boolean intAddress) {
        code.iload(row).loadConstant(stride).imul().loadConstant(origin).iadd();
        if (intAddress) code.istore(address); else code.i2l().lstore(address);
    }

    private static void incrementConstant(CodeBuilder code, int address, boolean intAddress) {
        if (intAddress) code.iinc(address, 1);
        else code.lload(address).loadConstant(1L).ladd().lstore(address);
    }

    private static String field(String identity, String marker) {
        int begin = identity.indexOf(marker) + marker.length();
        int end = identity.indexOf(':', begin);
        return identity.substring(begin, end);
    }

    private static CodeBuilder loadSlice(CodeBuilder code, int slice, int intSlice,
            boolean intLoop) {
        return intLoop ? code.iload(intSlice).i2l() : code.lload(slice);
    }

    private static void guardGeometry(CodeBuilder code, int geometry, int index, long expected,
            java.lang.classfile.Label fallback) {
        CpuNormEmitter.geometry(code, geometry, index).loadConstant(expected).lcmp()
                .branch(Opcode.IFNE, fallback);
    }

    private static void loadGeometry(CodeBuilder code, int geometry, boolean intLoop, int width,
            int inputAxisStride, int outputAxisStride, int inputOrigin, int outputOrigin,
            int inputRowStride, int outputRowStride, int inputStrides, int outputStrides, int axis) {
        CpuNormEmitter.geometry(code, geometry, 4);
        if (intLoop) code.l2i().istore(width); else code.lstore(width);
        CpuNormEmitter.geometry(code, geometry, inputStrides + axis).lstore(inputAxisStride);
        CpuNormEmitter.geometry(code, geometry, outputStrides + axis).lstore(outputAxisStride);
        CpuNormEmitter.geometry(code, geometry, 0).lstore(inputOrigin);
        CpuNormEmitter.geometry(code, geometry, 1).lstore(outputOrigin);
        CpuNormEmitter.geometry(code, geometry, inputStrides).lstore(inputRowStride);
        CpuNormEmitter.geometry(code, geometry, outputStrides).lstore(outputRowStride);
    }

    private static void load(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int address,
            int represented, int value, boolean intAddress) {
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, address, intAddress);
        if (type == DataType.FLOAT64) code.dstore(represented).dload(represented).dstore(value);
        else if (type == DataType.FLOAT32) code.fstore(represented).fload(represented).fstore(value);
        else if (type == DataType.FLOAT16) code.istore(represented).iload(represented).i2s()
                .invokestatic(ClassDesc.of(Float.class.getName()), "float16ToFloat",
                        MethodTypeDesc.of(TypeKind.FLOAT.upperBound(), TypeKind.SHORT.upperBound()))
                .fstore(value);
        else code.istore(represented).iload(represented).loadConstant(16).ishl()
                .invokestatic(ClassDesc.of(Float.class.getName()), "intBitsToFloat",
                        MethodTypeDesc.of(TypeKind.FLOAT.upperBound(), TypeKind.INT.upperBound()))
                .fstore(value);
    }

    private static void loadFrozen(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int address,
            int represented, int value, boolean intAddress) {
        carriers.loadFrozen(type, specialization.carrierPattern().getFirst(), 0, address,
                intAddress);
        if (type == DataType.FLOAT64) code.dstore(represented).dload(represented).dstore(value);
        else if (type == DataType.FLOAT32) code.fstore(represented).fload(represented)
                .fstore(value);
        else if (type == DataType.FLOAT16) code.istore(represented).iload(represented).i2s()
                .invokestatic(ClassDesc.of(Float.class.getName()), "float16ToFloat",
                        MethodTypeDesc.of(TypeKind.FLOAT.upperBound(), TypeKind.SHORT.upperBound()))
                .fstore(value);
        else code.istore(represented).iload(represented).loadConstant(16).ishl()
                .invokestatic(ClassDesc.of(Float.class.getName()), "intBitsToFloat",
                        MethodTypeDesc.of(TypeKind.FLOAT.upperBound(), TypeKind.INT.upperBound()))
                .fstore(value);
    }

    private static void initializeMaximum(CodeBuilder code, boolean f, int local) {
        if (f) code.loadConstant(Float.NEGATIVE_INFINITY).fstore(local);
        else code.loadConstant(Double.NEGATIVE_INFINITY).dstore(local);
    }
    private static void updateMaximum(CodeBuilder code, boolean f, int value, int maximum,
            int unshifted) {
        var finiteOrPositive = code.newLabel();
        compareConstant(code, f, value, Double.NEGATIVE_INFINITY)
                .branch(Opcode.IFNE, finiteOrPositive);
        code.iconst_1().istore(unshifted).labelBinding(finiteOrPositive);
        if (f) code.fload(maximum).fload(value).invokestatic(MATH, "max",
                MethodTypeDesc.of(TypeKind.FLOAT.upperBound(), TypeKind.FLOAT.upperBound(),
                        TypeKind.FLOAT.upperBound())).fstore(maximum);
        else code.dload(maximum).dload(value).invokestatic(MATH, "max",
                MethodTypeDesc.of(TypeKind.DOUBLE.upperBound(), TypeKind.DOUBLE.upperBound(),
                        TypeKind.DOUBLE.upperBound())).dstore(maximum);
    }

    /**
     * Begins the row's existing normalization passes. A positive-infinity row bypasses EXP and
     * accumulation; the other branches retain shifted finite or literal negative-infinity sites.
     */
    private static Label beginNormalization(CodeBuilder code, boolean f, boolean softmax,
            int maximum, int sum, int logarithm, int unshifted, int positiveInfinity) {
        var ordinary = code.newLabel();
        var normalized = code.newLabel();
        compareConstant(code, f, maximum, Double.POSITIVE_INFINITY).branch(Opcode.IFNE, ordinary);
        if (softmax) {
            code.iconst_1().istore(positiveInfinity);
            constant(code, f, Double.POSITIVE_INFINITY, sum);
        } else constant(code, f, Double.POSITIVE_INFINITY, logarithm);
        zero(code, f, maximum);
        code.branch(Opcode.GOTO, normalized).labelBinding(ordinary);
        if (softmax) code.iconst_0().istore(positiveInfinity);
        var shifted = code.newLabel();
        code.iload(unshifted).branch(Opcode.IFEQ, shifted);
        if (f) code.fload(maximum);
        else code.dload(maximum);
        code.invokestatic(f ? FLOAT : DOUBLE, "isNaN",
                MethodTypeDesc.of(TypeKind.BOOLEAN.upperBound(),
                        (f ? TypeKind.FLOAT : TypeKind.DOUBLE).upperBound()))
                .branch(Opcode.IFNE, shifted);
        zero(code, f, maximum);
        code.labelBinding(shifted);
        return normalized;
    }

    private static void result(CodeBuilder code, boolean f, boolean softmax, int value,
            int maximum, int sum, int logarithm, int target, int positiveInfinity) {
        if (!softmax) {
            subtract(code, f, value, maximum, target);
            subtract(code, f, target, logarithm, target);
            return;
        }
        var ordinary = code.newLabel();
        var zero = code.newLabel();
        var done = code.newLabel();
        code.iload(positiveInfinity).branch(Opcode.IFEQ, ordinary);
        compareConstant(code, f, value, Double.POSITIVE_INFINITY).branch(Opcode.IFNE, zero);
        constant(code, f, Double.NaN, target);
        code.branch(Opcode.GOTO, done).labelBinding(zero);
        zero(code, f, target);
        code.branch(Opcode.GOTO, done).labelBinding(ordinary);
        subtract(code, f, value, maximum, target);
        exponential(code, f, target);
        divide(code, f, target, sum);
        code.labelBinding(done);
    }

    private static void accumulate(CodeBuilder code, boolean f, int value, int sum,
            int compensation, int temporary, int unshifted) {
        var compensated = code.newLabel();
        var done = code.newLabel();
        code.iload(unshifted).branch(Opcode.IFEQ, compensated);
        if (f) code.fload(sum).fload(value).fadd().fstore(sum);
        else code.dload(sum).dload(value).dadd().dstore(sum);
        code.branch(Opcode.GOTO, done).labelBinding(compensated);
        kahan(code, f, value, sum, compensation, temporary);
        code.labelBinding(done);
    }

    private static CodeBuilder compareConstant(CodeBuilder code, boolean f, int local,
            double value) {
        return f ? code.fload(local).loadConstant((float) value).fcmpl()
                : code.dload(local).loadConstant(value).dcmpl();
    }

    private static void constant(CodeBuilder code, boolean f, double value, int target) {
        if (f) code.loadConstant((float) value).fstore(target);
        else code.loadConstant(value).dstore(target);
    }
    private static void zero(CodeBuilder code, boolean f, int local) {
        if (f) code.loadConstant(0.0f).fstore(local); else code.loadConstant(0.0).dstore(local);
    }
    private static void subtract(CodeBuilder code, boolean f, int a, int b, int target) {
        if (f) code.fload(a).fload(b).fsub().fstore(target);
        else code.dload(a).dload(b).dsub().dstore(target);
    }
    private static void divide(CodeBuilder code, boolean f, int target, int denominator) {
        if (f) code.fload(target).fload(denominator).fdiv().fstore(target);
        else code.dload(target).dload(denominator).ddiv().dstore(target);
    }
    private static void exponential(CodeBuilder code, boolean f, int local) {
        if (f) code.fload(local).f2d().invokestatic(MATH, "exp", CpuNormEmitter.doubleUnary())
                .d2f().fstore(local);
        else code.dload(local).invokestatic(MATH, "exp", CpuNormEmitter.doubleUnary()).dstore(local);
    }
    private static void logarithm(CodeBuilder code, boolean f, int source, int target) {
        if (f) code.fload(source).f2d().invokestatic(MATH, "log", CpuNormEmitter.doubleUnary())
                .d2f().fstore(target);
        else code.dload(source).invokestatic(MATH, "log", CpuNormEmitter.doubleUnary()).dstore(target);
    }

    private static void store(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, int address, int value,
            boolean intAddress, boolean frozenLayout) {
        if (type == DataType.FLOAT64) {
            CpuNormEmitter.emitStore(code, carriers, specialization, type, address, value,
                    intAddress, frozenLayout);
            return;
        }
        int represented = value;
        if (type == DataType.FLOAT16) {
            represented = code.allocateLocal(TypeKind.INT);
            code.fload(value).invokestatic(FLOAT, "floatToFloat16",
                    MethodTypeDesc.of(TypeKind.SHORT.upperBound(), TypeKind.FLOAT.upperBound()))
                    .istore(represented);
        } else if (type == DataType.BFLOAT16) {
            represented = code.allocateLocal(TypeKind.INT);
            int bits = code.allocateLocal(TypeKind.INT);
            code.fload(value).invokestatic(FLOAT, "floatToRawIntBits",
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
        if (frozenLayout) carriers.storeFrozen(type, specialization.carrierPattern().getLast(),
                1, address, represented, intAddress);
        else carriers.store(type, specialization.carrierPattern().getLast(), 1, address,
                represented, intAddress);
    }
    private static void kahan(CodeBuilder code, boolean f, int value, int sum,
            int compensation, int temporary) {
        if (f) code.fload(value).fload(compensation).fsub().fstore(value)
                .fload(sum).fload(value).fadd().fstore(temporary)
                .fload(temporary).fload(sum).fsub().fload(value).fsub().fstore(compensation)
                .fload(temporary).fstore(sum);
        else CpuNormEmitter.kahan(code, value, sum, compensation, temporary);
    }
    private static void assignAddress(CodeBuilder code, int base, int address, boolean intAddress) {
        if (intAddress) code.lload(base).l2i().istore(address);
        else code.lload(base).lstore(address);
    }
    private static void incrementAddress(CodeBuilder code, int address, int stride,
            boolean intAddress) {
        if (intAddress) code.iload(address).lload(stride).l2i().iadd().istore(address);
        else code.lload(address).lload(stride).ladd().lstore(address);
    }
    private static void zeroElement(CodeBuilder code, int element, boolean intLoop) {
        if (intLoop) code.loadConstant(0).istore(element);
        else code.loadConstant(0L).lstore(element);
    }
    private static void compareElement(CodeBuilder code, int element, int width, boolean intLoop,
            java.lang.classfile.Label done) {
        if (intLoop) code.iload(element).iload(width).branch(Opcode.IF_ICMPGE, done);
        else code.lload(element).lload(width).lcmp().branch(Opcode.IFGE, done);
    }
    private static void incrementElement(CodeBuilder code, int element, boolean intLoop) {
        if (intLoop) code.iinc(element, 1);
        else code.lload(element).loadConstant(1L).ladd().lstore(element);
    }
    private static void advance(CodeBuilder code, int stride,
            int element, boolean intLoop, int address, boolean intAddress,
            java.lang.classfile.Label loop) {
        incrementAddress(code, address, stride, intAddress);
        incrementElement(code, element, intLoop);
        code.branch(Opcode.GOTO, loop);
    }
}
