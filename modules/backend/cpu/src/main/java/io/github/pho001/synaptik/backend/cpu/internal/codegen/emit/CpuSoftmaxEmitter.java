package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;

/**
 * Emits direct stable three-pass softmax and log-softmax bodies.
 *
 * <p>Kind selection occurs while generating the class. Both bodies find the slice maximum and
 * accumulate shifted exponentials; only log-softmax computes the logarithm of that sum, while
 * softmax proceeds directly to exponential-and-division stores. FLOAT64 uses double sites;
 * FLOAT32, BFLOAT16, and FLOAT16 use float sites with one final low narrowing. A FLOAT32 shift
 * may overflow to negative infinity for admitted finite input, giving zero probability or
 * negative-infinity log-probability. CPU validation rejects non-finite inputs and additionally
 * rejects non-finite shifts for FLOAT64 and BFLOAT16 before this body writes.</p>
 */
public final class CpuSoftmaxEmitter {
    private static final ClassDesc MATH = ClassDesc.of(Math.class.getName());
    private static final ClassDesc FLOAT = ClassDesc.of(Float.class.getName());

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
                    temporary, logarithm);
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
        code.labelBinding(slices);
        if (intLoop) code.iload(intSlice).iload(intEnd).branch(Opcode.IF_ICMPGE, done);
        else code.lload(slice).lload(end).lcmp().branch(Opcode.IFGE, done);
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
        zeroElement(code, element, intLoop);
        assignAddress(code, inputBase, inputAddress, inputArray);
        var maxLoop = code.newLabel(); var maxDone = code.newLabel();
        code.labelBinding(maxLoop);
        compareElement(code, element, width, intLoop, maxDone);
        load(code, carriers, specialization, type, inputAddress, represented, value, inputArray);
        var retain = code.newLabel();
        compare(code, floatWork, value, maximum).branch(Opcode.IFLE, retain);
        copy(code, floatWork, value, maximum); code.labelBinding(retain);
        advance(code, inputAxisStride, element, intLoop, inputAddress, inputArray, maxLoop);
        code.labelBinding(maxDone);

        zero(code, floatWork, sum); zero(code, floatWork, compensation);
        zeroElement(code, element, intLoop);
        assignAddress(code, inputBase, inputAddress, inputArray);
        var sumLoop = code.newLabel(); var sumDone = code.newLabel();
        code.labelBinding(sumLoop);
        compareElement(code, element, width, intLoop, sumDone);
        load(code, carriers, specialization, type, inputAddress, represented, value, inputArray);
        subtract(code, floatWork, value, maximum, shifted);
        exponential(code, floatWork, shifted);
        kahan(code, floatWork, shifted, sum, compensation, temporary);
        advance(code, inputAxisStride, element, intLoop, inputAddress, inputArray, sumLoop);
        code.labelBinding(sumDone);
        if (logSoftmax) logarithm(code, floatWork, sum, logarithm);

        zeroElement(code, element, intLoop);
        assignAddress(code, inputBase, inputAddress, inputArray);
        assignAddress(code, outputBase, outputAddress, outputArray);
        var storeLoop = code.newLabel(); var storeDone = code.newLabel();
        code.labelBinding(storeLoop);
        compareElement(code, element, width, intLoop, storeDone);
        load(code, carriers, specialization, type, inputAddress, represented, value, inputArray);
        subtract(code, floatWork, value, maximum, shifted);
        if (softmax) {
            exponential(code, floatWork, shifted);
            divide(code, floatWork, shifted, sum);
        } else subtract(code, floatWork, shifted, logarithm, shifted);
        store(code, carriers, specialization, type, outputAddress, shifted, outputArray, false);
        incrementAddress(code, inputAddress, inputAxisStride, inputArray);
        incrementAddress(code, outputAddress, outputAxisStride, outputArray);
        incrementElement(code, element, intLoop);
        code.branch(Opcode.GOTO, storeLoop).labelBinding(storeDone);
        if (intLoop) code.iinc(intSlice, 1);
        else code.lload(slice).loadConstant(1L).ladd().lstore(slice);
        code.branch(Opcode.GOTO, slices).labelBinding(done);
    }

    private static void emitFrozen(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, boolean softmax, boolean floatWork,
            boolean generalInput, int start, int end, int row, int rowEnd, int inputAddress,
            int outputAddress, boolean inputArray, boolean outputArray, int element, int represented,
            int value, int maximum, int shifted, int sum, int compensation, int temporary,
            int logarithm) {
        code.lload(start).l2i().istore(row).lload(end).l2i().istore(rowEnd);
        var rows = code.newLabel(); var rowsDone = code.newLabel();
        code.labelBinding(rows).iload(row).iload(rowEnd).branch(Opcode.IF_ICMPGE, rowsDone);
        frozenAddress(code, row, generalInput ? 2049 : 2048, generalInput ? 1 : 0,
                inputAddress, inputArray);
        frozenAddress(code, row, 2048, 0, outputAddress, outputArray);
        initializeMaximum(code, floatWork, maximum);
        code.loadConstant(0).istore(element);
        var maxLoop = code.newLabel(); var maxDone = code.newLabel();
        code.labelBinding(maxLoop).iload(element).loadConstant(2048)
                .branch(Opcode.IF_ICMPGE, maxDone);
        loadFrozen(code, carriers, specialization, type, inputAddress, represented, value,
                inputArray);
        var retain = code.newLabel();
        compare(code, floatWork, value, maximum).branch(Opcode.IFLE, retain);
        copy(code, floatWork, value, maximum); code.labelBinding(retain);
        incrementConstant(code, inputAddress, inputArray);
        code.iinc(element, 1).branch(Opcode.GOTO, maxLoop).labelBinding(maxDone);

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
        kahan(code, floatWork, shifted, sum, compensation, temporary);
        incrementConstant(code, inputAddress, inputArray);
        code.iinc(element, 1).branch(Opcode.GOTO, sumLoop).labelBinding(sumDone);
        if (!softmax) logarithm(code, floatWork, sum, logarithm);

        frozenAddress(code, row, generalInput ? 2049 : 2048, generalInput ? 1 : 0,
                inputAddress, inputArray);
        frozenAddress(code, row, 2048, 0, outputAddress, outputArray);
        code.loadConstant(0).istore(element);
        var storeLoop = code.newLabel(); var storeDone = code.newLabel();
        code.labelBinding(storeLoop).iload(element).loadConstant(2048)
                .branch(Opcode.IF_ICMPGE, storeDone);
        loadFrozen(code, carriers, specialization, type, inputAddress, represented, value,
                inputArray);
        subtract(code, floatWork, value, maximum, shifted);
        if (softmax) { exponential(code, floatWork, shifted); divide(code, floatWork, shifted, sum); }
        else subtract(code, floatWork, shifted, logarithm, shifted);
        store(code, carriers, specialization, type, outputAddress, shifted, outputArray, true);
        incrementConstant(code, inputAddress, inputArray);
        incrementConstant(code, outputAddress, outputArray);
        code.iinc(element, 1).branch(Opcode.GOTO, storeLoop).labelBinding(storeDone);
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
    private static CodeBuilder compare(CodeBuilder code, boolean f, int a, int b) {
        return f ? code.fload(a).fload(b).fcmpl() : code.dload(a).dload(b).dcmpl();
    }
    private static void copy(CodeBuilder code, boolean f, int from, int to) {
        if (f) code.fload(from).fstore(to); else code.dload(from).dstore(to);
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
