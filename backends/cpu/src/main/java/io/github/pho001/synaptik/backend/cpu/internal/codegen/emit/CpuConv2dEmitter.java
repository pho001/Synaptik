package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.constant.ConstantDescs;

/**
 * Emits direct typed grouped NCHW Conv2d complete-output-cell loops.
 *
 * <p>The generated body traverses input channels within the selected group, kernel height, and
 * kernel width in increasing logical order. It loads every weight contribution, represents an
 * out-of-range input coordinate as positive zero before ordinary multiplication, accumulates in
 * FLOAT64 or FLOAT32 as selected by the output type, and narrows BFLOAT16 only at the final store.
 * No Synaptik method is called by generated hot work.</p>
 */
public final class CpuConv2dEmitter {
    private static final ClassDesc MATH = ClassDesc.of(Math.class.getName());
    /** Creates a stateless direct-convolution emitter. */
    public CpuConv2dEmitter() { }

    /**
     * Emits one direct scalar entry over the primitive half-open output-cell range.
     *
     * @param code non-null Class-File method builder to mutate
     * @param specialization non-null exact typed carrier specialization
     * @param ir non-null canonical Conv2d identity
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if boundary, type, strategy, or identity facts disagree
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        String identity = ir.familyIdentity();
        boolean bias = identity.startsWith("conv2d:bias=true:");
        boolean add = identity.contains(":epilogue=ADD:")
                || identity.contains(":epilogue=ADD_RELU:");
        boolean relu = identity.contains(":epilogue=ADD_RELU:");
        int boundaries = specialization.carrierPattern().size();
        if (!identity.startsWith("conv2d:") || boundaries != (bias ? 4 : 3) + (add ? 1 : 0)
                || specialization.executionStrategy().compute()
                    != io.github.pho001.synaptik.backend.cpu.internal.prepare
                        .CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR) {
            throw new IllegalArgumentException("Conv2d generated facts disagree");
        }
        DataType inputType = specialization.boundaryDataTypes().get(0);
        DataType weightType = specialization.boundaryDataTypes().get(1);
        DataType biasType = bias ? specialization.boundaryDataTypes().get(2) : null;
        DataType resultType = specialization.boundaryDataTypes().getLast();
        int outputBoundary = boundaries - 1;
        int externalBoundary = add ? (bias ? 3 : 2) : -1;
        int geometry = boundaries;
        int start = geometry + 1;
        int end = start + 2;
        int inputExtents = boundaries;
        int inputStrides = inputExtents + 4;
        int weightExtents = inputStrides + 4;
        int weightStrides = weightExtents + 4;
        int externalExtents = weightStrides + 4 + (bias ? 2 : 0);
        int externalStrides = externalExtents + 4;
        int outputExtents = externalExtents + (add ? 8 : 0);
        int outputStrides = outputExtents + 4;
        long strideH = CpuNormEmitter.longAfter(identity, ":strideH=");
        long strideW = CpuNormEmitter.longAfter(identity, ":strideW=");
        long padH = CpuNormEmitter.longAfter(identity, ":padH=");
        long padW = CpuNormEmitter.longAfter(identity, ":padW=");
        long dilationH = CpuNormEmitter.longAfter(identity, ":dilationH=");
        long dilationW = CpuNormEmitter.longAfter(identity, ":dilationW=");
        long groups = CpuNormEmitter.longAfter(identity, ":groups=");
        var carriers = new CpuCarrierEmitter(code);

        int cell = code.allocateLocal(TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.LONG);
        int n = code.allocateLocal(TypeKind.LONG);
        int oc = code.allocateLocal(TypeKind.LONG);
        int oh = code.allocateLocal(TypeKind.LONG);
        int ow = code.allocateLocal(TypeKind.LONG);
        int group = code.allocateLocal(TypeKind.LONG);
        int channel = code.allocateLocal(TypeKind.LONG);
        int kh = code.allocateLocal(TypeKind.LONG);
        int kw = code.allocateLocal(TypeKind.LONG);
        int ih = code.allocateLocal(TypeKind.LONG);
        int iw = code.allocateLocal(TypeKind.LONG);
        int inputAddress = code.allocateLocal(TypeKind.LONG);
        int weightAddress = code.allocateLocal(TypeKind.LONG);
        int outputAddress = code.allocateLocal(TypeKind.LONG);
        int representedInput = representedLocal(code, inputType);
        int representedWeight = representedLocal(code, weightType);
        int representedBias = bias ? representedLocal(code, biasType) : -1;
        DataType externalType = add ? specialization.boundaryDataTypes().get(externalBoundary) : null;
        int representedExternal = add ? representedLocal(code, externalType) : -1;
        boolean binary64 = resultType == DataType.FLOAT64;
        TypeKind accumulationKind = binary64 ? TypeKind.DOUBLE : TypeKind.FLOAT;
        int inputValue = code.allocateLocal(accumulationKind);
        int weightValue = code.allocateLocal(accumulationKind);
        int accumulator = code.allocateLocal(accumulationKind);
        int externalValue = add ? code.allocateLocal(accumulationKind) : -1;

        code.lload(start).lstore(cell);
        var cells = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(cells).lload(cell).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.lload(cell).lstore(remaining);
        decodeCoordinate(code, geometry, outputExtents + 3, remaining, ow);
        decodeCoordinate(code, geometry, outputExtents + 2, remaining, oh);
        decodeCoordinate(code, geometry, outputExtents + 1, remaining, oc);
        code.lload(remaining).lstore(n);
        code.lload(oc);
        CpuNormEmitter.geometry(code, geometry, outputExtents + 1)
                .loadConstant(groups).ldiv().ldiv().lstore(group);

        if (bias) {
            int biasAddress = code.allocateLocal(TypeKind.LONG);
            CpuNormEmitter.geometry(code, geometry, 2).lload(oc);
            CpuNormEmitter.geometry(code, geometry, weightStrides + 5).lmul().ladd()
                    .lstore(biasAddress);
            load(code, carriers, specialization, biasType, resultType, 2, biasAddress,
                    representedBias, accumulator);
        } else {
            if (binary64) code.loadConstant(0.0).dstore(accumulator);
            else code.loadConstant(0.0f).fstore(accumulator);
        }

        code.loadConstant(0L).lstore(channel);
        var channels = code.newLabel(); var channelsDone = code.newLabel();
        code.labelBinding(channels).lload(channel);
        CpuNormEmitter.geometry(code, geometry, weightExtents + 1).lcmp()
                .branch(Opcode.IFGE, channelsDone);
        code.loadConstant(0L).lstore(kh);
        var heights = code.newLabel(); var heightsDone = code.newLabel();
        code.labelBinding(heights).lload(kh);
        CpuNormEmitter.geometry(code, geometry, weightExtents + 2).lcmp()
                .branch(Opcode.IFGE, heightsDone);
        code.lload(oh).loadConstant(strideH).lmul().loadConstant(padH).lsub()
                .lload(kh).loadConstant(dilationH).lmul().ladd().lstore(ih);
        code.loadConstant(0L).lstore(kw);
        var widths = code.newLabel(); var widthsDone = code.newLabel();
        code.labelBinding(widths).lload(kw);
        CpuNormEmitter.geometry(code, geometry, weightExtents + 3).lcmp()
                .branch(Opcode.IFGE, widthsDone);
        code.lload(ow).loadConstant(strideW).lmul().loadConstant(padW).lsub()
                .lload(kw).loadConstant(dilationW).lmul().ladd().lstore(iw);

        weightAddress(code, geometry, weightStrides, channel, kh, kw, oc, weightAddress);
        load(code, carriers, specialization, weightType, resultType, 1, weightAddress,
                representedWeight, weightValue);
        var padded = code.newLabel(); var inputReady = code.newLabel();
        code.lload(ih).loadConstant(0L).lcmp().branch(Opcode.IFLT, padded);
        code.lload(iw).loadConstant(0L).lcmp().branch(Opcode.IFLT, padded);
        code.lload(ih); CpuNormEmitter.geometry(code, geometry, inputExtents + 2).lcmp()
                .branch(Opcode.IFGE, padded);
        code.lload(iw); CpuNormEmitter.geometry(code, geometry, inputExtents + 3).lcmp()
                .branch(Opcode.IFGE, padded);
        inputAddress(code, geometry, inputStrides, n, group, channel, ih, iw, inputAddress,
                weightExtents + 1);
        load(code, carriers, specialization, inputType, resultType, 0, inputAddress,
                representedInput, inputValue);
        code.branch(Opcode.GOTO, inputReady).labelBinding(padded);
        if (binary64) code.loadConstant(0.0); else code.loadConstant(0.0f);
        if (binary64) code.dstore(inputValue); else code.fstore(inputValue);
        code.labelBinding(inputReady);
        accumulate(code, resultType, accumulator, inputValue, weightValue);
        code.lload(kw).loadConstant(1L).ladd().lstore(kw).branch(Opcode.GOTO, widths)
                .labelBinding(widthsDone);
        code.lload(kh).loadConstant(1L).ladd().lstore(kh).branch(Opcode.GOTO, heights)
                .labelBinding(heightsDone);
        code.lload(channel).loadConstant(1L).ladd().lstore(channel)
                .branch(Opcode.GOTO, channels).labelBinding(channelsDone);

        if (add) {
            address(code, geometry, externalBoundary, externalStrides, n, oc, oh, ow, inputAddress);
            load(code, carriers, specialization, externalType, resultType, externalBoundary, inputAddress,
                    representedExternal, externalValue);
            if (resultType == DataType.FLOAT64) {
                code.dload(accumulator).dload(externalValue).dadd().dstore(accumulator);
            } else {
                code.fload(accumulator).fload(externalValue).fadd().fstore(accumulator);
            }
            if (relu) {
                if (resultType == DataType.FLOAT64) {
                    code.dload(accumulator).loadConstant(+0.0d)
                            .invokestatic(MATH, "max", MethodTypeDesc.of(ConstantDescs.CD_double,
                                    ConstantDescs.CD_double, ConstantDescs.CD_double))
                            .dstore(accumulator);
                } else {
                    code.fload(accumulator).loadConstant(+0.0f)
                            .invokestatic(MATH, "max", MethodTypeDesc.of(ConstantDescs.CD_float,
                                    ConstantDescs.CD_float, ConstantDescs.CD_float))
                            .fstore(accumulator);
                }
            }
        }

        address(code, geometry, outputBoundary, outputStrides, n, oc, oh, ow, outputAddress);
        int stored = accumulator;
        if (!binary64) {
            stored = code.allocateLocal(TypeKind.DOUBLE);
            code.fload(accumulator).f2d().dstore(stored);
        }
        CpuNormEmitter.emitStore(code, carriers, specialization, resultType, outputBoundary,
                outputAddress, stored, false, true);
        code.lload(cell).loadConstant(1L).ladd().lstore(cell).branch(Opcode.GOTO, cells)
                .labelBinding(done);
    }

    private static int representedLocal(CodeBuilder code, DataType type) {
        return code.allocateLocal(type == DataType.FLOAT64 ? TypeKind.DOUBLE
                : type == DataType.FLOAT32 ? TypeKind.FLOAT : TypeKind.INT);
    }

    private static void load(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, DataType resultType,
            int boundary, int address,
            int represented, int decoded) {
        carriers.loadFrozen(type, specialization.carrierPattern().get(boundary), boundary,
                address, false);
        if (type == DataType.FLOAT64) code.dstore(represented);
        else if (type == DataType.FLOAT32) code.fstore(represented);
        else code.istore(represented);
        if (resultType == DataType.FLOAT64) {
            CpuNormEmitter.decodeRepresented(code, type, represented, decoded);
        } else if (type == DataType.FLOAT32) {
            code.fload(represented).fstore(decoded);
        } else {
            code.iload(represented).loadConstant(16).ishl().invokestatic(
                    ClassDesc.of(Float.class.getName()), "intBitsToFloat",
                    MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int))
                    .fstore(decoded);
        }
    }

    private static void accumulate(CodeBuilder code, DataType result, int accumulator,
            int input, int weight) {
        if (result == DataType.FLOAT64) {
            code.dload(accumulator).dload(input).dload(weight).dmul().dadd().dstore(accumulator);
        } else {
            code.fload(accumulator).fload(input).fload(weight).fmul().fadd().fstore(accumulator);
        }
    }

    private static void decodeCoordinate(CodeBuilder code, int geometry, int extentIndex,
            int remaining, int coordinate) {
        code.lload(remaining); CpuNormEmitter.geometry(code, geometry, extentIndex).lrem()
                .lstore(coordinate);
        code.lload(remaining); CpuNormEmitter.geometry(code, geometry, extentIndex).ldiv()
                .lstore(remaining);
    }

    private static void weightAddress(CodeBuilder code, int geometry, int strides, int channel,
            int kh, int kw, int oc, int target) {
        CpuNormEmitter.geometry(code, geometry, 1).lload(oc);
        CpuNormEmitter.geometry(code, geometry, strides).lmul().ladd()
                .lload(channel); CpuNormEmitter.geometry(code, geometry, strides + 1).lmul().ladd()
                .lload(kh); CpuNormEmitter.geometry(code, geometry, strides + 2).lmul().ladd()
                .lload(kw); CpuNormEmitter.geometry(code, geometry, strides + 3).lmul().ladd()
                .lstore(target);
    }

    private static void inputAddress(CodeBuilder code, int geometry, int strides, int n,
            int group, int channel, int ih, int iw, int target, int channelsPerGroupIndex) {
        CpuNormEmitter.geometry(code, geometry, 0).lload(n);
        CpuNormEmitter.geometry(code, geometry, strides).lmul().ladd()
                .lload(group); CpuNormEmitter.geometry(code, geometry, channelsPerGroupIndex)
                .lmul().lload(channel).ladd();
        CpuNormEmitter.geometry(code, geometry, strides + 1).lmul().ladd()
                .lload(ih); CpuNormEmitter.geometry(code, geometry, strides + 2).lmul().ladd()
                .lload(iw); CpuNormEmitter.geometry(code, geometry, strides + 3).lmul().ladd()
                .lstore(target);
    }

    private static void address(CodeBuilder code, int geometry, int boundary, int strides,
            int n, int c, int h, int w, int target) {
        CpuNormEmitter.geometry(code, geometry, boundary).lload(n);
        CpuNormEmitter.geometry(code, geometry, strides).lmul().ladd().lload(c);
        CpuNormEmitter.geometry(code, geometry, strides + 1).lmul().ladd().lload(h);
        CpuNormEmitter.geometry(code, geometry, strides + 2).lmul().ladd().lload(w);
        CpuNormEmitter.geometry(code, geometry, strides + 3).lmul().ladd().lstore(target);
    }
}
