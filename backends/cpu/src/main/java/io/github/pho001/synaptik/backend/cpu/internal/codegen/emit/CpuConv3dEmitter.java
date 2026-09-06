package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Label;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.constant.ConstantDescs;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;

/**
 * Emits direct typed grouped NCDHW Conv3d scalar or output-width-vector loops.
 *
 * <p>The generated body traverses input channels within the selected group, kernel depth,
 * height, and width in increasing logical order. It loads every weight contribution, represents an
 * out-of-range input coordinate as positive zero before ordinary multiplication, accumulates in
 * FLOAT64 or FLOAT32 as selected by the output type, and narrows BFLOAT16 only at the final store.
 * No Synaptik method is called by generated hot work. The schema-63 vector form is limited to
 * same-typed dense FLOAT32 or FLOAT64 direct convolution. Only the width stride and width
 * dilation must be one; depth and height stride and dilation remain eligible. Preferred-species
 * chunks cover complete in-bounds output-width cells, while padding borders, worker-range
 * fragments, and tails use the same ordered scalar-cell semantics. Every other form, including
 * BFLOAT16, short or interior-free widths, and non-dense access, remains scalar.</p>
 */
public final class CpuConv3dEmitter {
    /** Creates a stateless direct-convolution emitter. */
    public CpuConv3dEmitter() { }

    /**
     * Emits one direct entry over the primitive half-open output-cell range.
     *
     * <p>A schema-52 specialization emits the established scalar body. An eligible schema-63
     * specialization emits preferred-species output-width chunks plus scalar borders, range
     * fragments, and tails. Parallel orchestration reuses the corresponding generated entry over
     * disjoint ranges.</p>
     *
     * @param code non-null Class-File method builder to mutate
     * @param specialization non-null exact typed carrier specialization
     * @param ir non-null canonical Conv3d identity
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if boundary, type, strategy, or identity facts disagree
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        String identity = ir.familyIdentity();
        boolean bias = identity.startsWith("conv3d:bias=true:");
        int boundaries = specialization.carrierPattern().size();
        boolean widthVector = specialization.classIdentitySchema() == 63
                && specialization.executionStrategy().compute()
                    == io.github.pho001.synaptik.backend.cpu.internal.prepare
                        .CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR;
        if (!identity.startsWith("conv3d:") || boundaries != (bias ? 4 : 3)
                || !widthVector && specialization.executionStrategy().compute()
                    != io.github.pho001.synaptik.backend.cpu.internal.prepare
                        .CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR) {
            throw new IllegalArgumentException("Conv3d generated facts disagree");
        }
        if (widthVector) {
            emitWidthVector(code, specialization, ir, bias);
            return;
        }
        DataType inputType = specialization.boundaryDataTypes().get(0);
        DataType weightType = specialization.boundaryDataTypes().get(1);
        DataType biasType = bias ? specialization.boundaryDataTypes().get(2) : null;
        DataType resultType = specialization.boundaryDataTypes().getLast();
        boolean denseArrays = specialization.carrierPattern().stream()
                .allMatch(access -> access != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                && ir.values().stream().allMatch(value -> value.accessPlan().regime()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.DENSE_LINEAR);
        int outputBoundary = boundaries - 1;
        int geometry = boundaries;
        int start = geometry + 1;
        int end = start + 2;
        int inputExtents = boundaries;
        int inputStrides = inputExtents + 5;
        int weightExtents = inputStrides + 5;
        int weightStrides = weightExtents + 5;
        int outputExtents = weightStrides + 5 + (bias ? 2 : 0);
        int outputStrides = outputExtents + 5;
        long strideD = CpuNormEmitter.longAfter(identity, ":strideD=");
        long strideH = CpuNormEmitter.longAfter(identity, ":strideH=");
        long strideW = CpuNormEmitter.longAfter(identity, ":strideW=");
        long padD = CpuNormEmitter.longAfter(identity, ":padD=");
        long padH = CpuNormEmitter.longAfter(identity, ":padH=");
        long padW = CpuNormEmitter.longAfter(identity, ":padW=");
        long dilationD = CpuNormEmitter.longAfter(identity, ":dilationD=");
        long dilationH = CpuNormEmitter.longAfter(identity, ":dilationH=");
        long dilationW = CpuNormEmitter.longAfter(identity, ":dilationW=");
        long groups = CpuNormEmitter.longAfter(identity, ":groups=");
        var carriers = new CpuCarrierEmitter(code);
        int inputBase = geometryLocal(code, geometry, 0);
        int weightBase = geometryLocal(code, geometry, 1);
        int biasBase = bias ? geometryLocal(code, geometry, 2) : -1;
        int outputBase = geometryLocal(code, geometry, outputBoundary);
        int inD = geometryLocal(code, geometry, inputExtents + 2);
        int inH = geometryLocal(code, geometry, inputExtents + 3);
        int inW = geometryLocal(code, geometry, inputExtents + 4);
        int inSN = geometryLocal(code, geometry, inputStrides);
        int inSC = geometryLocal(code, geometry, inputStrides + 1);
        int inSD = geometryLocal(code, geometry, inputStrides + 2);
        int inSH = geometryLocal(code, geometry, inputStrides + 3);
        int inSW = geometryLocal(code, geometry, inputStrides + 4);
        int channelsPerGroup = geometryLocal(code, geometry, weightExtents + 1);
        int kernelD = geometryLocal(code, geometry, weightExtents + 2);
        int kernelH = geometryLocal(code, geometry, weightExtents + 3);
        int kernelW = geometryLocal(code, geometry, weightExtents + 4);
        int weightSO = geometryLocal(code, geometry, weightStrides);
        int weightSC = geometryLocal(code, geometry, weightStrides + 1);
        int weightSD = geometryLocal(code, geometry, weightStrides + 2);
        int weightSH = geometryLocal(code, geometry, weightStrides + 3);
        int weightSW = geometryLocal(code, geometry, weightStrides + 4);
        int biasStride = bias ? geometryLocal(code, geometry, weightStrides + 6) : -1;
        int outC = geometryLocal(code, geometry, outputExtents + 1);
        int outD = geometryLocal(code, geometry, outputExtents + 2);
        int outH = geometryLocal(code, geometry, outputExtents + 3);
        int outW = geometryLocal(code, geometry, outputExtents + 4);
        int outSN = geometryLocal(code, geometry, outputStrides);
        int outSC = geometryLocal(code, geometry, outputStrides + 1);
        int outSD = geometryLocal(code, geometry, outputStrides + 2);
        int outSH = geometryLocal(code, geometry, outputStrides + 3);
        int outSW = geometryLocal(code, geometry, outputStrides + 4);

        int cell = code.allocateLocal(TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.LONG);
        int n = code.allocateLocal(TypeKind.LONG);
        int oc = code.allocateLocal(TypeKind.LONG);
        int od = code.allocateLocal(TypeKind.LONG);
        int oh = code.allocateLocal(TypeKind.LONG);
        int ow = code.allocateLocal(TypeKind.LONG);
        int group = code.allocateLocal(TypeKind.LONG);
        int channel = code.allocateLocal(TypeKind.LONG);
        int kd = code.allocateLocal(TypeKind.LONG);
        int kh = code.allocateLocal(TypeKind.LONG);
        int kw = code.allocateLocal(TypeKind.LONG);
        int id = code.allocateLocal(TypeKind.LONG);
        int ih = code.allocateLocal(TypeKind.LONG);
        int iw = code.allocateLocal(TypeKind.LONG);
        int inputAddress = code.allocateLocal(denseArrays ? TypeKind.INT : TypeKind.LONG);
        int weightAddress = code.allocateLocal(denseArrays ? TypeKind.INT : TypeKind.LONG);
        int outputAddress = code.allocateLocal(denseArrays ? TypeKind.INT : TypeKind.LONG);
        int representedInput = representedLocal(code, inputType);
        int representedWeight = representedLocal(code, weightType);
        int representedBias = bias ? representedLocal(code, biasType) : -1;
        boolean binary64 = resultType == DataType.FLOAT64;
        TypeKind accumulationKind = binary64 ? TypeKind.DOUBLE : TypeKind.FLOAT;
        int inputValue = code.allocateLocal(accumulationKind);
        int weightValue = code.allocateLocal(accumulationKind);
        int accumulator = code.allocateLocal(accumulationKind);

        code.lload(start).lstore(cell);
        var cells = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(cells).lload(cell).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.lload(cell).lstore(remaining);
        decodeCoordinate(code, outW, remaining, ow);
        decodeCoordinate(code, outH, remaining, oh);
        decodeCoordinate(code, outD, remaining, od);
        decodeCoordinate(code, outC, remaining, oc);
        code.lload(remaining).lstore(n);
        code.lload(oc);
        code.lload(outC).loadConstant(groups).ldiv().ldiv().lstore(group);

        if (bias) {
            int biasAddress = code.allocateLocal(denseArrays ? TypeKind.INT : TypeKind.LONG);
            if (denseArrays) {
                code.lload(biasBase).l2i().lload(oc).l2i().iadd().istore(biasAddress);
            } else {
                code.lload(biasBase).lload(oc).lload(biasStride).lmul().ladd().lstore(biasAddress);
            }
            load(code, carriers, specialization, biasType, resultType, 2, biasAddress,
                    representedBias, accumulator,denseArrays);
        } else {
            if (binary64) {
                code.loadConstant(0.0).dstore(accumulator);
            } else {
                code.loadConstant(0.0f).fstore(accumulator);
            }
        }

        code.loadConstant(0L).lstore(channel);
        var channels = code.newLabel();
        var channelsDone = code.newLabel();
        code.labelBinding(channels).lload(channel);
        code.lload(channelsPerGroup).lcmp()
                .branch(Opcode.IFGE, channelsDone);
        code.loadConstant(0L).lstore(kd);
        var depths = code.newLabel();
        var depthsDone = code.newLabel();
        code.labelBinding(depths).lload(kd);
        code.lload(kernelD).lcmp()
                .branch(Opcode.IFGE, depthsDone);
        code.lload(od).loadConstant(strideD).lmul().loadConstant(padD).lsub()
                .lload(kd).loadConstant(dilationD).lmul().ladd().lstore(id);
        code.loadConstant(0L).lstore(kh);
        var heights = code.newLabel();
        var heightsDone = code.newLabel();
        code.labelBinding(heights).lload(kh);
        code.lload(kernelH).lcmp()
                .branch(Opcode.IFGE, heightsDone);
        code.lload(oh).loadConstant(strideH).lmul().loadConstant(padH).lsub()
                .lload(kh).loadConstant(dilationH).lmul().ladd().lstore(ih);
        code.loadConstant(0L).lstore(kw);
        var widths = code.newLabel();
        var widthsDone = code.newLabel();
        code.labelBinding(widths).lload(kw);
        code.lload(kernelW).lcmp()
                .branch(Opcode.IFGE, widthsDone);
        code.lload(ow).loadConstant(strideW).lmul().loadConstant(padW).lsub()
                .lload(kw).loadConstant(dilationW).lmul().ladd().lstore(iw);

        if (denseArrays) {
            weightAddressDense(code, weightBase, channelsPerGroup, kernelD, kernelH, kernelW,
                    channel, kd, kh, kw, oc, weightAddress);
        } else {
            weightAddress(code, weightBase, weightSO, weightSC, weightSD, weightSH, weightSW,
                    channel, kd, kh, kw, oc, weightAddress);
        }
        load(code, carriers, specialization, weightType, resultType, 1, weightAddress,
                representedWeight, weightValue,denseArrays);
        var padded = code.newLabel();
        var inputReady = code.newLabel();
        code.lload(id).loadConstant(0L).lcmp().branch(Opcode.IFLT, padded);
        code.lload(ih).loadConstant(0L).lcmp().branch(Opcode.IFLT, padded);
        code.lload(iw).loadConstant(0L).lcmp().branch(Opcode.IFLT, padded);
        code.lload(id).lload(inD).lcmp()
                .branch(Opcode.IFGE, padded);
        code.lload(ih).lload(inH).lcmp()
                .branch(Opcode.IFGE, padded);
        code.lload(iw).lload(inW).lcmp()
                .branch(Opcode.IFGE, padded);
        if (denseArrays) {
            inputAddressDense(code, inputBase, channelsPerGroup, inD, inH, inW, groups, n,
                    group, channel, id, ih, iw, inputAddress);
        } else {
            inputAddress(code, inputBase, inSN, inSC, inSD, inSH, inSW, n, group, channel,
                    id, ih, iw, inputAddress, channelsPerGroup);
        }
        load(code, carriers, specialization, inputType, resultType, 0, inputAddress,
                representedInput, inputValue,denseArrays);
        code.branch(Opcode.GOTO, inputReady).labelBinding(padded);
        if (binary64) {
            code.loadConstant(0.0);
            code.dstore(inputValue);
        } else {
            code.loadConstant(0.0f);
            code.fstore(inputValue);
        }
        code.labelBinding(inputReady);
        accumulate(code, resultType, accumulator, inputValue, weightValue);
        code.lload(kw).loadConstant(1L).ladd().lstore(kw).branch(Opcode.GOTO, widths)
                .labelBinding(widthsDone);
        code.lload(kh).loadConstant(1L).ladd().lstore(kh).branch(Opcode.GOTO, heights)
                .labelBinding(heightsDone);
        code.lload(kd).loadConstant(1L).ladd().lstore(kd).branch(Opcode.GOTO, depths)
                .labelBinding(depthsDone);
        code.lload(channel).loadConstant(1L).ladd().lstore(channel)
                .branch(Opcode.GOTO, channels).labelBinding(channelsDone);

        if (denseArrays) {
            outputAddressDense(code, outputBase, outC, outD, outH, outW, n, oc, od, oh, ow,
                    outputAddress);
        } else {
            address(code, outputBase, outSN, outSC, outSD, outSH, outSW, n, oc, od, oh, ow,
                    outputAddress);
        }
        int stored = accumulator;
        if (!binary64) {
            stored = code.allocateLocal(TypeKind.DOUBLE);
            code.fload(accumulator).f2d().dstore(stored);
        }
        CpuNormEmitter.emitStore(code, carriers, specialization, resultType, outputBoundary,
                outputAddress, stored, denseArrays, true);
        code.lload(cell).loadConstant(1L).ladd().lstore(cell).branch(Opcode.GOTO, cells)
                .labelBinding(done);
    }

    /**
     * Emits the schema-63 dense-array NCDHW width-vector realization. Width stride and dilation
     * are one; depth and height stride and dilation remain arbitrary positive admitted values.
     */
    private static void emitWidthVector(CodeBuilder code, CpuKernelSpecialization specialization,
            CpuKernelIr ir, boolean bias) {
        DataType type = specialization.boundaryDataTypes().getFirst();
        CpuKernelSpecialization.CarrierAccess array = type == DataType.FLOAT32
                ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                : CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
        String identity = ir.familyIdentity();
        if (type != specialization.boundaryDataTypes().getLast()
                || type != DataType.FLOAT32 && type != DataType.FLOAT64
                || specialization.boundaryDataTypes().stream().anyMatch(t -> t != type)
                || specialization.carrierPattern().stream().anyMatch(a -> a != array
                        && a != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                || CpuNormEmitter.longAfter(identity, ":strideW=") != 1L
                || CpuNormEmitter.longAfter(identity, ":dilationW=") != 1L) throw new IllegalArgumentException(
                        "Conv3d width vector requires same-typed dense carriers and unit width stride/dilation");
        if (specialization.carrierPattern().contains(
                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)) {
            emitWidthVectorGeneral(code, specialization, ir, bias, type);
            return;
        }
        int boundaries = specialization.carrierPattern().size();
        int geometry = boundaries;
        int start = geometry + 1;
        int end = start + 2;
        int inputExtents = boundaries;
        int inputStrides = inputExtents + 5;
        int weightExtents = inputStrides + 5;
        int weightStrides = weightExtents + 5;
        int outputExtents = weightStrides + 5 + (bias ? 2 : 0);
        int outputBoundary = boundaries - 1;
        int inputBase = intGeometry(code, geometry, 0);
        int weightBase = intGeometry(code, geometry, 1);
        int biasBase = bias ? intGeometry(code, geometry, 2) : -1;
        int outputBase = intGeometry(code, geometry, outputBoundary);
        int inputChannels = intGeometry(code, geometry, inputExtents + 1);
        int inputDepth = intGeometry(code, geometry, inputExtents + 2);
        int inputHeight = intGeometry(code, geometry, inputExtents + 3);
        int inputWidth = intGeometry(code, geometry, inputExtents + 4);
        int channelsPerGroup = intGeometry(code, geometry, weightExtents + 1);
        int kernelDepth = intGeometry(code, geometry, weightExtents + 2);
        int kernelHeight = intGeometry(code, geometry, weightExtents + 3);
        int kernelWidth = intGeometry(code, geometry, weightExtents + 4);
        int outputChannels = intGeometry(code, geometry, outputExtents + 1);
        int outputDepth = intGeometry(code, geometry, outputExtents + 2);
        int outputHeight = intGeometry(code, geometry, outputExtents + 3);
        int outputWidth = intGeometry(code, geometry, outputExtents + 4);
        int ordinal = code.allocateLocal(TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.LONG);
        int n = code.allocateLocal(TypeKind.INT);
        int oc = code.allocateLocal(TypeKind.INT);
        int od = code.allocateLocal(TypeKind.INT);
        int oh = code.allocateLocal(TypeKind.INT);
        int ow = code.allocateLocal(TypeKind.INT);
        int group = code.allocateLocal(TypeKind.INT);
        int inputGroupBase = code.allocateLocal(TypeKind.INT);
        int weightChannelBase = code.allocateLocal(TypeKind.INT);
        int outputChannelBase = code.allocateLocal(TypeKind.INT);
        int depthOrigin = code.allocateLocal(TypeKind.INT);
        int heightOrigin = code.allocateLocal(TypeKind.INT);
        int inputCursor = code.allocateLocal(TypeKind.INT);
        int outputCursor = code.allocateLocal(TypeKind.INT);
        int planeInterior = code.allocateLocal(TypeKind.INT);
        int channel = code.allocateLocal(TypeKind.INT);
        int kd = code.allocateLocal(TypeKind.INT);
        int kh = code.allocateLocal(TypeKind.INT);
        int kw = code.allocateLocal(TypeKind.INT);
        int id = code.allocateLocal(TypeKind.INT);
        int ih = code.allocateLocal(TypeKind.INT);
        int iw = code.allocateLocal(TypeKind.INT);
        int inputAddress = code.allocateLocal(TypeKind.INT);
        int weightAddress = code.allocateLocal(TypeKind.INT);
        TypeKind scalar = type == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT;
        int biasValue = code.allocateLocal(scalar);
        int accumulator = code.allocateLocal(scalar);
        int vectorAccumulator = code.allocateLocal(TypeKind.REFERENCE);
        int vectorInput = code.allocateLocal(TypeKind.REFERENCE);
        int vectorWeight = code.allocateLocal(TypeKind.REFERENCE);
        int lanes = specialization.vectorSpeciesBitSize() / type.bitWidth();
        int groups = Math.toIntExact(CpuNormEmitter.longAfter(identity, ":groups="));
        int strideDepth = Math.toIntExact(CpuNormEmitter.longAfter(identity, ":strideD="));
        int strideHeight = Math.toIntExact(CpuNormEmitter.longAfter(identity, ":strideH="));
        int paddingDepth = Math.toIntExact(CpuNormEmitter.longAfter(identity, ":padD="));
        int paddingHeight = Math.toIntExact(CpuNormEmitter.longAfter(identity, ":padH="));
        int paddingWidth = Math.toIntExact(CpuNormEmitter.longAfter(identity, ":padW="));
        int dilationDepth = Math.toIntExact(CpuNormEmitter.longAfter(identity, ":dilationD="));
        int dilationHeight = Math.toIntExact(CpuNormEmitter.longAfter(identity, ":dilationH="));
        ClassDesc vector = type == DataType.FLOAT64
                ? ClassDesc.of("jdk.incubator.vector.DoubleVector")
                : ClassDesc.of("jdk.incubator.vector.FloatVector");
        ClassDesc species = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
        ClassDesc vectorBase = ClassDesc.of("jdk.incubator.vector.Vector");
        CpuCarrierEmitter carriers = new CpuCarrierEmitter(code);
        carriers.prepareVectorSpecies(type);
        Label done = code.newLabel();
        Label batches = code.newLabel();
        Label outputChannelsLoop = code.newLabel();
        Label depths = code.newLabel();
        Label rows = code.newLabel();
        Label widths = code.newLabel();
        Label scalarCell = code.newLabel();
        Label advance = code.newLabel();
        code.lload(start).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.lload(start).lstore(ordinal);
        code.lload(start).lstore(remaining);
        decodeInt(code, remaining, outputWidth, ow);
        decodeInt(code, remaining, outputHeight, oh);
        decodeInt(code, remaining, outputDepth, od);
        decodeInt(code, remaining, outputChannels, oc);
        code.lload(remaining).l2i().istore(n);
        code.labelBinding(batches).lload(ordinal).lload(end).lcmp()
                .branch(Opcode.IFGE, done);
        code.labelBinding(outputChannelsLoop).lload(ordinal).lload(end).lcmp()
                .branch(Opcode.IFGE, done);
        code.iload(oc).iload(outputChannels).loadConstant(groups).idiv().idiv().istore(group);
        code.iload(inputBase).iload(n).iload(inputChannels).imul()
                .iload(group).iload(channelsPerGroup).imul().iadd()
                .iload(inputDepth).imul().iload(inputHeight).imul().iload(inputWidth).imul()
                .iadd().istore(inputGroupBase);
        code.iload(weightBase).iload(oc).iload(channelsPerGroup).imul()
                .iload(kernelDepth).imul().iload(kernelHeight).imul().iload(kernelWidth).imul()
                .iadd().istore(weightChannelBase);
        code.iload(outputBase).iload(n).iload(outputChannels).imul().iload(oc).iadd()
                .iload(outputDepth).imul().iload(outputHeight).imul().iload(outputWidth).imul()
                .iadd().istore(outputChannelBase);
        if (bias) {
            code.aload(2).iload(biasBase).iload(oc).iadd();
            if (type == DataType.FLOAT64) code.daload().dstore(biasValue);
            else code.faload().fstore(biasValue);
        } else if (type == DataType.FLOAT64) {
            code.loadConstant(0d).dstore(biasValue);
        } else {
            code.loadConstant(0f).fstore(biasValue);
        }
        code.labelBinding(depths).lload(ordinal).lload(end).lcmp()
                .branch(Opcode.IFGE, done);
        code.iload(od).loadConstant(strideDepth).imul().loadConstant(paddingDepth).isub()
                .istore(depthOrigin);
        code.labelBinding(rows).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.iload(oh).loadConstant(strideHeight).imul().loadConstant(paddingHeight).isub()
                .istore(heightOrigin);
        code.iload(inputGroupBase).iload(depthOrigin).iload(inputHeight).imul()
                .iload(inputWidth).imul().iadd().iload(heightOrigin).iload(inputWidth).imul()
                .iadd().loadConstant(paddingWidth).isub().iload(ow).iadd().istore(inputCursor);
        code.iload(outputChannelBase).iload(od).iload(outputHeight).imul()
                .iload(outputWidth).imul().iadd().iload(oh).iload(outputWidth).imul().iadd()
                .iload(ow).iadd().istore(outputCursor);
        Label notInterior = code.newLabel();
        Label interiorReady = code.newLabel();
        code.iload(depthOrigin).branch(Opcode.IFLT, notInterior);
        code.iload(depthOrigin).iload(kernelDepth).loadConstant(1).isub()
                .loadConstant(dilationDepth).imul().iadd().iload(inputDepth)
                .branch(Opcode.IF_ICMPGE, notInterior);
        code.iload(heightOrigin).branch(Opcode.IFLT, notInterior);
        code.iload(heightOrigin).iload(kernelHeight).loadConstant(1).isub()
                .loadConstant(dilationHeight).imul().iadd().iload(inputHeight)
                .branch(Opcode.IF_ICMPGE, notInterior);
        code.loadConstant(1).istore(planeInterior).branch(Opcode.GOTO, interiorReady);
        code.labelBinding(notInterior).loadConstant(0).istore(planeInterior);
        code.labelBinding(interiorReady);
        code.labelBinding(widths).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.iload(planeInterior).branch(Opcode.IFEQ, scalarCell);
        code.iload(ow).loadConstant(lanes).iadd().iload(outputWidth)
                .branch(Opcode.IF_ICMPGT, scalarCell);
        code.lload(ordinal).loadConstant((long) lanes).ladd().lload(end).lcmp()
                .branch(Opcode.IFGT, scalarCell);
        code.iload(ow).loadConstant(paddingWidth).isub().branch(Opcode.IFLT, scalarCell);
        code.iload(ow).loadConstant(paddingWidth).isub().loadConstant(lanes).iadd()
                .iload(kernelWidth).iadd().loadConstant(2).isub().iload(inputWidth)
                .branch(Opcode.IF_ICMPGE, scalarCell);
        code.getstatic(vector, "SPECIES_PREFERRED", species);
        if (type == DataType.FLOAT64) code.dload(biasValue);
        else code.fload(biasValue);
        code.invokestatic(vector, "broadcast", MethodTypeDesc.of(vector, species,
                type == DataType.FLOAT64 ? ConstantDescs.CD_double : ConstantDescs.CD_float))
                .astore(vectorAccumulator);
        emitVectorAccumulation(code, carriers, type, array, vector, vectorBase,
                channelsPerGroup, kernelDepth, kernelHeight, kernelWidth, inputDepth,
                inputHeight, inputWidth, dilationDepth, dilationHeight, inputCursor,
                weightChannelBase, channel, kd, kh, kw, inputAddress, weightAddress,
                vectorAccumulator, vectorInput, vectorWeight);
        carriers.vectorStore(type, array, outputBoundary, outputCursor, vectorAccumulator, true);
        code.iinc(ow, lanes).iinc(inputCursor, lanes).iinc(outputCursor, lanes)
                .lload(ordinal).loadConstant((long) lanes).ladd().lstore(ordinal)
                .branch(Opcode.GOTO, advance);
        code.labelBinding(scalarCell);
        emitScalarAccumulation(code, type, inputGroupBase, weightChannelBase,
                channelsPerGroup, kernelDepth, kernelHeight, kernelWidth, inputDepth,
                inputHeight, inputWidth, depthOrigin, heightOrigin, ow, paddingWidth,
                dilationDepth, dilationHeight, channel, kd, kh, kw, id, ih, iw,
                inputAddress, weightAddress, biasValue, accumulator);
        code.aload(outputBoundary).iload(outputCursor);
        if (type == DataType.FLOAT64) code.dload(accumulator).dastore();
        else code.fload(accumulator).fastore();
        code.iinc(ow, 1).iinc(inputCursor, 1).iinc(outputCursor, 1)
                .lload(ordinal).loadConstant(1L).ladd().lstore(ordinal);
        code.labelBinding(advance).iload(ow).iload(outputWidth)
                .branch(Opcode.IF_ICMPLT, widths);
        code.loadConstant(0).istore(ow).iinc(oh, 1);
        code.iload(oh).iload(outputHeight).branch(Opcode.IF_ICMPLT, rows);
        code.loadConstant(0).istore(oh).iinc(od, 1);
        code.iload(od).iload(outputDepth).branch(Opcode.IF_ICMPLT, depths);
        code.loadConstant(0).istore(od).iinc(oc, 1);
        code.iload(oc).iload(outputChannels).branch(Opcode.IF_ICMPLT, outputChannelsLoop);
        code.loadConstant(0).istore(oc).iinc(n, 1).branch(Opcode.GOTO, batches);
        code.labelBinding(done);
    }

    /**
     * Emits the schema-63 segment or ordered mixed-carrier counterpart with checked long
     * addresses and the same width-only eligibility and scalar-fragment contract.
     */
    private static void emitWidthVectorGeneral(CodeBuilder code,
            CpuKernelSpecialization specialization, CpuKernelIr ir, boolean bias, DataType type) {
        int boundaries = specialization.carrierPattern().size();
        int geometry = boundaries;
        int start = geometry + 1;
        int end = start + 2;
        int inputExtents = boundaries;
        int inputStrides = inputExtents + 5;
        int weightExtents = inputStrides + 5;
        int weightStrides = weightExtents + 5;
        int outputExtents = weightStrides + 5 + (bias ? 2 : 0);
        int outputStrides = outputExtents + 5;
        int outputBoundary = boundaries - 1;
        int inputBase = longGeometry(code, geometry, 0);
        int weightBase = longGeometry(code, geometry, 1);
        int biasBase = bias ? longGeometry(code, geometry, 2) : -1;
        int outputBase = longGeometry(code, geometry, outputBoundary);
        int inputDepth = longGeometry(code, geometry, inputExtents + 2);
        int inputHeight = longGeometry(code, geometry, inputExtents + 3);
        int inputWidth = longGeometry(code, geometry, inputExtents + 4);
        int inputBatchStride = longGeometry(code, geometry, inputStrides);
        int inputChannelStride = longGeometry(code, geometry, inputStrides + 1);
        int inputDepthStride = longGeometry(code, geometry, inputStrides + 2);
        int inputHeightStride = longGeometry(code, geometry, inputStrides + 3);
        int inputWidthStride = longGeometry(code, geometry, inputStrides + 4);
        int channelsPerGroup = longGeometry(code, geometry, weightExtents + 1);
        int kernelDepth = longGeometry(code, geometry, weightExtents + 2);
        int kernelHeight = longGeometry(code, geometry, weightExtents + 3);
        int kernelWidth = longGeometry(code, geometry, weightExtents + 4);
        int weightOutputStride = longGeometry(code, geometry, weightStrides);
        int weightChannelStride = longGeometry(code, geometry, weightStrides + 1);
        int weightDepthStride = longGeometry(code, geometry, weightStrides + 2);
        int weightHeightStride = longGeometry(code, geometry, weightStrides + 3);
        int weightWidthStride = longGeometry(code, geometry, weightStrides + 4);
        int biasStride = bias ? longGeometry(code, geometry, weightStrides + 6) : -1;
        int outputChannels = longGeometry(code, geometry, outputExtents + 1);
        int outputDepth = longGeometry(code, geometry, outputExtents + 2);
        int outputHeight = longGeometry(code, geometry, outputExtents + 3);
        int outputWidth = longGeometry(code, geometry, outputExtents + 4);
        int outputBatchStride = longGeometry(code, geometry, outputStrides);
        int outputChannelStride = longGeometry(code, geometry, outputStrides + 1);
        int outputDepthStride = longGeometry(code, geometry, outputStrides + 2);
        int outputHeightStride = longGeometry(code, geometry, outputStrides + 3);
        int outputWidthStride = longGeometry(code, geometry, outputStrides + 4);
        int ordinal = code.allocateLocal(TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.LONG);
        int n = code.allocateLocal(TypeKind.LONG);
        int oc = code.allocateLocal(TypeKind.LONG);
        int od = code.allocateLocal(TypeKind.LONG);
        int oh = code.allocateLocal(TypeKind.LONG);
        int ow = code.allocateLocal(TypeKind.LONG);
        int group = code.allocateLocal(TypeKind.LONG);
        int inputBatchGroupBase = code.allocateLocal(TypeKind.LONG);
        int weightChannelBase = code.allocateLocal(TypeKind.LONG);
        int outputChannelBase = code.allocateLocal(TypeKind.LONG);
        int inputDepthOrigin = code.allocateLocal(TypeKind.LONG);
        int inputHeightOrigin = code.allocateLocal(TypeKind.LONG);
        int inputWidthCursor = code.allocateLocal(TypeKind.LONG);
        int outputCursor = code.allocateLocal(TypeKind.LONG);
        int planeInterior = code.allocateLocal(TypeKind.INT);
        int channel = code.allocateLocal(TypeKind.LONG);
        int kd = code.allocateLocal(TypeKind.LONG);
        int kh = code.allocateLocal(TypeKind.LONG);
        int kw = code.allocateLocal(TypeKind.LONG);
        int id = code.allocateLocal(TypeKind.LONG);
        int ih = code.allocateLocal(TypeKind.LONG);
        int iw = code.allocateLocal(TypeKind.LONG);
        int inputAddress = code.allocateLocal(TypeKind.LONG);
        int weightAddress = code.allocateLocal(TypeKind.LONG);
        TypeKind scalarKind = type == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT;
        int biasValue = code.allocateLocal(scalarKind);
        int inputValue = code.allocateLocal(scalarKind);
        int weightValue = code.allocateLocal(scalarKind);
        int scalarAccumulator = code.allocateLocal(scalarKind);
        int vectorAccumulator = code.allocateLocal(TypeKind.REFERENCE);
        int vectorInput = code.allocateLocal(TypeKind.REFERENCE);
        int vectorWeight = code.allocateLocal(TypeKind.REFERENCE);
        String identity = ir.familyIdentity();
        long groups = CpuNormEmitter.longAfter(identity, ":groups=");
        long strideDepth = CpuNormEmitter.longAfter(identity, ":strideD=");
        long strideHeight = CpuNormEmitter.longAfter(identity, ":strideH=");
        long paddingDepth = CpuNormEmitter.longAfter(identity, ":padD=");
        long paddingHeight = CpuNormEmitter.longAfter(identity, ":padH=");
        long paddingWidth = CpuNormEmitter.longAfter(identity, ":padW=");
        long dilationDepth = CpuNormEmitter.longAfter(identity, ":dilationD=");
        long dilationHeight = CpuNormEmitter.longAfter(identity, ":dilationH=");
        long lanes = specialization.vectorSpeciesBitSize() / type.bitWidth();
        ClassDesc vectorClass = type == DataType.FLOAT64 ? ClassDesc.of(DoubleVector.class.getName())
                : ClassDesc.of(FloatVector.class.getName());
        CpuCarrierEmitter carriers = new CpuCarrierEmitter(code);
        carriers.prepareVectorSpecies(type);
        carriers.prepareVectorByteOrder();
        code.aconst_null().astore(vectorAccumulator).aconst_null().astore(vectorInput)
                .aconst_null().astore(vectorWeight);
        Label done = code.newLabel();
        Label batches = code.newLabel();
        Label outputChannelLoop = code.newLabel();
        Label depths = code.newLabel();
        Label rows = code.newLabel();
        Label widths = code.newLabel();
        Label scalarCell = code.newLabel();
        Label advanceWidth = code.newLabel();
        code.lload(start).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.lload(start).lstore(ordinal);
        code.lload(start).lstore(remaining);
        decodeLong(code, remaining, outputWidth, ow);
        decodeLong(code, remaining, outputHeight, oh);
        decodeLong(code, remaining, outputDepth, od);
        decodeLong(code, remaining, outputChannels, oc);
        code.lload(remaining).lstore(n);
        code.labelBinding(batches).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.labelBinding(outputChannelLoop).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        // Division is intentionally cold: group state changes only with the output channel.
        code.lload(oc).lload(outputChannels).loadConstant(groups).ldiv().ldiv().lstore(group);
        code.lload(inputBase).lload(n).lload(inputBatchStride).lmul().ladd()
                .lload(group).lload(channelsPerGroup).lmul().lload(inputChannelStride).lmul().ladd()
                .lstore(inputBatchGroupBase);
        code.lload(weightBase).lload(oc).lload(weightOutputStride).lmul().ladd().lstore(weightChannelBase);
        code.lload(outputBase).lload(n).lload(outputBatchStride).lmul().ladd()
                .lload(oc).lload(outputChannelStride).lmul().ladd().lstore(outputChannelBase);
        if (bias) {
            code.lload(biasBase).lload(oc).lload(biasStride).lmul().ladd().lstore(weightAddress);
            carriers.loadFrozen(type, specialization.carrierPattern().get(2), 2, weightAddress, false);
            if (type == DataType.FLOAT64) code.dstore(biasValue);
            else code.fstore(biasValue);
        } else if (type == DataType.FLOAT64) {
            code.loadConstant(0.0d).dstore(biasValue);
        } else {
            code.loadConstant(0.0f).fstore(biasValue);
        }
        code.labelBinding(depths).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.lload(od).loadConstant(strideDepth).lmul().loadConstant(paddingDepth).lsub().lstore(inputDepthOrigin);
        code.labelBinding(rows).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.lload(oh).loadConstant(strideHeight).lmul().loadConstant(paddingHeight).lsub().lstore(inputHeightOrigin);
        code.lload(inputBatchGroupBase).lload(inputDepthOrigin).lload(inputDepthStride).lmul().ladd()
                .lload(inputHeightOrigin).lload(inputHeightStride).lmul().ladd()
                .loadConstant(paddingWidth).lload(inputWidthStride).lmul().lsub()
                .lload(ow).lload(inputWidthStride).lmul().ladd().lstore(inputWidthCursor);
        code.lload(outputChannelBase).lload(od).lload(outputDepthStride).lmul().ladd()
                .lload(oh).lload(outputHeightStride).lmul().ladd()
                .lload(ow).lload(outputWidthStride).lmul().ladd().lstore(outputCursor);
        Label notInterior = code.newLabel(), interiorReady = code.newLabel();
        code.lload(inputDepthOrigin).loadConstant(0L).lcmp().branch(Opcode.IFLT, notInterior);
        code.lload(inputDepthOrigin).lload(kernelDepth).loadConstant(1L).lsub().loadConstant(dilationDepth).lmul().ladd()
                .lload(inputDepth).lcmp().branch(Opcode.IFGE, notInterior);
        code.lload(inputHeightOrigin).loadConstant(0L).lcmp().branch(Opcode.IFLT, notInterior);
        code.lload(inputHeightOrigin).lload(kernelHeight).loadConstant(1L).lsub().loadConstant(dilationHeight).lmul().ladd()
                .lload(inputHeight).lcmp().branch(Opcode.IFGE, notInterior);
        code.loadConstant(1).istore(planeInterior).branch(Opcode.GOTO, interiorReady)
                .labelBinding(notInterior).loadConstant(0).istore(planeInterior).labelBinding(interiorReady);
        code.labelBinding(widths).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.iload(planeInterior).branch(Opcode.IFEQ, scalarCell);
        code.lload(ow).loadConstant(lanes).ladd().lload(outputWidth).lcmp().branch(Opcode.IFGT, scalarCell);
        code.lload(ordinal).loadConstant(lanes).ladd().lload(end).lcmp().branch(Opcode.IFGT, scalarCell);
        code.lload(ow).loadConstant(paddingWidth).lsub().loadConstant(0L).lcmp().branch(Opcode.IFLT, scalarCell);
        code.lload(ow).loadConstant(paddingWidth).lsub().loadConstant(lanes).ladd().lload(kernelWidth).ladd()
                .loadConstant(2L).lsub().lload(inputWidth).lcmp().branch(Opcode.IFGE, scalarCell);
        code.getstatic(vectorClass, "SPECIES_PREFERRED",
                ClassDesc.of("jdk.incubator.vector.VectorSpecies"));
        if (type == DataType.FLOAT64) code.dload(biasValue);
        else code.fload(biasValue);
        code.invokestatic(vectorClass, "broadcast", MethodTypeDesc.of(vectorClass,
                ClassDesc.of("jdk.incubator.vector.VectorSpecies"),
                type == DataType.FLOAT64 ? ConstantDescs.CD_double : ConstantDescs.CD_float))
                .astore(vectorAccumulator);
        emitWidthVectorGeneralAccumulation(code, carriers, specialization, type, vectorClass,
                inputWidthCursor, weightChannelBase, channelsPerGroup, kernelDepth, kernelHeight, kernelWidth,
                inputChannelStride, inputDepthStride, inputHeightStride, inputWidthStride,
                weightChannelStride, weightDepthStride, weightHeightStride, weightWidthStride,
                dilationDepth, dilationHeight, channel, kd, kh, kw, inputAddress, weightAddress,
                vectorAccumulator, vectorInput, vectorWeight);
        carriers.vectorStore(type, specialization.carrierPattern().get(outputBoundary), outputBoundary,
                outputCursor, vectorAccumulator, false);
        code.lload(ow).loadConstant(lanes).ladd().lstore(ow);
        code.lload(inputWidthCursor).loadConstant(lanes).lload(inputWidthStride).lmul()
                .ladd().lstore(inputWidthCursor);
        code.lload(outputCursor).loadConstant(lanes).lload(outputWidthStride).lmul()
                .ladd().lstore(outputCursor);
        code.lload(ordinal).loadConstant(lanes).ladd().lstore(ordinal)
                .branch(Opcode.GOTO, advanceWidth);
        code.labelBinding(scalarCell);
        emitWidthVectorGeneralScalar(code, carriers, specialization, type, inputBatchGroupBase,
                weightChannelBase, channelsPerGroup, kernelDepth, kernelHeight, kernelWidth, inputDepth, inputHeight,
                inputWidth, inputChannelStride, inputDepthStride, inputHeightStride, inputWidthStride,
                weightChannelStride, weightDepthStride, weightHeightStride, weightWidthStride, inputDepthOrigin,
                inputHeightOrigin, ow, paddingWidth, dilationDepth, dilationHeight, channel, kd, kh, kw, id, ih, iw,
                inputAddress, weightAddress, biasValue, inputValue, weightValue, scalarAccumulator);
        carriers.storeFrozen(type, specialization.carrierPattern().get(outputBoundary), outputBoundary,
                outputCursor, scalarAccumulator, false);
        code.lload(ow).loadConstant(1L).ladd().lstore(ow);
        code.lload(inputWidthCursor).lload(inputWidthStride).ladd()
                .lstore(inputWidthCursor);
        code.lload(outputCursor).lload(outputWidthStride).ladd().lstore(outputCursor);
        code.lload(ordinal).loadConstant(1L).ladd().lstore(ordinal);
        code.labelBinding(advanceWidth).lload(ow).lload(outputWidth).lcmp()
                .branch(Opcode.IFLT, widths);
        code.loadConstant(0L).lstore(ow);
        code.lload(oh).loadConstant(1L).ladd().lstore(oh);
        code.lload(oh).lload(outputHeight).lcmp().branch(Opcode.IFLT, rows);
        code.loadConstant(0L).lstore(oh);
        code.lload(od).loadConstant(1L).ladd().lstore(od);
        code.lload(od).lload(outputDepth).lcmp().branch(Opcode.IFLT, depths);
        code.loadConstant(0L).lstore(od);
        code.lload(oc).loadConstant(1L).ladd().lstore(oc);
        code.lload(oc).lload(outputChannels).lcmp().branch(Opcode.IFLT, outputChannelLoop);
        code.loadConstant(0L).lstore(oc);
        code.lload(n).loadConstant(1L).ladd().lstore(n).branch(Opcode.GOTO, batches);
        code.labelBinding(done);
    }

    private static void emitWidthVectorGeneralAccumulation(CodeBuilder code,
            CpuCarrierEmitter carriers, CpuKernelSpecialization specialization, DataType type,
            ClassDesc vectorClass, int inputWidthCursor, int weightChannelBase, int channels,
            int kernelDepth, int kernelHeight, int kernelWidth, int inputChannelStride,
            int inputDepthStride, int inputHeightStride, int inputWidthStride,
            int weightChannelStride, int weightDepthStride, int weightHeightStride,
            int weightWidthStride, long dilationDepth, long dilationHeight, int channel, int kd,
            int kh, int kw, int inputAddress, int weightAddress, int accumulator, int input,
            int weight) {
        code.loadConstant(0L).lstore(channel);
        Label channelsLoop = code.newLabel();
        Label channelsDone = code.newLabel();
        Label depths = code.newLabel();
        Label depthsDone = code.newLabel();
        Label heights = code.newLabel();
        Label heightsDone = code.newLabel();
        Label widths = code.newLabel();
        Label widthsDone = code.newLabel();
        code.labelBinding(channelsLoop).lload(channel).lload(channels).lcmp()
                .branch(Opcode.IFGE, channelsDone);
        code.loadConstant(0L).lstore(kd);
        code.labelBinding(depths).lload(kd).lload(kernelDepth).lcmp()
                .branch(Opcode.IFGE, depthsDone);
        code.loadConstant(0L).lstore(kh);
        code.labelBinding(heights).lload(kh).lload(kernelHeight).lcmp()
                .branch(Opcode.IFGE, heightsDone);
        code.lload(inputWidthCursor).lload(channel).lload(inputChannelStride).lmul().ladd()
                .lload(kd).loadConstant(dilationDepth).lmul().lload(inputDepthStride).lmul()
                .ladd().lload(kh).loadConstant(dilationHeight).lmul()
                .lload(inputHeightStride).lmul().ladd().lstore(inputAddress);
        code.lload(weightChannelBase).lload(channel).lload(weightChannelStride).lmul().ladd()
                .lload(kd).lload(weightDepthStride).lmul().ladd()
                .lload(kh).lload(weightHeightStride).lmul().ladd().lstore(weightAddress);
        code.loadConstant(0L).lstore(kw);
        code.labelBinding(widths).lload(kw).lload(kernelWidth).lcmp()
                .branch(Opcode.IFGE, widthsDone);
        carriers.vectorLoad(type, specialization.carrierPattern().get(0), 0, inputAddress,
                false, false);
        code.astore(input);
        carriers.vectorLoad(type, specialization.carrierPattern().get(1), 1, weightAddress,
                true, false);
        code.astore(weight);
        code.aload(input).aload(weight).invokevirtual(vectorClass, "mul",
                MethodTypeDesc.of(vectorClass, ClassDesc.of("jdk.incubator.vector.Vector")))
                .astore(input);
        code.aload(accumulator).aload(input).invokevirtual(vectorClass, "add",
                MethodTypeDesc.of(vectorClass, ClassDesc.of("jdk.incubator.vector.Vector")))
                .astore(accumulator);
        code.lload(inputAddress).lload(inputWidthStride).ladd().lstore(inputAddress);
        code.lload(weightAddress).lload(weightWidthStride).ladd().lstore(weightAddress);
        code.lload(kw).loadConstant(1L).ladd().lstore(kw)
                .branch(Opcode.GOTO, widths);
        code.labelBinding(widthsDone);
        code.lload(kh).loadConstant(1L).ladd().lstore(kh)
                .branch(Opcode.GOTO, heights);
        code.labelBinding(heightsDone);
        code.lload(kd).loadConstant(1L).ladd().lstore(kd)
                .branch(Opcode.GOTO, depths);
        code.labelBinding(depthsDone);
        code.lload(channel).loadConstant(1L).ladd().lstore(channel)
                .branch(Opcode.GOTO, channelsLoop);
        code.labelBinding(channelsDone);
    }

    private static void emitWidthVectorGeneralScalar(CodeBuilder code,
            CpuCarrierEmitter carriers, CpuKernelSpecialization specialization, DataType type,
            int inputBase, int weightBase, int channels, int kernelDepth, int kernelHeight,
            int kernelWidth, int inputDepth, int inputHeight, int inputWidth,
            int inputChannelStride, int inputDepthStride, int inputHeightStride,
            int inputWidthStride, int weightChannelStride, int weightDepthStride,
            int weightHeightStride, int weightWidthStride, int depthOrigin, int heightOrigin,
            int ow, long paddingWidth, long dilationDepth, long dilationHeight, int channel,
            int kd, int kh, int kw, int id, int ih, int iw, int inputAddress,
            int weightAddress, int bias, int input, int weight, int accumulator) {
        if (type == DataType.FLOAT64) code.dload(bias).dstore(accumulator);
        else code.fload(bias).fstore(accumulator);
        code.loadConstant(0L).lstore(channel);
        Label channelsLoop = code.newLabel();
        Label channelsDone = code.newLabel();
        Label depths = code.newLabel();
        Label depthsDone = code.newLabel();
        Label heights = code.newLabel();
        Label heightsDone = code.newLabel();
        Label widths = code.newLabel();
        Label widthsDone = code.newLabel();
        Label padding = code.newLabel();
        Label inputReady = code.newLabel();
        code.labelBinding(channelsLoop).lload(channel).lload(channels).lcmp()
                .branch(Opcode.IFGE, channelsDone);
        code.loadConstant(0L).lstore(kd);
        code.labelBinding(depths).lload(kd).lload(kernelDepth).lcmp()
                .branch(Opcode.IFGE, depthsDone);
        code.lload(depthOrigin).lload(kd).loadConstant(dilationDepth).lmul().ladd()
                .lstore(id);
        code.loadConstant(0L).lstore(kh);
        code.labelBinding(heights).lload(kh).lload(kernelHeight).lcmp()
                .branch(Opcode.IFGE, heightsDone);
        code.lload(heightOrigin).lload(kh).loadConstant(dilationHeight).lmul().ladd()
                .lstore(ih);
        code.lload(weightBase).lload(channel).lload(weightChannelStride).lmul().ladd()
                .lload(kd).lload(weightDepthStride).lmul().ladd()
                .lload(kh).lload(weightHeightStride).lmul().ladd().lstore(weightAddress);
        code.loadConstant(0L).lstore(kw);
        code.labelBinding(widths).lload(kw).lload(kernelWidth).lcmp()
                .branch(Opcode.IFGE, widthsDone);
        code.lload(ow).loadConstant(paddingWidth).lsub().lload(kw).ladd().lstore(iw);
        code.lload(id).loadConstant(0L).lcmp().branch(Opcode.IFLT, padding);
        code.lload(id).lload(inputDepth).lcmp().branch(Opcode.IFGE, padding);
        code.lload(ih).loadConstant(0L).lcmp().branch(Opcode.IFLT, padding);
        code.lload(ih).lload(inputHeight).lcmp().branch(Opcode.IFGE, padding);
        code.lload(iw).loadConstant(0L).lcmp().branch(Opcode.IFLT, padding);
        code.lload(iw).lload(inputWidth).lcmp().branch(Opcode.IFGE, padding);
        code.lload(inputBase).lload(channel).lload(inputChannelStride).lmul().ladd()
                .lload(id).lload(inputDepthStride).lmul().ladd()
                .lload(ih).lload(inputHeightStride).lmul().ladd()
                .lload(iw).lload(inputWidthStride).lmul().ladd().lstore(inputAddress);
        carriers.loadFrozen(type, specialization.carrierPattern().get(0), 0, inputAddress, false);
        if (type == DataType.FLOAT64) code.dstore(input);
        else code.fstore(input);
        code.branch(Opcode.GOTO, inputReady);
        code.labelBinding(padding);
        if (type == DataType.FLOAT64) code.loadConstant(0d).dstore(input);
        else code.loadConstant(0f).fstore(input);
        code.labelBinding(inputReady);
        carriers.loadFrozen(type, specialization.carrierPattern().get(1), 1,
                weightAddress, false);
        if (type == DataType.FLOAT64) {
            code.dstore(weight).dload(accumulator).dload(input).dload(weight)
                    .dmul().dadd().dstore(accumulator);
        } else {
            code.fstore(weight).fload(accumulator).fload(input).fload(weight)
                    .fmul().fadd().fstore(accumulator);
        }
        code.lload(weightAddress).lload(weightWidthStride).ladd().lstore(weightAddress);
        code.lload(kw).loadConstant(1L).ladd().lstore(kw).branch(Opcode.GOTO, widths);
        code.labelBinding(widthsDone);
        code.lload(kh).loadConstant(1L).ladd().lstore(kh).branch(Opcode.GOTO, heights);
        code.labelBinding(heightsDone);
        code.lload(kd).loadConstant(1L).ladd().lstore(kd).branch(Opcode.GOTO, depths);
        code.labelBinding(depthsDone);
        code.lload(channel).loadConstant(1L).ladd().lstore(channel)
                .branch(Opcode.GOTO, channelsLoop);
        code.labelBinding(channelsDone);
    }

    private static int longGeometry(CodeBuilder code, int geometry, int index) {
        int local = code.allocateLocal(TypeKind.LONG);
        CpuNormEmitter.geometry(code, geometry, index).lstore(local);
        return local;
    }

    private static void decodeLong(CodeBuilder code, int remaining, int extent, int coordinate) {
        code.lload(remaining).lload(extent).lrem().lstore(coordinate)
                .lload(remaining).lload(extent).ldiv().lstore(remaining);
    }

    private static int representedLocal(CodeBuilder code, DataType type) {
        return code.allocateLocal(type == DataType.FLOAT64 ? TypeKind.DOUBLE
                : type == DataType.FLOAT32 ? TypeKind.FLOAT : TypeKind.INT);
    }

    private static int intGeometry(CodeBuilder code, int geometry, int index) {
        int local = code.allocateLocal(TypeKind.INT);
        CpuNormEmitter.geometry(code, geometry, index).l2i().istore(local);
        return local;
    }

    private static void decodeInt(CodeBuilder code, int remaining, int extent, int coordinate) {
        code.lload(remaining).iload(extent).i2l().lrem().l2i().istore(coordinate)
                .lload(remaining).iload(extent).i2l().ldiv().lstore(remaining);
    }

    private static void emitVectorAccumulation(CodeBuilder code, CpuCarrierEmitter carriers,
            DataType type, CpuKernelSpecialization.CarrierAccess array, ClassDesc vector,
            ClassDesc vectorBase, int cpg, int kD, int kH, int kW, int inD, int inH, int inW,
            int dilationD, int dilationH, int inputCursor, int weightBase, int channel, int kd,
            int kh, int kw, int inputAddress, int weightAddress, int acc, int input, int weight) {
        code.loadConstant(0).istore(channel);
        Label cs = code.newLabel();
        Label ce = code.newLabel();
        Label ds = code.newLabel();
        Label de = code.newLabel();
        Label hs = code.newLabel();
        Label he = code.newLabel();
        Label ws = code.newLabel();
        Label we = code.newLabel();
        code.labelBinding(cs).iload(channel).iload(cpg).branch(Opcode.IF_ICMPGE, ce)
                .loadConstant(0).istore(kd).labelBinding(ds).iload(kd).iload(kD)
                .branch(Opcode.IF_ICMPGE, de).loadConstant(0).istore(kh).labelBinding(hs)
                .iload(kh).iload(kH).branch(Opcode.IF_ICMPGE, he);
        code.iload(inputCursor).iload(channel).iload(inD).imul().iload(inH).imul()
                .iload(inW).imul().iadd().iload(kd).loadConstant(dilationD).imul()
                .iload(inH).imul().iload(inW).imul().iadd().iload(kh).loadConstant(dilationH)
                .imul().iload(inW).imul().iadd().istore(inputAddress);
        code.iload(weightBase).iload(channel).iload(kD).imul().iload(kH).imul().iload(kW)
                .imul().iadd().iload(kd).iload(kH).imul().iload(kW).imul().iadd().iload(kh)
                .iload(kW).imul().iadd().istore(weightAddress);
        code.loadConstant(0).istore(kw).labelBinding(ws).iload(kw).iload(kW)
                .branch(Opcode.IF_ICMPGE, we);
        carriers.vectorLoad(type, array, 0, inputAddress, false, true);
        code.astore(input);
        carriers.vectorLoad(type, array, 1, weightAddress, true, true);
        code.astore(weight);
        code.aload(input).aload(weight).invokevirtual(vector, "mul", MethodTypeDesc.of(vector, vectorBase))
                .astore(input);
        code.aload(acc).aload(input).invokevirtual(vector, "add", MethodTypeDesc.of(vector, vectorBase))
                .astore(acc);
        code.iinc(inputAddress, 1).iinc(weightAddress, 1).iinc(kw, 1).branch(Opcode.GOTO, ws)
                .labelBinding(we);
        code.iinc(kh, 1).branch(Opcode.GOTO, hs).labelBinding(he);
        code.iinc(kd, 1).branch(Opcode.GOTO, ds).labelBinding(de);
        code.iinc(channel, 1).branch(Opcode.GOTO, cs).labelBinding(ce);
    }
    private static void emitScalarAccumulation(CodeBuilder code, DataType type, int inputBase,
            int weightBase, int cpg, int kD, int kH, int kW, int inD, int inH, int inW,
            int depthOrigin, int heightOrigin, int ow, int pW, int dD, int dH, int channel,
            int kd, int kh, int kw, int id, int ih, int iw, int inputAddress, int weightAddress,
            int bias, int acc) {
        if (type == DataType.FLOAT64) {
            code.dload(bias).dstore(acc);
        } else {
            code.fload(bias).fstore(acc);
        }
        code.loadConstant(0).istore(channel);
        Label cs = code.newLabel();
        Label ce = code.newLabel();
        Label ds = code.newLabel();
        Label de = code.newLabel();
        Label hs = code.newLabel();
        Label he = code.newLabel();
        Label ws = code.newLabel();
        Label we = code.newLabel();
        Label pad = code.newLabel();
        Label ready = code.newLabel();
        code.labelBinding(cs).iload(channel).iload(cpg).branch(Opcode.IF_ICMPGE, ce)
                .loadConstant(0).istore(kd).labelBinding(ds).iload(kd).iload(kD)
                .branch(Opcode.IF_ICMPGE, de);
        code.iload(depthOrigin).iload(kd).loadConstant(dD).imul().iadd().istore(id)
                .loadConstant(0).istore(kh).labelBinding(hs).iload(kh).iload(kH)
                .branch(Opcode.IF_ICMPGE, he);
        code.iload(heightOrigin).iload(kh).loadConstant(dH).imul().iadd().istore(ih);
        code.iload(weightBase).iload(channel).iload(kD).imul().iload(kH).imul().iload(kW)
                .imul().iadd().iload(kd).iload(kH).imul().iload(kW).imul().iadd().iload(kh)
                .iload(kW).imul().iadd().istore(weightAddress);
        code.loadConstant(0).istore(kw).labelBinding(ws).iload(kw).iload(kW)
                .branch(Opcode.IF_ICMPGE, we);
        code.iload(ow).loadConstant(pW).isub().iload(kw).iadd().istore(iw);
        code.iload(id).branch(Opcode.IFLT, pad).iload(id).iload(inD)
                .branch(Opcode.IF_ICMPGE, pad).iload(ih).branch(Opcode.IFLT, pad).iload(ih)
                .iload(inH).branch(Opcode.IF_ICMPGE, pad).iload(iw).branch(Opcode.IFLT, pad)
                .iload(iw).iload(inW).branch(Opcode.IF_ICMPGE, pad);
        code.iload(inputBase).iload(channel).iload(inD).imul().iload(inH).imul().iload(inW)
                .imul().iadd().iload(id).iload(inH).imul().iload(inW).imul().iadd().iload(ih)
                .iload(inW).imul().iadd().iload(iw).iadd().istore(inputAddress);
        if (type == DataType.FLOAT64) {
            code.dload(acc).aload(0).iload(inputAddress).daload();
        } else {
            code.fload(acc).aload(0).iload(inputAddress).faload();
        }
        code.branch(Opcode.GOTO, ready).labelBinding(pad);
        if (type == DataType.FLOAT64) {
            code.dload(acc).loadConstant(0d);
        } else {
            code.fload(acc).loadConstant(0f);
        }
        code.labelBinding(ready).aload(1).iload(weightAddress);
        if (type == DataType.FLOAT64) {
            code.daload().dmul().dadd().dstore(acc);
        } else {
            code.faload().fmul().fadd().fstore(acc);
        }
        code.iinc(weightAddress, 1).iinc(kw, 1).branch(Opcode.GOTO, ws).labelBinding(we);
        code.iinc(kh, 1).branch(Opcode.GOTO, hs).labelBinding(he);
        code.iinc(kd, 1).branch(Opcode.GOTO, ds).labelBinding(de);
        code.iinc(channel, 1).branch(Opcode.GOTO, cs).labelBinding(ce);
    }

    private static void load(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, DataType resultType,
            int boundary, int address,
            int represented, int decoded) {
        load(code,carriers,specialization,type,resultType,boundary,address,represented,decoded,false);
    }

    private static void load(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization specialization, DataType type, DataType resultType,
            int boundary, int address, int represented, int decoded,boolean intAddress) {
        carriers.loadFrozen(type, specialization.carrierPattern().get(boundary), boundary,
                address, intAddress);
        if (type == resultType && type == DataType.FLOAT64) {
            code.dstore(decoded);
            return;
        }
        if (type == resultType && type == DataType.FLOAT32) {
            code.fstore(decoded);
            return;
        }
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

    private static void decodeCoordinate(CodeBuilder code, int extent,
            int remaining, int coordinate) {
        code.lload(remaining).lload(extent).lrem()
                .lstore(coordinate);
        code.lload(remaining).lload(extent).ldiv()
                .lstore(remaining);
    }

    private static void weightAddress(CodeBuilder code,int base,int so,int sc,int sd,int sh,int sw,int channel,
            int kd, int kh, int kw, int oc, int target) {
        code.lload(base).lload(oc).lload(so).lmul().ladd()
                .lload(channel).lload(sc).lmul().ladd()
                .lload(kd).lload(sd).lmul().ladd()
                .lload(kh).lload(sh).lmul().ladd()
                .lload(kw).lload(sw).lmul().ladd()
                .lstore(target);
    }

    private static void inputAddress(CodeBuilder code,int base,int sn,int sc,int sd,int sh,int sw,int n,
            int group, int channel, int id, int ih, int iw, int target, int channelsPerGroup) {
        code.lload(base).lload(n).lload(sn).lmul().ladd()
                .lload(group).lload(channelsPerGroup)
                .lmul().lload(channel).ladd();
        code.lload(sc).lmul().ladd()
                .lload(id).lload(sd).lmul().ladd()
                .lload(ih).lload(sh).lmul().ladd()
                .lload(iw).lload(sw).lmul().ladd()
                .lstore(target);
    }

    private static void address(CodeBuilder code,int base,int sn,int sc,int sd,int sh,int sw,
            int n, int c, int d, int h, int w, int target) {
        code.lload(base).lload(n).lload(sn).lmul().ladd().lload(c).lload(sc).lmul().ladd()
                .lload(d).lload(sd).lmul().ladd().lload(h).lload(sh).lmul().ladd()
                .lload(w).lload(sw).lmul().ladd().lstore(target);
    }

    private static void weightAddressDense(CodeBuilder code, int base, int channels, int depth,
            int height, int width, int channel, int kd, int kh, int kw, int oc, int target) {
        code.lload(base).l2i().lload(oc).l2i().lload(channels).l2i().imul().lload(channel)
                .l2i().iadd().lload(depth).l2i().imul().lload(kd).l2i().iadd().lload(height)
                .l2i().imul().lload(kh).l2i().iadd().lload(width).l2i().imul().lload(kw)
                .l2i().iadd().iadd().istore(target);
    }

    private static void inputAddressDense(CodeBuilder code, int base, int channelsPerGroup,
            int depth, int height, int width, long groups, int n, int group, int channel, int id,
            int ih, int iw, int target) {
        code.lload(base).l2i().lload(n).l2i().lload(channelsPerGroup).l2i()
                .loadConstant(Math.toIntExact(groups)).imul().imul().lload(group).l2i()
                .lload(channelsPerGroup).l2i().imul().lload(channel).l2i().iadd().iadd()
                .lload(depth).l2i().imul().lload(id).l2i().iadd().lload(height).l2i().imul()
                .lload(ih).l2i().iadd().lload(width).l2i().imul().lload(iw).l2i().iadd()
                .iadd().istore(target);
    }

    private static void outputAddressDense(CodeBuilder code, int base, int channels, int depth,
            int height, int width, int n, int c, int d, int h, int w, int target) {
        code.lload(base).l2i().lload(n).l2i().lload(channels).l2i().imul().lload(c).l2i()
                .iadd().lload(depth).l2i().imul().lload(d).l2i().iadd().lload(height).l2i()
                .imul().lload(h).l2i().iadd().lload(width).l2i().imul().lload(w).l2i().iadd()
                .iadd().istore(target);
    }

    private static int geometryLocal(CodeBuilder code, int geometry, int index) {
        int local = code.allocateLocal(TypeKind.LONG);
        CpuNormEmitter.geometry(code, geometry, index).lstore(local);
        return local;
    }
}
