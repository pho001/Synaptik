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

/**
 * Emits direct typed grouped NCHW Conv2d scalar or output-width-vector loops.
 *
 * <p>The generated body traverses input channels within the selected group, kernel height, and
 * kernel width in increasing logical order. It loads every weight contribution, represents an
 * out-of-range input coordinate as positive zero before ordinary multiplication, accumulates in
 * FLOAT64 or FLOAT32 as selected by the output type, and narrows BFLOAT16 only at the final store.
 * No Synaptik method is called by generated hot work. The schema-63 vector form is limited to
 * same-typed dense FLOAT32 or FLOAT64 direct convolution without an external epilogue. Only the
 * width stride and width dilation must be one; non-width stride and dilation remain eligible.
 * Preferred-species chunks cover complete in-bounds output-width cells, while padding borders,
 * worker-range fragments, and tails use the same ordered scalar-cell semantics. Every other form,
 * including BFLOAT16, short or interior-free widths, and non-dense access, remains scalar.</p>
 */
public final class CpuConv2dEmitter {
    private static final ClassDesc MATH = ClassDesc.of(Math.class.getName());
    private static final ClassDesc VECTOR = ClassDesc.of("jdk.incubator.vector.Vector");
    private static final ClassDesc VECTOR_SPECIES =
            ClassDesc.of("jdk.incubator.vector.VectorSpecies");
    private static final ClassDesc FLOAT_VECTOR =
            ClassDesc.of("jdk.incubator.vector.FloatVector");
    private static final ClassDesc DOUBLE_VECTOR =
            ClassDesc.of("jdk.incubator.vector.DoubleVector");
    /** Creates a stateless direct-convolution emitter. */
    public CpuConv2dEmitter() { }

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
        boolean widthVector = specialization.classIdentitySchema() == 63
                && specialization.executionStrategy().compute()
                    == io.github.pho001.synaptik.backend.cpu.internal.prepare
                        .CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR;
        if (!identity.startsWith("conv2d:") || boundaries != (bias ? 4 : 3) + (add ? 1 : 0)
                || !widthVector && specialization.executionStrategy().compute()
                    != io.github.pho001.synaptik.backend.cpu.internal.prepare
                        .CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR) {
            throw new IllegalArgumentException("Conv2d generated facts disagree");
        }
        if (widthVector) {
            emitWidthVector(code, specialization, ir, bias, add);
            return;
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

    /**
     * Emits the schema-63 dense-array nested width-vector realization.
     *
     * <p>It decodes the first coordinate once, carries enclosing coordinates in nested loops,
     * computes group state only when the output channel changes, and increments the current
     * input/output width cursors after every scalar cell or vector block. Width stride and
     * dilation are one; height stride and dilation remain arbitrary positive admitted values.</p>
     */
    private static void emitWidthVector(CodeBuilder code,
            CpuKernelSpecialization specialization, CpuKernelIr ir, boolean bias, boolean add) {
        DataType type = specialization.boundaryDataTypes().getFirst();
        CpuKernelSpecialization.CarrierAccess array = type == DataType.FLOAT32
                ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                : CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
        if (add || type != specialization.boundaryDataTypes().getLast()
                || type != DataType.FLOAT32 && type != DataType.FLOAT64
                || specialization.boundaryDataTypes().stream().anyMatch(candidate -> candidate != type)
                || specialization.carrierPattern().stream().anyMatch(access -> access != array
                        && access != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                || CpuNormEmitter.longAfter(ir.familyIdentity(), ":strideW=") != 1L
                || CpuNormEmitter.longAfter(ir.familyIdentity(), ":dilationW=") != 1L) {
            throw new IllegalArgumentException(
                    "Conv2d width vector requires same-typed dense carriers and unit width stride/dilation");
        }
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
        int inputStrides = inputExtents + 4;
        int weightExtents = inputStrides + 4;
        int weightStrides = weightExtents + 4;
        int outputExtents = weightStrides + 4 + (bias ? 2 : 0);
        int inputBase = intGeometry(code, geometry, 0);
        int weightBase = intGeometry(code, geometry, 1);
        int biasBase = bias ? intGeometry(code, geometry, 2) : -1;
        int outputBoundary = boundaries - 1;
        int outputBase = intGeometry(code, geometry, outputBoundary);
        int inputChannels = intGeometry(code, geometry, inputExtents + 1);
        int inputHeight = intGeometry(code, geometry, inputExtents + 2);
        int inputWidth = intGeometry(code, geometry, inputExtents + 3);
        int channelsPerGroup = intGeometry(code, geometry, weightExtents + 1);
        int kernelHeight = intGeometry(code, geometry, weightExtents + 2);
        int kernelWidth = intGeometry(code, geometry, weightExtents + 3);
        int outputChannels = intGeometry(code, geometry, outputExtents + 1);
        int outputHeight = intGeometry(code, geometry, outputExtents + 2);
        int outputWidth = intGeometry(code, geometry, outputExtents + 3);
        int ordinal = code.allocateLocal(TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.LONG);
        int n = code.allocateLocal(TypeKind.INT);
        int oc = code.allocateLocal(TypeKind.INT);
        int oh = code.allocateLocal(TypeKind.INT);
        int ow = code.allocateLocal(TypeKind.INT);
        int group = code.allocateLocal(TypeKind.INT);
        int inputBatchGroupBase = code.allocateLocal(TypeKind.INT);
        int weightChannelBase = code.allocateLocal(TypeKind.INT);
        int outputChannelBase = code.allocateLocal(TypeKind.INT);
        int inputHeightOrigin = code.allocateLocal(TypeKind.INT);
        int inputWidthCursor = code.allocateLocal(TypeKind.INT);
        int outputCursor = code.allocateLocal(TypeKind.INT);
        int rowInterior = code.allocateLocal(TypeKind.INT);
        int channel = code.allocateLocal(TypeKind.INT);
        int kh = code.allocateLocal(TypeKind.INT);
        int kw = code.allocateLocal(TypeKind.INT);
        int ih = code.allocateLocal(TypeKind.INT);
        int iw = code.allocateLocal(TypeKind.INT);
        int inputAddress = code.allocateLocal(TypeKind.INT);
        int weightAddress = code.allocateLocal(TypeKind.INT);
        TypeKind scalarKind = type == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT;
        int biasValue = code.allocateLocal(scalarKind);
        int scalarAccumulator = code.allocateLocal(scalarKind);
        int vectorAccumulator = code.allocateLocal(TypeKind.REFERENCE);
        int vectorInput = code.allocateLocal(TypeKind.REFERENCE);
        int vectorWeight = code.allocateLocal(TypeKind.REFERENCE);
        int lanes = specialization.vectorSpeciesBitSize() / type.bitWidth();
        int groups = Math.toIntExact(CpuNormEmitter.longAfter(ir.familyIdentity(), ":groups="));
        int strideHeight = Math.toIntExact(CpuNormEmitter.longAfter(
                ir.familyIdentity(), ":strideH="));
        int paddingHeight = Math.toIntExact(CpuNormEmitter.longAfter(
                ir.familyIdentity(), ":padH="));
        int paddingWidth = Math.toIntExact(CpuNormEmitter.longAfter(
                ir.familyIdentity(), ":padW="));
        int dilationHeight = Math.toIntExact(CpuNormEmitter.longAfter(
                ir.familyIdentity(), ":dilationH="));
        ClassDesc vectorClass = type == DataType.FLOAT64 ? DOUBLE_VECTOR : FLOAT_VECTOR;
        CpuCarrierEmitter carriers = new CpuCarrierEmitter(code);
        carriers.prepareVectorSpecies(type);

        Label done = code.newLabel();
        code.lload(start).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.lload(start).lstore(ordinal).lload(start).lstore(remaining);
        decodeInt(code, remaining, outputWidth, ow);
        decodeInt(code, remaining, outputHeight, oh);
        decodeInt(code, remaining, outputChannels, oc);
        code.lload(remaining).l2i().istore(n);

        Label batches = code.newLabel();
        Label channels = code.newLabel();
        Label rows = code.newLabel();
        Label widths = code.newLabel();
        Label scalarCell = code.newLabel();
        Label advanceWidth = code.newLabel();
        code.labelBinding(batches).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.labelBinding(channels).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        // Group division is deliberately outside the row and width loops.
        code.iload(oc).iload(outputChannels).loadConstant(groups).idiv().idiv().istore(group);
        code.iload(inputBase).iload(n).iload(inputChannels).imul()
                .iload(group).iload(channelsPerGroup).imul().iadd()
                .iload(inputHeight).imul().iload(inputWidth).imul().iadd()
                .istore(inputBatchGroupBase);
        code.iload(weightBase).iload(oc).iload(channelsPerGroup).imul()
                .iload(kernelHeight).imul().iload(kernelWidth).imul().iadd()
                .istore(weightChannelBase);
        code.iload(outputBase).iload(n).iload(outputChannels).imul().iload(oc).iadd()
                .iload(outputHeight).imul().iload(outputWidth).imul().iadd()
                .istore(outputChannelBase);
        if (bias) {
            code.aload(2).iload(biasBase).iload(oc).iadd();
            if (type == DataType.FLOAT64) code.daload().dstore(biasValue);
            else code.faload().fstore(biasValue);
        } else if (type == DataType.FLOAT64) code.loadConstant(0.0d).dstore(biasValue);
        else code.loadConstant(0.0f).fstore(biasValue);

        code.labelBinding(rows).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.iload(oh).loadConstant(strideHeight).imul().loadConstant(paddingHeight).isub()
                .istore(inputHeightOrigin);
        code.iload(inputBatchGroupBase).iload(inputHeightOrigin).iload(inputWidth).imul().iadd()
                .loadConstant(paddingWidth).isub().iload(ow).iadd().istore(inputWidthCursor);
        code.iload(outputChannelBase).iload(oh).iload(outputWidth).imul().iadd()
                .iload(ow).iadd().istore(outputCursor);
        Label rowNotInterior = code.newLabel();
        Label rowInteriorReady = code.newLabel();
        code.iload(inputHeightOrigin).branch(Opcode.IFLT, rowNotInterior);
        code.iload(inputHeightOrigin).iload(kernelHeight).loadConstant(1).isub()
                .loadConstant(dilationHeight).imul().iadd().iload(inputHeight)
                .branch(Opcode.IF_ICMPGE, rowNotInterior);
        code.loadConstant(1).istore(rowInterior).branch(Opcode.GOTO, rowInteriorReady)
                .labelBinding(rowNotInterior).loadConstant(0).istore(rowInterior)
                .labelBinding(rowInteriorReady);

        code.labelBinding(widths).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.iload(rowInterior).branch(Opcode.IFEQ, scalarCell);
        code.iload(ow).loadConstant(lanes).iadd().iload(outputWidth)
                .branch(Opcode.IF_ICMPGT, scalarCell);
        code.lload(ordinal).loadConstant((long) lanes).ladd().lload(end).lcmp()
                .branch(Opcode.IFGT, scalarCell);
        code.iload(ow).loadConstant(paddingWidth).isub().branch(Opcode.IFLT, scalarCell);
        code.iload(ow).loadConstant(paddingWidth).isub().loadConstant(lanes).iadd()
                .iload(kernelWidth).iadd().loadConstant(2).isub().iload(inputWidth)
                .branch(Opcode.IF_ICMPGE, scalarCell);

        code.getstatic(vectorClass, "SPECIES_PREFERRED", VECTOR_SPECIES);
        if (type == DataType.FLOAT64) code.dload(biasValue);
        else code.fload(biasValue);
        code.invokestatic(vectorClass, "broadcast", MethodTypeDesc.of(vectorClass,
                VECTOR_SPECIES, type == DataType.FLOAT64
                        ? ConstantDescs.CD_double : ConstantDescs.CD_float))
                .astore(vectorAccumulator);
        code.loadConstant(0).istore(channel);
        Label vectorChannels = code.newLabel();
        Label vectorChannelsDone = code.newLabel();
        code.labelBinding(vectorChannels).iload(channel).iload(channelsPerGroup)
                .branch(Opcode.IF_ICMPGE, vectorChannelsDone);
        code.loadConstant(0).istore(kh);
        Label vectorHeights = code.newLabel();
        Label vectorHeightsDone = code.newLabel();
        code.labelBinding(vectorHeights).iload(kh).iload(kernelHeight)
                .branch(Opcode.IF_ICMPGE, vectorHeightsDone);
        code.iload(inputWidthCursor).iload(channel).iload(inputHeight).imul()
                .iload(inputWidth).imul().iadd().iload(kh).loadConstant(dilationHeight).imul()
                .iload(inputWidth).imul().iadd().istore(inputAddress);
        code.iload(weightChannelBase).iload(channel).iload(kernelHeight).imul()
                .iload(kh).iadd().iload(kernelWidth).imul().iadd().istore(weightAddress);
        code.loadConstant(0).istore(kw);
        Label vectorKernelWidths = code.newLabel();
        Label vectorKernelWidthsDone = code.newLabel();
        code.labelBinding(vectorKernelWidths).iload(kw).iload(kernelWidth)
                .branch(Opcode.IF_ICMPGE, vectorKernelWidthsDone);
        carriers.vectorLoad(type, array, 0, inputAddress, false, true);
        code.astore(vectorInput);
        carriers.vectorLoad(type, array, 1, weightAddress, true, true);
        code.astore(vectorWeight);
        code.aload(vectorInput).aload(vectorWeight).invokevirtual(vectorClass, "mul",
                MethodTypeDesc.of(vectorClass, VECTOR)).astore(vectorInput);
        code.aload(vectorAccumulator).aload(vectorInput).invokevirtual(vectorClass, "add",
                MethodTypeDesc.of(vectorClass, VECTOR)).astore(vectorAccumulator);
        code.iinc(inputAddress, 1).iinc(weightAddress, 1).iinc(kw, 1)
                .branch(Opcode.GOTO, vectorKernelWidths).labelBinding(vectorKernelWidthsDone);
        code.iinc(kh, 1).branch(Opcode.GOTO, vectorHeights)
                .labelBinding(vectorHeightsDone);
        code.iinc(channel, 1).branch(Opcode.GOTO, vectorChannels)
                .labelBinding(vectorChannelsDone);
        carriers.vectorStore(type, array, outputBoundary, outputCursor,
                vectorAccumulator, true);
        code.iinc(ow, lanes).iinc(inputWidthCursor, lanes).iinc(outputCursor, lanes)
                .lload(ordinal).loadConstant((long) lanes).ladd().lstore(ordinal)
                .branch(Opcode.GOTO, advanceWidth);

        code.labelBinding(scalarCell);
        if (type == DataType.FLOAT64) code.dload(biasValue).dstore(scalarAccumulator);
        else code.fload(biasValue).fstore(scalarAccumulator);
        code.loadConstant(0).istore(channel);
        Label scalarChannels = code.newLabel();
        Label scalarChannelsDone = code.newLabel();
        code.labelBinding(scalarChannels).iload(channel).iload(channelsPerGroup)
                .branch(Opcode.IF_ICMPGE, scalarChannelsDone);
        code.loadConstant(0).istore(kh);
        Label scalarHeights = code.newLabel();
        Label scalarHeightsDone = code.newLabel();
        code.labelBinding(scalarHeights).iload(kh).iload(kernelHeight)
                .branch(Opcode.IF_ICMPGE, scalarHeightsDone);
        code.iload(inputHeightOrigin).iload(kh).loadConstant(dilationHeight).imul().iadd()
                .istore(ih);
        code.iload(weightChannelBase).iload(channel).iload(kernelHeight).imul()
                .iload(kh).iadd().iload(kernelWidth).imul().iadd().istore(weightAddress);
        code.loadConstant(0).istore(kw);
        Label scalarKernelWidths = code.newLabel();
        Label scalarKernelWidthsDone = code.newLabel();
        code.labelBinding(scalarKernelWidths).iload(kw).iload(kernelWidth)
                .branch(Opcode.IF_ICMPGE, scalarKernelWidthsDone);
        code.iload(ow).loadConstant(paddingWidth).isub().iload(kw).iadd().istore(iw);
        Label padded = code.newLabel();
        Label valueReady = code.newLabel();
        code.iload(ih).branch(Opcode.IFLT, padded).iload(ih).iload(inputHeight)
                .branch(Opcode.IF_ICMPGE, padded).iload(iw).branch(Opcode.IFLT, padded)
                .iload(iw).iload(inputWidth).branch(Opcode.IF_ICMPGE, padded);
        code.iload(inputBatchGroupBase).iload(channel).iload(inputHeight).imul()
                .iload(inputWidth).imul().iadd().iload(ih).iload(inputWidth).imul().iadd()
                .iload(iw).iadd().istore(inputAddress);
        if (type == DataType.FLOAT64) {
            code.dload(scalarAccumulator).aload(0).iload(inputAddress).daload();
        } else {
            code.fload(scalarAccumulator).aload(0).iload(inputAddress).faload();
        }
        code.branch(Opcode.GOTO, valueReady).labelBinding(padded);
        if (type == DataType.FLOAT64) code.dload(scalarAccumulator).loadConstant(0.0d);
        else code.fload(scalarAccumulator).loadConstant(0.0f);
        code.labelBinding(valueReady).aload(1).iload(weightAddress);
        if (type == DataType.FLOAT64) {
            code.daload().dmul().dadd().dstore(scalarAccumulator);
        } else {
            code.faload().fmul().fadd().fstore(scalarAccumulator);
        }
        code.iinc(weightAddress, 1).iinc(kw, 1).branch(Opcode.GOTO, scalarKernelWidths)
                .labelBinding(scalarKernelWidthsDone);
        code.iinc(kh, 1).branch(Opcode.GOTO, scalarHeights)
                .labelBinding(scalarHeightsDone);
        code.iinc(channel, 1).branch(Opcode.GOTO, scalarChannels)
                .labelBinding(scalarChannelsDone);
        code.aload(outputBoundary).iload(outputCursor);
        if (type == DataType.FLOAT64) code.dload(scalarAccumulator).dastore();
        else code.fload(scalarAccumulator).fastore();
        code.iinc(ow, 1).iinc(inputWidthCursor, 1).iinc(outputCursor, 1)
                .lload(ordinal).loadConstant(1L).ladd().lstore(ordinal);

        code.labelBinding(advanceWidth).iload(ow).iload(outputWidth)
                .branch(Opcode.IF_ICMPLT, widths);
        code.loadConstant(0).istore(ow).iinc(oh, 1);
        code.iload(oh).iload(outputHeight).branch(Opcode.IF_ICMPLT, rows);
        code.loadConstant(0).istore(oh).iinc(oc, 1);
        code.iload(oc).iload(outputChannels).branch(Opcode.IF_ICMPLT, channels);
        code.loadConstant(0).istore(oc).iinc(n, 1).branch(Opcode.GOTO, batches)
                .labelBinding(done);
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
        int inputStrides = inputExtents + 4;
        int weightExtents = inputStrides + 4;
        int weightStrides = weightExtents + 4;
        int outputExtents = weightStrides + 4 + (bias ? 2 : 0);
        int outputStrides = outputExtents + 4;
        int outputBoundary = boundaries - 1;
        int inputBase = longGeometry(code, geometry, 0);
        int weightBase = longGeometry(code, geometry, 1);
        int biasBase = bias ? longGeometry(code, geometry, 2) : -1;
        int outputBase = longGeometry(code, geometry, outputBoundary);
        int inputHeight = longGeometry(code, geometry, inputExtents + 2);
        int inputWidth = longGeometry(code, geometry, inputExtents + 3);
        int inputBatchStride = longGeometry(code, geometry, inputStrides);
        int inputChannelStride = longGeometry(code, geometry, inputStrides + 1);
        int inputHeightStride = longGeometry(code, geometry, inputStrides + 2);
        int inputWidthStride = longGeometry(code, geometry, inputStrides + 3);
        int channelsPerGroup = longGeometry(code, geometry, weightExtents + 1);
        int kernelHeight = longGeometry(code, geometry, weightExtents + 2);
        int kernelWidth = longGeometry(code, geometry, weightExtents + 3);
        int weightOutputStride = longGeometry(code, geometry, weightStrides);
        int weightChannelStride = longGeometry(code, geometry, weightStrides + 1);
        int weightHeightStride = longGeometry(code, geometry, weightStrides + 2);
        int weightWidthStride = longGeometry(code, geometry, weightStrides + 3);
        int biasStride = bias ? longGeometry(code, geometry, weightStrides + 5) : -1;
        int outputChannels = longGeometry(code, geometry, outputExtents + 1);
        int outputHeight = longGeometry(code, geometry, outputExtents + 2);
        int outputWidth = longGeometry(code, geometry, outputExtents + 3);
        int outputBatchStride = longGeometry(code, geometry, outputStrides);
        int outputChannelStride = longGeometry(code, geometry, outputStrides + 1);
        int outputHeightStride = longGeometry(code, geometry, outputStrides + 2);
        int outputWidthStride = longGeometry(code, geometry, outputStrides + 3);
        int ordinal = code.allocateLocal(TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.LONG);
        int n = code.allocateLocal(TypeKind.LONG);
        int oc = code.allocateLocal(TypeKind.LONG);
        int oh = code.allocateLocal(TypeKind.LONG);
        int ow = code.allocateLocal(TypeKind.LONG);
        int group = code.allocateLocal(TypeKind.LONG);
        int inputBatchGroupBase = code.allocateLocal(TypeKind.LONG);
        int weightChannelBase = code.allocateLocal(TypeKind.LONG);
        int outputChannelBase = code.allocateLocal(TypeKind.LONG);
        int inputHeightOrigin = code.allocateLocal(TypeKind.LONG);
        int inputWidthCursor = code.allocateLocal(TypeKind.LONG);
        int outputCursor = code.allocateLocal(TypeKind.LONG);
        int rowInterior = code.allocateLocal(TypeKind.INT);
        int channel = code.allocateLocal(TypeKind.LONG);
        int kh = code.allocateLocal(TypeKind.LONG);
        int kw = code.allocateLocal(TypeKind.LONG);
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
        long groups = CpuNormEmitter.longAfter(ir.familyIdentity(), ":groups=");
        long strideHeight = CpuNormEmitter.longAfter(ir.familyIdentity(), ":strideH=");
        long paddingHeight = CpuNormEmitter.longAfter(ir.familyIdentity(), ":padH=");
        long paddingWidth = CpuNormEmitter.longAfter(ir.familyIdentity(), ":padW=");
        long dilationHeight = CpuNormEmitter.longAfter(ir.familyIdentity(), ":dilationH=");
        long lanes = specialization.vectorSpeciesBitSize() / type.bitWidth();
        ClassDesc vectorClass = type == DataType.FLOAT64 ? DOUBLE_VECTOR : FLOAT_VECTOR;
        CpuCarrierEmitter carriers = new CpuCarrierEmitter(code);
        carriers.prepareVectorSpecies(type);
        carriers.prepareVectorByteOrder();
        // Keep verifier local types stable across scalar-only border paths and vector interiors.
        code.aconst_null().astore(vectorAccumulator)
                .aconst_null().astore(vectorInput)
                .aconst_null().astore(vectorWeight);
        Label done = code.newLabel();
        code.lload(start).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.lload(start).lstore(ordinal).lload(start).lstore(remaining);
        decodeLong(code, remaining, outputWidth, ow);
        decodeLong(code, remaining, outputHeight, oh);
        decodeLong(code, remaining, outputChannels, oc);
        code.lload(remaining).lstore(n);
        Label batches = code.newLabel();
        Label outputChannelLoop = code.newLabel();
        Label rows = code.newLabel();
        Label widths = code.newLabel();
        Label scalarCell = code.newLabel();
        Label advanceWidth = code.newLabel();
        code.labelBinding(batches).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.labelBinding(outputChannelLoop).lload(ordinal).lload(end).lcmp()
                .branch(Opcode.IFGE, done);
        code.lload(oc).lload(outputChannels).loadConstant(groups).ldiv().ldiv().lstore(group);
        code.lload(inputBase).lload(n).lload(inputBatchStride).lmul().ladd()
                .lload(group).lload(channelsPerGroup).lmul().lload(inputChannelStride).lmul()
                .ladd().lstore(inputBatchGroupBase);
        code.lload(weightBase).lload(oc).lload(weightOutputStride).lmul().ladd()
                .lstore(weightChannelBase);
        code.lload(outputBase).lload(n).lload(outputBatchStride).lmul().ladd()
                .lload(oc).lload(outputChannelStride).lmul().ladd().lstore(outputChannelBase);
        if (bias) {
            code.lload(biasBase).lload(oc).lload(biasStride).lmul().ladd().lstore(weightAddress);
            carriers.loadFrozen(type, specialization.carrierPattern().get(2), 2,
                    weightAddress, false);
            if (type == DataType.FLOAT64) code.dstore(biasValue);
            else code.fstore(biasValue);
        } else if (type == DataType.FLOAT64) code.loadConstant(0.0d).dstore(biasValue);
        else code.loadConstant(0.0f).fstore(biasValue);
        code.labelBinding(rows).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.lload(oh).loadConstant(strideHeight).lmul().loadConstant(paddingHeight).lsub()
                .lstore(inputHeightOrigin);
        code.lload(inputBatchGroupBase).lload(inputHeightOrigin).lload(inputHeightStride).lmul()
                .ladd().loadConstant(paddingWidth).lload(inputWidthStride).lmul().lsub()
                .lload(ow).lload(inputWidthStride).lmul().ladd().lstore(inputWidthCursor);
        code.lload(outputChannelBase).lload(oh).lload(outputHeightStride).lmul().ladd()
                .lload(ow).lload(outputWidthStride).lmul().ladd().lstore(outputCursor);
        Label rowNotInterior = code.newLabel();
        Label rowInteriorReady = code.newLabel();
        code.lload(inputHeightOrigin).loadConstant(0L).lcmp().branch(Opcode.IFLT, rowNotInterior);
        code.lload(inputHeightOrigin).lload(kernelHeight).loadConstant(1L).lsub()
                .loadConstant(dilationHeight).lmul().ladd().lload(inputHeight).lcmp()
                .branch(Opcode.IFGE, rowNotInterior);
        code.loadConstant(1).istore(rowInterior).branch(Opcode.GOTO, rowInteriorReady)
                .labelBinding(rowNotInterior).loadConstant(0).istore(rowInterior)
                .labelBinding(rowInteriorReady);
        code.labelBinding(widths).lload(ordinal).lload(end).lcmp().branch(Opcode.IFGE, done);
        code.iload(rowInterior).branch(Opcode.IFEQ, scalarCell);
        code.lload(ow).loadConstant(lanes).ladd().lload(outputWidth).lcmp()
                .branch(Opcode.IFGT, scalarCell);
        code.lload(ordinal).loadConstant(lanes).ladd().lload(end).lcmp()
                .branch(Opcode.IFGT, scalarCell);
        code.lload(ow).loadConstant(paddingWidth).lsub().loadConstant(0L).lcmp()
                .branch(Opcode.IFLT, scalarCell);
        code.lload(ow).loadConstant(paddingWidth).lsub().loadConstant(lanes).ladd()
                .lload(kernelWidth).ladd().loadConstant(2L).lsub().lload(inputWidth).lcmp()
                .branch(Opcode.IFGE, scalarCell);
        code.getstatic(vectorClass, "SPECIES_PREFERRED", VECTOR_SPECIES);
        if (type == DataType.FLOAT64) code.dload(biasValue); else code.fload(biasValue);
        code.invokestatic(vectorClass, "broadcast", MethodTypeDesc.of(vectorClass,
                VECTOR_SPECIES, type == DataType.FLOAT64
                        ? ConstantDescs.CD_double : ConstantDescs.CD_float))
                .astore(vectorAccumulator);
        code.loadConstant(0L).lstore(channel);
        Label vectorChannels = code.newLabel();
        Label vectorChannelsDone = code.newLabel();
        code.labelBinding(vectorChannels).lload(channel).lload(channelsPerGroup).lcmp()
                .branch(Opcode.IFGE, vectorChannelsDone).loadConstant(0L).lstore(kh);
        Label vectorHeights = code.newLabel();
        Label vectorHeightsDone = code.newLabel();
        code.labelBinding(vectorHeights).lload(kh).lload(kernelHeight).lcmp()
                .branch(Opcode.IFGE, vectorHeightsDone);
        code.lload(inputWidthCursor).lload(channel).lload(inputChannelStride).lmul().ladd()
                .lload(kh).loadConstant(dilationHeight).lmul().lload(inputHeightStride).lmul()
                .ladd().lstore(inputAddress);
        code.lload(weightChannelBase).lload(channel).lload(weightChannelStride).lmul().ladd()
                .lload(kh).lload(weightHeightStride).lmul().ladd().lstore(weightAddress);
        code.loadConstant(0L).lstore(kw);
        Label vectorKernelWidths = code.newLabel();
        Label vectorKernelWidthsDone = code.newLabel();
        code.labelBinding(vectorKernelWidths).lload(kw).lload(kernelWidth).lcmp()
                .branch(Opcode.IFGE, vectorKernelWidthsDone);
        carriers.vectorLoad(type, specialization.carrierPattern().get(0), 0,
                inputAddress, false, false);
        code.astore(vectorInput);
        carriers.vectorLoad(type, specialization.carrierPattern().get(1), 1,
                weightAddress, true, false);
        code.astore(vectorWeight);
        code.aload(vectorInput).aload(vectorWeight).invokevirtual(vectorClass, "mul",
                MethodTypeDesc.of(vectorClass, VECTOR)).astore(vectorInput);
        code.aload(vectorAccumulator).aload(vectorInput).invokevirtual(vectorClass, "add",
                MethodTypeDesc.of(vectorClass, VECTOR)).astore(vectorAccumulator);
        code.lload(inputAddress).lload(inputWidthStride).ladd().lstore(inputAddress)
                .lload(weightAddress).lload(weightWidthStride).ladd().lstore(weightAddress)
                .lload(kw).loadConstant(1L).ladd().lstore(kw)
                .branch(Opcode.GOTO, vectorKernelWidths).labelBinding(vectorKernelWidthsDone);
        code.lload(kh).loadConstant(1L).ladd().lstore(kh).branch(Opcode.GOTO, vectorHeights)
                .labelBinding(vectorHeightsDone);
        code.lload(channel).loadConstant(1L).ladd().lstore(channel)
                .branch(Opcode.GOTO, vectorChannels).labelBinding(vectorChannelsDone);
        carriers.vectorStore(type, specialization.carrierPattern().get(outputBoundary),
                outputBoundary, outputCursor, vectorAccumulator, false);
        code.lload(ow).loadConstant(lanes).ladd().lstore(ow)
                .lload(inputWidthCursor).loadConstant(lanes).lload(inputWidthStride).lmul().ladd()
                .lstore(inputWidthCursor)
                .lload(outputCursor).loadConstant(lanes).lload(outputWidthStride).lmul().ladd()
                .lstore(outputCursor)
                .lload(ordinal).loadConstant(lanes).ladd().lstore(ordinal)
                .branch(Opcode.GOTO, advanceWidth);
        code.labelBinding(scalarCell);
        if (type == DataType.FLOAT64) code.dload(biasValue).dstore(scalarAccumulator);
        else code.fload(biasValue).fstore(scalarAccumulator);
        code.loadConstant(0L).lstore(channel);
        Label scalarChannels = code.newLabel();
        Label scalarChannelsDone = code.newLabel();
        code.labelBinding(scalarChannels).lload(channel).lload(channelsPerGroup).lcmp()
                .branch(Opcode.IFGE, scalarChannelsDone).loadConstant(0L).lstore(kh);
        Label scalarHeights = code.newLabel();
        Label scalarHeightsDone = code.newLabel();
        code.labelBinding(scalarHeights).lload(kh).lload(kernelHeight).lcmp()
                .branch(Opcode.IFGE, scalarHeightsDone);
        code.lload(inputHeightOrigin).lload(kh).loadConstant(dilationHeight).lmul().ladd()
                .lstore(ih);
        code.lload(weightChannelBase).lload(channel).lload(weightChannelStride).lmul().ladd()
                .lload(kh).lload(weightHeightStride).lmul().ladd().lstore(weightAddress);
        code.loadConstant(0L).lstore(kw);
        Label scalarKernelWidths = code.newLabel();
        Label scalarKernelWidthsDone = code.newLabel();
        code.labelBinding(scalarKernelWidths).lload(kw).lload(kernelWidth).lcmp()
                .branch(Opcode.IFGE, scalarKernelWidthsDone);
        code.lload(ow).loadConstant(paddingWidth).lsub().lload(kw).ladd().lstore(iw);
        Label padded = code.newLabel();
        Label inputReady = code.newLabel();
        code.lload(ih).loadConstant(0L).lcmp().branch(Opcode.IFLT, padded)
                .lload(ih).lload(inputHeight).lcmp().branch(Opcode.IFGE, padded)
                .lload(iw).loadConstant(0L).lcmp().branch(Opcode.IFLT, padded)
                .lload(iw).lload(inputWidth).lcmp().branch(Opcode.IFGE, padded);
        code.lload(inputBatchGroupBase).lload(channel).lload(inputChannelStride).lmul().ladd()
                .lload(ih).lload(inputHeightStride).lmul().ladd()
                .lload(iw).lload(inputWidthStride).lmul().ladd().lstore(inputAddress);
        carriers.loadFrozen(type, specialization.carrierPattern().get(0), 0,
                inputAddress, false);
        if (type == DataType.FLOAT64) code.dstore(inputValue); else code.fstore(inputValue);
        code.branch(Opcode.GOTO, inputReady).labelBinding(padded);
        if (type == DataType.FLOAT64) code.loadConstant(0.0d).dstore(inputValue);
        else code.loadConstant(0.0f).fstore(inputValue);
        code.labelBinding(inputReady);
        carriers.loadFrozen(type, specialization.carrierPattern().get(1), 1,
                weightAddress, false);
        if (type == DataType.FLOAT64) {
            code.dstore(weightValue).dload(scalarAccumulator).dload(inputValue).dload(weightValue)
                    .dmul().dadd().dstore(scalarAccumulator);
        } else {
            code.fstore(weightValue).fload(scalarAccumulator).fload(inputValue).fload(weightValue)
                    .fmul().fadd().fstore(scalarAccumulator);
        }
        code.lload(weightAddress).lload(weightWidthStride).ladd().lstore(weightAddress)
                .lload(kw).loadConstant(1L).ladd().lstore(kw)
                .branch(Opcode.GOTO, scalarKernelWidths).labelBinding(scalarKernelWidthsDone);
        code.lload(kh).loadConstant(1L).ladd().lstore(kh).branch(Opcode.GOTO, scalarHeights)
                .labelBinding(scalarHeightsDone);
        code.lload(channel).loadConstant(1L).ladd().lstore(channel)
                .branch(Opcode.GOTO, scalarChannels).labelBinding(scalarChannelsDone);
        carriers.storeFrozen(type, specialization.carrierPattern().get(outputBoundary),
                outputBoundary, outputCursor, scalarAccumulator, false);
        code.lload(ow).loadConstant(1L).ladd().lstore(ow)
                .lload(inputWidthCursor).lload(inputWidthStride).ladd().lstore(inputWidthCursor)
                .lload(outputCursor).lload(outputWidthStride).ladd().lstore(outputCursor)
                .lload(ordinal).loadConstant(1L).ladd().lstore(ordinal);
        code.labelBinding(advanceWidth).lload(ow).lload(outputWidth).lcmp()
                .branch(Opcode.IFLT, widths);
        code.loadConstant(0L).lstore(ow).lload(oh).loadConstant(1L).ladd().lstore(oh)
                .lload(oh).lload(outputHeight).lcmp().branch(Opcode.IFLT, rows);
        code.loadConstant(0L).lstore(oh).lload(oc).loadConstant(1L).ladd().lstore(oc)
                .lload(oc).lload(outputChannels).lcmp().branch(Opcode.IFLT, outputChannelLoop);
        code.loadConstant(0L).lstore(oc).lload(n).loadConstant(1L).ladd().lstore(n)
                .branch(Opcode.GOTO, batches).labelBinding(done);
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
