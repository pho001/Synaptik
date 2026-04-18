package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedReductionHints;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import operations.loss.crossEntropyLossIndices;
import tensor.DataType;
import tensor.loss.LossReduction;
import tensor.Tensor;
import tensor.TensorMetadata;

final class CrossEntropyLossIndicesExecutor {
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int MIN_VECTOR_AXIS_MULTIPLIER = 4;

    private CrossEntropyLossIndicesExecutor() {}

    static void executeF64(crossEntropyLossIndices loss, Tensor logits, Tensor targetIndices, Tensor node, CpuKernelContext context) {
        validate(loss, logits, targetIndices, node, context);
        double[] logitsData = logits.getFloat64Data();
        if (loss.getReduction() == LossReduction.NONE) {
            double[] out = node.getFloat64Data();
            runGroups(logits, targetIndices, node, loss, context, (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                SampleResult result = computeLossF64(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                out[outOffset] = result.valid() ? result.loss() : 0.0d;
            });
            return;
        }
        ReductionResult result = reduceGroups(logits, targetIndices, loss, context, (baseLogits, targetOffset, axisStride, axisSize) ->
                computeLossF64(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex()));
        node.getFloat64Data()[node.getStorageOffsetUnsafe()] = finalizeReduction(result, loss.getReduction(), targetIndices.getFlatDataSize());
    }

    static void executeF32(crossEntropyLossIndices loss, Tensor logits, Tensor targetIndices, Tensor node, CpuKernelContext context) {
        validate(loss, logits, targetIndices, node, context);
        float[] logitsData = logits.getFloat32Data();
        if (loss.getReduction() == LossReduction.NONE) {
            float[] out = node.getFloat32Data();
            runGroups(logits, targetIndices, node, loss, context, (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                SampleResult result = computeLossF32(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                out[outOffset] = result.valid() ? (float) result.loss() : 0.0f;
            });
            return;
        }
        ReductionResult result = reduceGroups(logits, targetIndices, loss, context, (baseLogits, targetOffset, axisStride, axisSize) ->
                computeLossF32(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex()));
        node.getFloat32Data()[node.getStorageOffsetUnsafe()] = (float) finalizeReduction(result, loss.getReduction(), targetIndices.getFlatDataSize());
    }

    static void executeBF16(crossEntropyLossIndices loss, Tensor logits, Tensor targetIndices, Tensor node, CpuKernelContext context) {
        validate(loss, logits, targetIndices, node, context);
        short[] logitsData = logits.getBFloat16Data();
        if (loss.getReduction() == LossReduction.NONE) {
            short[] out = node.getBFloat16Data();
            runGroups(logits, targetIndices, node, loss, context, (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                SampleResult result = computeLossBF16(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                out[outOffset] = CpuDTypeOps.toBFloat16Bits(result.valid() ? (float) result.loss() : 0.0f);
            });
            return;
        }
        ReductionResult result = reduceGroups(logits, targetIndices, loss, context, (baseLogits, targetOffset, axisStride, axisSize) ->
                computeLossBF16(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex()));
        node.getBFloat16Data()[node.getStorageOffsetUnsafe()] = CpuDTypeOps.toBFloat16Bits((float) finalizeReduction(result, loss.getReduction(), targetIndices.getFlatDataSize()));
    }

    static void executeF32ToBF16(crossEntropyLossIndices loss, Tensor logits, float[] logitsData, Tensor targetIndices, Tensor node, CpuKernelContext context) {
        validate(loss, logits, targetIndices, node, context);
        if (logitsData == null) {
            throw new IllegalArgumentException("Float continuation logits cannot be null");
        }
        if (loss.getReduction() == LossReduction.NONE) {
            short[] out = node.getBFloat16Data();
            runGroups(logits, targetIndices, node, loss, context, (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                SampleResult result = computeLossF32(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                out[outOffset] = CpuDTypeOps.toBFloat16Bits(result.valid() ? (float) result.loss() : 0.0f);
            });
            return;
        }
        ReductionResult result = reduceGroups(logits, targetIndices, loss, context, (baseLogits, targetOffset, axisStride, axisSize) ->
                computeLossF32(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex()));
        node.getBFloat16Data()[node.getStorageOffsetUnsafe()] = CpuDTypeOps.toBFloat16Bits((float) finalizeReduction(result, loss.getReduction(), targetIndices.getFlatDataSize()));
    }

    private static void validate(crossEntropyLossIndices loss, Tensor logits, Tensor targetIndices, Tensor node, CpuKernelContext context) {
        if (loss == null || logits == null || targetIndices == null || node == null || context == null) {
            throw new IllegalArgumentException("cross entropy loss from indices execution arguments cannot be null");
        }
        if (targetIndices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("Target indices must be numeric integral values");
        }
        int[] logitsShape = logits.getShapeUnsafe();
        int classDimension = loss.getClassDimension();
        if (classDimension < 0 || classDimension >= logitsShape.length) {
            throw new IllegalArgumentException("Class dimension out of bounds: " + classDimension);
        }
        int[] expectedTargetShape = reduceShape(logitsShape, classDimension);
        validateShape(targetIndices.getShapeUnsafe(), expectedTargetShape, "Target indices shape must equal logits shape without class axis");
        if (loss.getReduction() == LossReduction.NONE) {
            validateShape(node.getShapeUnsafe(), expectedTargetShape, "NONE reduction output shape must equal target indices shape");
        } else if (node.getShapeUnsafe().length != 1 || node.getShapeUnsafe()[0] != 1) {
            throw new IllegalArgumentException("Reduced loss output shape must be [1]");
        }
    }

    private static void runGroups(
            Tensor logits,
            Tensor targetIndices,
            Tensor node,
            crossEntropyLossIndices loss,
            CpuKernelContext context,
            GroupWriter writer
    ) {
        int[] logitsShape = logits.getShapeUnsafe();
        int axis = loss.getClassDimension();
        int[] reducedShape = reduceShape(logitsShape, axis);
        int[] reducedDenseStrides = TensorMetadata.computeStrides(reducedShape);
        int groupCount = logicalSize(reducedShape);
        int axisSize = logitsShape[axis];
        int axisStride = logits.getStridesUnsafe()[axis];
        ResolvedReductionHints hints = context.reductionHints();

        if (canUseDenseContiguousLastAxisFastPath(logits, targetIndices, node, axis)) {
            runDenseContiguousGroups(groupCount, axisSize, logits.getStorageOffsetUnsafe(), targetIndices.getStorageOffsetUnsafe(), node.getStorageOffsetUnsafe(), hints,
                    (group, baseLogits, targetOffset, outOffset) -> writer.write(baseLogits, targetOffset, outOffset, 1, axisSize));
            return;
        }

        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                for (int group = start; group < end; group++) {
                    GroupState state = groupState(group, logitsShape, logits.getStridesUnsafe(), logits.getStorageOffsetUnsafe(),
                            targetIndices.getStridesUnsafe(), targetIndices.getStorageOffsetUnsafe(),
                            node.getStridesUnsafe(), node.getStorageOffsetUnsafe(), axis, reducedDenseStrides, axisSize, axisStride);
                    writer.write(state.baseLogits(), state.targetOffset(), state.outOffset(), state.axisStride(), state.axisSize());
                }
            });
            return;
        }

        for (int group = 0; group < groupCount; group++) {
            GroupState state = groupState(group, logitsShape, logits.getStridesUnsafe(), logits.getStorageOffsetUnsafe(),
                    targetIndices.getStridesUnsafe(), targetIndices.getStorageOffsetUnsafe(),
                    node.getStridesUnsafe(), node.getStorageOffsetUnsafe(), axis, reducedDenseStrides, axisSize, axisStride);
            writer.write(state.baseLogits(), state.targetOffset(), state.outOffset(), state.axisStride(), state.axisSize());
        }
    }

    private static ReductionResult reduceGroups(
            Tensor logits,
            Tensor targetIndices,
            crossEntropyLossIndices loss,
            CpuKernelContext context,
            GroupReducer reducer
    ) {
        int[] logitsShape = logits.getShapeUnsafe();
        int axis = loss.getClassDimension();
        int[] reducedShape = reduceShape(logitsShape, axis);
        int[] reducedDenseStrides = TensorMetadata.computeStrides(reducedShape);
        int groupCount = logicalSize(reducedShape);
        int axisSize = logitsShape[axis];
        int axisStride = logits.getStridesUnsafe()[axis];
        ResolvedReductionHints hints = context.reductionHints();

        if (canUseDenseContiguousLastAxisFastPath(logits, targetIndices, null, axis)) {
            return reduceDenseContiguousGroups(groupCount, axisSize, logits.getStorageOffsetUnsafe(), targetIndices.getStorageOffsetUnsafe(), hints, reducer);
        }

        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            double[] partialLosses = new double[chunks];
            int[] partialValid = new int[chunks];
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                double partialLoss = 0.0d;
                int partialCount = 0;
                for (int group = start; group < end; group++) {
                    GroupState state = groupState(group, logitsShape, logits.getStridesUnsafe(), logits.getStorageOffsetUnsafe(),
                            targetIndices.getStridesUnsafe(), targetIndices.getStorageOffsetUnsafe(),
                            null, 0, axis, reducedDenseStrides, axisSize, axisStride);
                    SampleResult result = reducer.compute(state.baseLogits(), state.targetOffset(), state.axisStride(), state.axisSize());
                    if (result.valid()) {
                        partialLoss += result.loss();
                        partialCount++;
                    }
                }
                partialLosses[chunk] = partialLoss;
                partialValid[chunk] = partialCount;
            });
            double totalLoss = 0.0d;
            int totalValid = 0;
            for (int i = 0; i < chunks; i++) {
                totalLoss += partialLosses[i];
                totalValid += partialValid[i];
            }
            return new ReductionResult(totalLoss, totalValid);
        }

        double totalLoss = 0.0d;
        int totalValid = 0;
        for (int group = 0; group < groupCount; group++) {
            GroupState state = groupState(group, logitsShape, logits.getStridesUnsafe(), logits.getStorageOffsetUnsafe(),
                    targetIndices.getStridesUnsafe(), targetIndices.getStorageOffsetUnsafe(),
                    null, 0, axis, reducedDenseStrides, axisSize, axisStride);
            SampleResult result = reducer.compute(state.baseLogits(), state.targetOffset(), state.axisStride(), state.axisSize());
            if (result.valid()) {
                totalLoss += result.loss();
                totalValid++;
            }
        }
        return new ReductionResult(totalLoss, totalValid);
    }

    private static ReductionResult reduceDenseContiguousGroups(
            int groupCount,
            int axisSize,
            int logitsBaseOffset,
            int targetBaseOffset,
            ResolvedReductionHints hints,
            GroupReducer reducer
    ) {
        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            double[] partialLosses = new double[chunks];
            int[] partialValid = new int[chunks];
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                double partialLoss = 0.0d;
                int partialCount = 0;
                for (int group = start; group < end; group++) {
                    SampleResult result = reducer.compute(logitsBaseOffset + group * axisSize, targetBaseOffset + group, 1, axisSize);
                    if (result.valid()) {
                        partialLoss += result.loss();
                        partialCount++;
                    }
                }
                partialLosses[chunk] = partialLoss;
                partialValid[chunk] = partialCount;
            });
            double totalLoss = 0.0d;
            int totalValid = 0;
            for (int i = 0; i < chunks; i++) {
                totalLoss += partialLosses[i];
                totalValid += partialValid[i];
            }
            return new ReductionResult(totalLoss, totalValid);
        }

        double totalLoss = 0.0d;
        int totalValid = 0;
        for (int group = 0; group < groupCount; group++) {
            SampleResult result = reducer.compute(logitsBaseOffset + group * axisSize, targetBaseOffset + group, 1, axisSize);
            if (result.valid()) {
                totalLoss += result.loss();
                totalValid++;
            }
        }
        return new ReductionResult(totalLoss, totalValid);
    }

    private static void runDenseContiguousGroups(
            int groupCount,
            int axisSize,
            int logitsBaseOffset,
            int targetBaseOffset,
            int outBaseOffset,
            ResolvedReductionHints hints,
            DenseGroupWriter writer
    ) {
        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                for (int group = start; group < end; group++) {
                    writer.write(group, logitsBaseOffset + group * axisSize, targetBaseOffset + group, outBaseOffset + group);
                }
            });
            return;
        }
        for (int group = 0; group < groupCount; group++) {
            writer.write(group, logitsBaseOffset + group * axisSize, targetBaseOffset + group, outBaseOffset + group);
        }
    }

    private static GroupState groupState(
            int reducedIndex,
            int[] logitsShape,
            int[] logitsStrides,
            int logitsBaseOffset,
            int[] targetStrides,
            int targetBaseOffset,
            int[] outStrides,
            int outBaseOffset,
            int axis,
            int[] reducedDenseStrides,
            int axisSize,
            int axisStride
    ) {
        int rem = reducedIndex;
        int baseLogits = logitsBaseOffset;
        int targetOffset = targetBaseOffset;
        int outputOffset = outStrides == null ? -1 : outBaseOffset;
        for (int d = 0, rd = 0; d < logitsShape.length; d++) {
            if (d == axis) {
                continue;
            }
            int coord = rem / reducedDenseStrides[rd];
            rem %= reducedDenseStrides[rd];
            baseLogits += coord * logitsStrides[d];
            targetOffset += coord * targetStrides[rd];
            if (outStrides != null) {
                outputOffset += coord * outStrides[rd];
            }
            rd++;
        }
        return new GroupState(baseLogits, targetOffset, outputOffset, axisStride, axisSize);
    }

    private static boolean canUseDenseContiguousLastAxisFastPath(Tensor logits, Tensor targetIndices, Tensor out, int axis) {
        return axis == logits.getShapeUnsafe().length - 1
                && isDenseContiguous(logits.getShapeUnsafe(), logits.getStridesUnsafe())
                && logits.getStridesUnsafe()[axis] == 1
                && isDenseContiguous(targetIndices.getShapeUnsafe(), targetIndices.getStridesUnsafe())
                && (out == null || isDenseContiguous(out.getShapeUnsafe(), out.getStridesUnsafe()));
    }

    private static boolean isDenseContiguous(int[] shape, int[] strides) {
        if (shape.length != strides.length) {
            return false;
        }
        int expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expected) {
                return false;
            }
            expected *= shape[i];
        }
        return true;
    }

    private static SampleResult computeLossF64(double[] logits, int baseLogits, int axisStride, int axisSize, int targetIndex, Integer ignoreIndex) {
        if (ignoreIndex != null && targetIndex == ignoreIndex) {
            return SampleResult.ignored();
        }
        validateTargetIndex(targetIndex, axisSize);
        double targetLogit = logits[baseLogits + targetIndex * axisStride];
        if (axisStride == 1 && axisSize >= F64_SPECIES.length() * MIN_VECTOR_AXIS_MULTIPLIER) {
            return new SampleResult(logSumExpContiguousF64(logits, baseLogits, axisSize) - targetLogit, true);
        }
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            max = Math.max(max, logits[offset]);
        }
        double sumExp = 0.0d;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            sumExp += Math.exp(logits[offset] - max);
        }
        return new SampleResult(max + Math.log(sumExp) - targetLogit, true);
    }

    private static SampleResult computeLossF32(float[] logits, int baseLogits, int axisStride, int axisSize, int targetIndex, Integer ignoreIndex) {
        if (ignoreIndex != null && targetIndex == ignoreIndex) {
            return SampleResult.ignored();
        }
        validateTargetIndex(targetIndex, axisSize);
        float targetLogit = logits[baseLogits + targetIndex * axisStride];
        if (axisStride == 1 && axisSize >= F32_SPECIES.length() * MIN_VECTOR_AXIS_MULTIPLIER) {
            return new SampleResult(logSumExpContiguousF32(logits, baseLogits, axisSize) - targetLogit, true);
        }
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            max = Math.max(max, logits[offset]);
        }
        double sumExp = 0.0d;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            sumExp += Math.exp(logits[offset] - max);
        }
        return new SampleResult(max + Math.log(sumExp) - targetLogit, true);
    }

    private static SampleResult computeLossBF16(short[] logits, int baseLogits, int axisStride, int axisSize, int targetIndex, Integer ignoreIndex) {
        if (ignoreIndex != null && targetIndex == ignoreIndex) {
            return SampleResult.ignored();
        }
        validateTargetIndex(targetIndex, axisSize);
        float targetLogit = CpuDTypeOps.fromBFloat16Bits(logits[baseLogits + targetIndex * axisStride]);
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            max = Math.max(max, CpuDTypeOps.fromBFloat16Bits(logits[offset]));
        }
        double sumExp = 0.0d;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            sumExp += Math.exp(CpuDTypeOps.fromBFloat16Bits(logits[offset]) - max);
        }
        return new SampleResult(max + Math.log(sumExp) - targetLogit, true);
    }

    private static double logSumExpContiguousF64(double[] logits, int base, int length) {
        double max = maxContiguousF64(logits, base, length);
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector maxVector = DoubleVector.broadcast(F64_SPECIES, max);
        DoubleVector sumVector = DoubleVector.zero(F64_SPECIES);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            DoubleVector values = DoubleVector.fromArray(F64_SPECIES, logits, base + i).sub(maxVector).lanewise(VectorOperators.EXP);
            sumVector = sumVector.add(values);
        }
        double sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            sum += Math.exp(logits[base + i] - max);
        }
        return max + Math.log(sum);
    }

    private static double logSumExpContiguousF32(float[] logits, int base, int length) {
        float max = maxContiguousF32(logits, base, length);
        int upper = F32_SPECIES.loopBound(length);
        FloatVector maxVector = FloatVector.broadcast(F32_SPECIES, max);
        FloatVector sumVector = FloatVector.zero(F32_SPECIES);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            FloatVector values = FloatVector.fromArray(F32_SPECIES, logits, base + i).sub(maxVector).lanewise(VectorOperators.EXP);
            sumVector = sumVector.add(values);
        }
        double sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            sum += Math.exp(logits[base + i] - max);
        }
        return max + Math.log(sum);
    }

    private static double maxContiguousF64(double[] logits, int base, int length) {
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector vectorMax = DoubleVector.broadcast(F64_SPECIES, Double.NEGATIVE_INFINITY);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            vectorMax = vectorMax.max(DoubleVector.fromArray(F64_SPECIES, logits, base + i));
        }
        double max = vectorMax.reduceLanes(VectorOperators.MAX);
        for (; i < length; i++) {
            max = Math.max(max, logits[base + i]);
        }
        return max;
    }

    private static float maxContiguousF32(float[] logits, int base, int length) {
        int upper = F32_SPECIES.loopBound(length);
        FloatVector vectorMax = FloatVector.broadcast(F32_SPECIES, Float.NEGATIVE_INFINITY);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            vectorMax = vectorMax.max(FloatVector.fromArray(F32_SPECIES, logits, base + i));
        }
        float max = vectorMax.reduceLanes(VectorOperators.MAX);
        for (; i < length; i++) {
            max = Math.max(max, logits[base + i]);
        }
        return max;
    }

    private static int readIndex(Tensor targetIndices, int storageOffset) {
        return switch (targetIndices.getDataType()) {
            case INT32 -> targetIndices.getInt32Data()[storageOffset];
            case FLOAT64 -> toIntegralIndex(targetIndices.getFloat64Data()[storageOffset]);
            case FLOAT32 -> toIntegralIndex(targetIndices.getFloat32Data()[storageOffset]);
            case BFLOAT16 -> toIntegralIndex(CpuDTypeOps.fromBFloat16Bits(targetIndices.getBFloat16Data()[storageOffset]));
            case BOOL -> throw new IllegalArgumentException("Target indices must be numeric integral values");
        };
    }

    private static int toIntegralIndex(double raw) {
        if (!Double.isFinite(raw)) {
            throw new IllegalArgumentException("Index tensor contains non-finite value");
        }
        long integral = Math.round(raw);
        if (Math.abs(raw - integral) > 1e-9) {
            throw new IllegalArgumentException("Index tensor contains non-integral value: " + raw);
        }
        return Math.toIntExact(integral);
    }

    private static int toIntegralIndex(float raw) {
        if (!Float.isFinite(raw)) {
            throw new IllegalArgumentException("Index tensor contains non-finite value");
        }
        int integral = Math.round(raw);
        if (Math.abs(raw - integral) > 1e-6f) {
            throw new IllegalArgumentException("Index tensor contains non-integral value: " + raw);
        }
        return integral;
    }

    private static void validateTargetIndex(int targetIndex, int axisSize) {
        if (targetIndex < 0 || targetIndex >= axisSize) {
            throw new IllegalArgumentException("Target index out of range: " + targetIndex + " for classes=" + axisSize);
        }
    }

    private static double finalizeReduction(ReductionResult result, LossReduction reduction, int totalGroups) {
        return switch (reduction) {
            case NONE -> throw new IllegalStateException("NONE reduction does not produce scalar output");
            case SUM -> result.totalLoss();
            case MEAN -> result.totalLoss() / Math.max(1, result.validCount() == 0 ? 0 : result.validCount());
        };
    }

    private static int[] reduceShape(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reduced[j++] = shape[i];
            }
        }
        return reduced;
    }

    private static int logicalSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    private static void validateShape(int[] actual, int[] expected, String message) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException(message);
        }
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] != expected[i]) {
                throw new IllegalArgumentException(message);
            }
        }
    }

    @FunctionalInterface
    private interface GroupWriter {
        void write(int baseLogits, int targetOffset, int outOffset, int axisStride, int axisSize);
    }

    @FunctionalInterface
    private interface DenseGroupWriter {
        void write(int group, int baseLogits, int targetOffset, int outOffset);
    }

    @FunctionalInterface
    private interface GroupReducer {
        SampleResult compute(int baseLogits, int targetOffset, int axisStride, int axisSize);
    }

    private record GroupState(int baseLogits, int targetOffset, int outOffset, int axisStride, int axisSize) {}

    private record SampleResult(double loss, boolean valid) {
        static SampleResult ignored() {
            return new SampleResult(0.0d, false);
        }
    }

    private record ReductionResult(double totalLoss, int validCount) {}
}
