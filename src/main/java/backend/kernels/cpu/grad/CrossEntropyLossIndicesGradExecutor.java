package backend.kernels.cpu.grad;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedReductionHints;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import operations.crossEntropyLossIndicesGrad;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;

final class CrossEntropyLossIndicesGradExecutor {
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int MIN_VECTOR_AXIS_MULTIPLIER = 4;

    private CrossEntropyLossIndicesGradExecutor() {}

    static void executeF64(crossEntropyLossIndicesGrad grad, Tensor logits, Tensor targetIndices, Tensor sampleScale, Tensor node, CpuKernelContext context) {
        validate(grad, logits, targetIndices, sampleScale, node, context);
        double[] logitsData = logits.getFloat64Data();
        double[] scaleData = sampleScale.getFloat64Data();
        double[] out = node.getFloat64Data();
        runGroups(logits, targetIndices, sampleScale, node, grad.getClassDimension(), context,
                (baseLogits, targetOffset, scaleOffset, outOffset, axisStride, axisSize, outAxisStride) ->
                        computeGroupF64(logitsData, scaleData, out, baseLogits, targetOffset, scaleOffset, outOffset, axisStride, axisSize, outAxisStride, targetIndices));
    }

    static void executeF32(crossEntropyLossIndicesGrad grad, Tensor logits, Tensor targetIndices, Tensor sampleScale, Tensor node, CpuKernelContext context) {
        validate(grad, logits, targetIndices, sampleScale, node, context);
        float[] logitsData = logits.getFloat32Data();
        float[] scaleData = sampleScale.getFloat32Data();
        float[] out = node.getFloat32Data();
        runGroups(logits, targetIndices, sampleScale, node, grad.getClassDimension(), context,
                (baseLogits, targetOffset, scaleOffset, outOffset, axisStride, axisSize, outAxisStride) ->
                        computeGroupF32(logitsData, scaleData, out, baseLogits, targetOffset, scaleOffset, outOffset, axisStride, axisSize, outAxisStride, targetIndices));
    }

    static void executeBF16(crossEntropyLossIndicesGrad grad, Tensor logits, Tensor targetIndices, Tensor sampleScale, Tensor node, CpuKernelContext context) {
        validate(grad, logits, targetIndices, sampleScale, node, context);
        short[] logitsData = logits.getBFloat16Data();
        short[] scaleData = sampleScale.getBFloat16Data();
        short[] out = node.getBFloat16Data();
        runGroups(logits, targetIndices, sampleScale, node, grad.getClassDimension(), context,
                (baseLogits, targetOffset, scaleOffset, outOffset, axisStride, axisSize, outAxisStride) ->
                        computeGroupBF16(logitsData, scaleData, out, baseLogits, targetOffset, scaleOffset, outOffset, axisStride, axisSize, outAxisStride, targetIndices));
    }

    private static void validate(
            crossEntropyLossIndicesGrad grad,
            Tensor logits,
            Tensor targetIndices,
            Tensor sampleScale,
            Tensor node,
            CpuKernelContext context
    ) {
        if (grad == null || logits == null || targetIndices == null || sampleScale == null || node == null || context == null) {
            throw new IllegalArgumentException("cross entropy loss indices grad execution arguments cannot be null");
        }
        if (logits.getDataType() == DataType.BOOL || logits.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("cross entropy loss indices grad requires floating logits");
        }
        if (targetIndices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("cross entropy loss indices grad requires numeric integral target indices");
        }
        if (sampleScale.getDataType() != logits.getDataType() || node.getDataType() != logits.getDataType()) {
            throw new IllegalArgumentException("cross entropy loss indices grad requires matching floating dtypes");
        }

        int[] logitsShape = logits.getShapeUnsafe();
        int axis = grad.getClassDimension();
        if (axis < 0 || axis >= logitsShape.length) {
            throw new IllegalArgumentException("Class dimension out of bounds: " + axis);
        }
        validateSameShape(node.getShapeUnsafe(), logitsShape, "cross entropy loss indices grad output shape must match logits shape");
        int[] reducedShape = reduceShape(logitsShape, axis);
        validateSameShape(targetIndices.getShapeUnsafe(), reducedShape, "cross entropy loss indices grad indices shape must equal logits shape without class axis");
        validateSameShape(sampleScale.getShapeUnsafe(), reducedShape, "cross entropy loss indices grad scale shape must equal logits shape without class axis");
    }

    private static void computeGroupF64(
            double[] logits,
            double[] scale,
            double[] out,
            int baseLogits,
            int targetOffset,
            int scaleOffset,
            int outOffset,
            int axisStride,
            int axisSize,
            int outAxisStride,
            Tensor targetIndices
    ) {
        double sampleScale = scale[scaleOffset];
        if (sampleScale == 0.0d) {
            zeroOutF64(out, outOffset, outAxisStride, axisSize);
            return;
        }
        int targetIndex = readIndex(targetIndices, targetOffset);
        validateTargetIndex(targetIndex, axisSize);
        if (axisStride == 1 && outAxisStride == 1 && axisSize >= F64_SPECIES.length() * MIN_VECTOR_AXIS_MULTIPLIER) {
            computeContiguousF64(logits, out, baseLogits, outOffset, axisSize, sampleScale, targetIndex);
            return;
        }

        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, logitsOffset = baseLogits; i < axisSize; i++, logitsOffset += axisStride) {
            max = Math.max(max, logits[logitsOffset]);
        }
        double sum = 0.0d;
        for (int i = 0, logitsOffset = baseLogits, outputOffset = outOffset; i < axisSize; i++, logitsOffset += axisStride, outputOffset += outAxisStride) {
            double value = Math.exp(logits[logitsOffset] - max);
            out[outputOffset] = value;
            sum += value;
        }
        double inv = sampleScale / sum;
        for (int i = 0, outputOffset = outOffset; i < axisSize; i++, outputOffset += outAxisStride) {
            out[outputOffset] *= inv;
        }
        out[outOffset + targetIndex * outAxisStride] -= sampleScale;
    }

    private static void computeGroupF32(
            float[] logits,
            float[] scale,
            float[] out,
            int baseLogits,
            int targetOffset,
            int scaleOffset,
            int outOffset,
            int axisStride,
            int axisSize,
            int outAxisStride,
            Tensor targetIndices
    ) {
        float sampleScale = scale[scaleOffset];
        if (sampleScale == 0.0f) {
            zeroOutF32(out, outOffset, outAxisStride, axisSize);
            return;
        }
        int targetIndex = readIndex(targetIndices, targetOffset);
        validateTargetIndex(targetIndex, axisSize);
        if (axisStride == 1 && outAxisStride == 1 && axisSize >= F32_SPECIES.length() * MIN_VECTOR_AXIS_MULTIPLIER) {
            computeContiguousF32(logits, out, baseLogits, outOffset, axisSize, sampleScale, targetIndex);
            return;
        }

        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, logitsOffset = baseLogits; i < axisSize; i++, logitsOffset += axisStride) {
            max = Math.max(max, logits[logitsOffset]);
        }
        float sum = 0.0f;
        for (int i = 0, logitsOffset = baseLogits, outputOffset = outOffset; i < axisSize; i++, logitsOffset += axisStride, outputOffset += outAxisStride) {
            float value = (float) Math.exp(logits[logitsOffset] - max);
            out[outputOffset] = value;
            sum += value;
        }
        float inv = sampleScale / sum;
        for (int i = 0, outputOffset = outOffset; i < axisSize; i++, outputOffset += outAxisStride) {
            out[outputOffset] *= inv;
        }
        out[outOffset + targetIndex * outAxisStride] -= sampleScale;
    }

    private static void computeGroupBF16(
            short[] logits,
            short[] scale,
            short[] out,
            int baseLogits,
            int targetOffset,
            int scaleOffset,
            int outOffset,
            int axisStride,
            int axisSize,
            int outAxisStride,
            Tensor targetIndices
    ) {
        float sampleScale = CpuDTypeOps.fromBFloat16Bits(scale[scaleOffset]);
        if (sampleScale == 0.0f) {
            zeroOutBF16(out, outOffset, outAxisStride, axisSize);
            return;
        }
        int targetIndex = readIndex(targetIndices, targetOffset);
        validateTargetIndex(targetIndex, axisSize);

        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, logitsOffset = baseLogits; i < axisSize; i++, logitsOffset += axisStride) {
            max = Math.max(max, CpuDTypeOps.fromBFloat16Bits(logits[logitsOffset]));
        }
        float sum = 0.0f;
        for (int i = 0, logitsOffset = baseLogits, outputOffset = outOffset; i < axisSize; i++, logitsOffset += axisStride, outputOffset += outAxisStride) {
            float value = (float) Math.exp(CpuDTypeOps.fromBFloat16Bits(logits[logitsOffset]) - max);
            out[outputOffset] = CpuDTypeOps.toBFloat16Bits(value);
            sum += value;
        }
        float inv = sampleScale / sum;
        for (int i = 0, outputOffset = outOffset; i < axisSize; i++, outputOffset += outAxisStride) {
            float value = CpuDTypeOps.fromBFloat16Bits(out[outputOffset]) * inv;
            if (i == targetIndex) {
                value -= sampleScale;
            }
            out[outputOffset] = CpuDTypeOps.toBFloat16Bits(value);
        }
    }

    private static void computeContiguousF64(
            double[] logits,
            double[] out,
            int baseLogits,
            int baseOut,
            int axisSize,
            double sampleScale,
            int targetIndex
    ) {
        double max = maxContiguousF64(logits, baseLogits, axisSize);
        double sum = expIntoOutAndAccumulateF64(logits, out, baseLogits, baseOut, axisSize, max);
        scaleOutContiguousF64(out, baseOut, axisSize, sampleScale / sum);
        out[baseOut + targetIndex] -= sampleScale;
    }

    private static void computeContiguousF32(
            float[] logits,
            float[] out,
            int baseLogits,
            int baseOut,
            int axisSize,
            float sampleScale,
            int targetIndex
    ) {
        float max = maxContiguousF32(logits, baseLogits, axisSize);
        float sum = expIntoOutAndAccumulateF32(logits, out, baseLogits, baseOut, axisSize, max);
        scaleOutContiguousF32(out, baseOut, axisSize, sampleScale / sum);
        out[baseOut + targetIndex] -= sampleScale;
    }

    private static void runGroups(
            Tensor logits,
            Tensor targetIndices,
            Tensor sampleScale,
            Tensor node,
            int axis,
            CpuKernelContext context,
            GroupComputer computer
    ) {
        int[] shape = logits.getShapeUnsafe();
        int[] reducedShape = reduceShape(shape, axis);
        int[] reducedDenseStrides = TensorMetadata.computeStrides(reducedShape);
        int groupCount = logicalSize(reducedShape);
        int axisSize = shape[axis];
        int axisStrideIn = logits.getStridesUnsafe()[axis];
        int axisStrideOut = node.getStridesUnsafe()[axis];
        ResolvedReductionHints hints = context.reductionHints();

        if (canUseDenseContiguousLastAxisFastPath(logits, targetIndices, sampleScale, node, axis, axisStrideIn, axisStrideOut)) {
            runDenseContiguousGroups(groupCount, axisSize, logits.getStorageOffsetUnsafe(), targetIndices.getStorageOffsetUnsafe(),
                    sampleScale.getStorageOffsetUnsafe(), node.getStorageOffsetUnsafe(), hints,
                    (group, baseLogits, targetOffset, scaleOffset, outOffset) ->
                            computer.compute(baseLogits, targetOffset, scaleOffset, outOffset, 1, axisSize, 1));
            return;
        }

        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                for (int group = start; group < end; group++) {
                    GroupState state = groupState(group, shape, logits.getStridesUnsafe(), logits.getStorageOffsetUnsafe(),
                            targetIndices.getStridesUnsafe(), targetIndices.getStorageOffsetUnsafe(),
                            sampleScale.getStridesUnsafe(), sampleScale.getStorageOffsetUnsafe(),
                            node.getStridesUnsafe(), node.getStorageOffsetUnsafe(),
                            axis, reducedDenseStrides, axisSize, axisStrideIn, axisStrideOut);
                    computer.compute(state.baseLogits(), state.targetOffset(), state.scaleOffset(), state.outOffset(), state.axisStrideIn(), state.axisSize(), state.axisStrideOut());
                }
            });
            return;
        }

        for (int group = 0; group < groupCount; group++) {
            GroupState state = groupState(group, shape, logits.getStridesUnsafe(), logits.getStorageOffsetUnsafe(),
                    targetIndices.getStridesUnsafe(), targetIndices.getStorageOffsetUnsafe(),
                    sampleScale.getStridesUnsafe(), sampleScale.getStorageOffsetUnsafe(),
                    node.getStridesUnsafe(), node.getStorageOffsetUnsafe(),
                    axis, reducedDenseStrides, axisSize, axisStrideIn, axisStrideOut);
            computer.compute(state.baseLogits(), state.targetOffset(), state.scaleOffset(), state.outOffset(), state.axisStrideIn(), state.axisSize(), state.axisStrideOut());
        }
    }

    private static void runDenseContiguousGroups(
            int groupCount,
            int axisSize,
            int logitsBaseOffset,
            int targetBaseOffset,
            int scaleBaseOffset,
            int outBaseOffset,
            ResolvedReductionHints hints,
            DenseGroupComputer computer
    ) {
        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                for (int group = start; group < end; group++) {
                    computer.compute(group, logitsBaseOffset + group * axisSize, targetBaseOffset + group, scaleBaseOffset + group, outBaseOffset + group * axisSize);
                }
            });
            return;
        }
        for (int group = 0; group < groupCount; group++) {
            computer.compute(group, logitsBaseOffset + group * axisSize, targetBaseOffset + group, scaleBaseOffset + group, outBaseOffset + group * axisSize);
        }
    }

    private static GroupState groupState(
            int reducedIndex,
            int[] logitsShape,
            int[] logitsStrides,
            int logitsBaseOffset,
            int[] targetStrides,
            int targetBaseOffset,
            int[] scaleStrides,
            int scaleBaseOffset,
            int[] outStrides,
            int outBaseOffset,
            int axis,
            int[] reducedDenseStrides,
            int axisSize,
            int axisStrideIn,
            int axisStrideOut
    ) {
        int rem = reducedIndex;
        int baseLogits = logitsBaseOffset;
        int targetOffset = targetBaseOffset;
        int scaleOffset = scaleBaseOffset;
        int outOffset = outBaseOffset;
        for (int d = 0, rd = 0; d < logitsShape.length; d++) {
            if (d == axis) {
                continue;
            }
            int coord = rem / reducedDenseStrides[rd];
            rem %= reducedDenseStrides[rd];
            baseLogits += coord * logitsStrides[d];
            targetOffset += coord * targetStrides[rd];
            scaleOffset += coord * scaleStrides[rd];
            outOffset += coord * outStrides[d];
            rd++;
        }
        return new GroupState(baseLogits, targetOffset, scaleOffset, outOffset, axisStrideIn, axisSize, axisStrideOut);
    }

    private static boolean canUseDenseContiguousLastAxisFastPath(
            Tensor logits,
            Tensor targetIndices,
            Tensor sampleScale,
            Tensor out,
            int axis,
            int axisStrideIn,
            int axisStrideOut
    ) {
        return axis == logits.getShapeUnsafe().length - 1
                && axisStrideIn == 1
                && axisStrideOut == 1
                && isDenseContiguous(logits.getShapeUnsafe(), logits.getStridesUnsafe())
                && isDenseContiguous(targetIndices.getShapeUnsafe(), targetIndices.getStridesUnsafe())
                && isDenseContiguous(sampleScale.getShapeUnsafe(), sampleScale.getStridesUnsafe())
                && isDenseContiguous(out.getShapeUnsafe(), out.getStridesUnsafe());
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

    private static void validateSameShape(int[] actual, int[] expected, String message) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException(message);
        }
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] != expected[i]) {
                throw new IllegalArgumentException(message);
            }
        }
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

    private static void zeroOutF64(double[] out, int baseOut, int axisStrideOut, int axisSize) {
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            out[outOffset] = 0.0d;
        }
    }

    private static void zeroOutF32(float[] out, int baseOut, int axisStrideOut, int axisSize) {
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            out[outOffset] = 0.0f;
        }
    }

    private static void zeroOutBF16(short[] out, int baseOut, int axisStrideOut, int axisSize) {
        for (int i = 0, outOffset = baseOut; i < axisSize; i++, outOffset += axisStrideOut) {
            out[outOffset] = 0;
        }
    }

    private static double maxContiguousF64(double[] in, int base, int length) {
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector vectorMax = DoubleVector.broadcast(F64_SPECIES, Double.NEGATIVE_INFINITY);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            vectorMax = vectorMax.max(DoubleVector.fromArray(F64_SPECIES, in, base + i));
        }
        double max = vectorMax.reduceLanes(VectorOperators.MAX);
        for (; i < length; i++) {
            max = Math.max(max, in[base + i]);
        }
        return max;
    }

    private static float maxContiguousF32(float[] in, int base, int length) {
        int upper = F32_SPECIES.loopBound(length);
        FloatVector vectorMax = FloatVector.broadcast(F32_SPECIES, Float.NEGATIVE_INFINITY);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            vectorMax = vectorMax.max(FloatVector.fromArray(F32_SPECIES, in, base + i));
        }
        float max = vectorMax.reduceLanes(VectorOperators.MAX);
        for (; i < length; i++) {
            max = Math.max(max, in[base + i]);
        }
        return max;
    }

    private static double expIntoOutAndAccumulateF64(double[] in, double[] out, int baseIn, int baseOut, int length, double max) {
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector sumVector = DoubleVector.zero(F64_SPECIES);
        DoubleVector maxVector = DoubleVector.broadcast(F64_SPECIES, max);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            DoubleVector values = DoubleVector.fromArray(F64_SPECIES, in, baseIn + i).sub(maxVector).lanewise(VectorOperators.EXP);
            values.intoArray(out, baseOut + i);
            sumVector = sumVector.add(values);
        }
        double sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            double value = Math.exp(in[baseIn + i] - max);
            out[baseOut + i] = value;
            sum += value;
        }
        return sum;
    }

    private static float expIntoOutAndAccumulateF32(float[] in, float[] out, int baseIn, int baseOut, int length, float max) {
        int upper = F32_SPECIES.loopBound(length);
        FloatVector sumVector = FloatVector.zero(F32_SPECIES);
        FloatVector maxVector = FloatVector.broadcast(F32_SPECIES, max);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            FloatVector values = FloatVector.fromArray(F32_SPECIES, in, baseIn + i).sub(maxVector).lanewise(VectorOperators.EXP);
            values.intoArray(out, baseOut + i);
            sumVector = sumVector.add(values);
        }
        float sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            float value = (float) Math.exp(in[baseIn + i] - max);
            out[baseOut + i] = value;
            sum += value;
        }
        return sum;
    }

    private static void scaleOutContiguousF64(double[] out, int baseOut, int length, double scale) {
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector scaleVector = DoubleVector.broadcast(F64_SPECIES, scale);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            DoubleVector.fromArray(F64_SPECIES, out, baseOut + i).mul(scaleVector).intoArray(out, baseOut + i);
        }
        for (; i < length; i++) {
            out[baseOut + i] *= scale;
        }
    }

    private static void scaleOutContiguousF32(float[] out, int baseOut, int length, float scale) {
        int upper = F32_SPECIES.loopBound(length);
        FloatVector scaleVector = FloatVector.broadcast(F32_SPECIES, scale);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            FloatVector.fromArray(F32_SPECIES, out, baseOut + i).mul(scaleVector).intoArray(out, baseOut + i);
        }
        for (; i < length; i++) {
            out[baseOut + i] *= scale;
        }
    }

    private record GroupState(
            int baseLogits,
            int targetOffset,
            int scaleOffset,
            int outOffset,
            int axisStrideIn,
            int axisSize,
            int axisStrideOut
    ) {}

    @FunctionalInterface
    private interface GroupComputer {
        void compute(int baseLogits, int targetOffset, int scaleOffset, int outOffset, int axisStride, int axisSize, int outAxisStride);
    }

    @FunctionalInterface
    private interface DenseGroupComputer {
        void compute(int group, int baseLogits, int targetOffset, int scaleOffset, int outOffset);
    }
}
