package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuKernelContext;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tensor.DataType;
import tensor.Tensor;

final class SoftmaxGradExecutor {
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int MIN_VECTOR_AXIS_MULTIPLIER = 4;

    private SoftmaxGradExecutor() {}

    static void executeSoftmaxF64(Tensor softmaxOut, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        validate(softmaxOut, outGrad, node, dimension, context, "softmaxGrad");
        double[] primary = softmaxOut.getFloat64Data();
        double[] grad = outGrad.getFloat64Data();
        double[] out = node.getFloat64Data();
        SoftmaxGradTraversal.runGroups(
                softmaxOut.getShapeUnsafe(),
                softmaxOut.getStridesUnsafe(),
                softmaxOut.getStorageOffsetUnsafe(),
                outGrad.getStridesUnsafe(),
                outGrad.getStorageOffsetUnsafe(),
                node.getStridesUnsafe(),
                node.getStorageOffsetUnsafe(),
                dimension,
                context,
                (primaryBase, gradBase, outBase, primaryAxisStride, gradAxisStride, outAxisStride, axisSize) ->
                        computeSoftmaxF64(primary, grad, out, primaryBase, gradBase, outBase, primaryAxisStride, gradAxisStride, outAxisStride, axisSize)
        );
    }

    static void executeSoftmaxF32(Tensor softmaxOut, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        validate(softmaxOut, outGrad, node, dimension, context, "softmaxGrad");
        float[] primary = softmaxOut.getFloat32Data();
        float[] grad = outGrad.getFloat32Data();
        float[] out = node.getFloat32Data();
        SoftmaxGradTraversal.runGroups(
                softmaxOut.getShapeUnsafe(),
                softmaxOut.getStridesUnsafe(),
                softmaxOut.getStorageOffsetUnsafe(),
                outGrad.getStridesUnsafe(),
                outGrad.getStorageOffsetUnsafe(),
                node.getStridesUnsafe(),
                node.getStorageOffsetUnsafe(),
                dimension,
                context,
                (primaryBase, gradBase, outBase, primaryAxisStride, gradAxisStride, outAxisStride, axisSize) ->
                        computeSoftmaxF32(primary, grad, out, primaryBase, gradBase, outBase, primaryAxisStride, gradAxisStride, outAxisStride, axisSize)
        );
    }

    static void executeSoftmaxBF16(Tensor softmaxOut, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        validate(softmaxOut, outGrad, node, dimension, context, "softmaxGrad");
        short[] primary = softmaxOut.getBFloat16Data();
        short[] grad = outGrad.getBFloat16Data();
        short[] out = node.getBFloat16Data();
        SoftmaxGradTraversal.runGroups(
                softmaxOut.getShapeUnsafe(),
                softmaxOut.getStridesUnsafe(),
                softmaxOut.getStorageOffsetUnsafe(),
                outGrad.getStridesUnsafe(),
                outGrad.getStorageOffsetUnsafe(),
                node.getStridesUnsafe(),
                node.getStorageOffsetUnsafe(),
                dimension,
                context,
                (primaryBase, gradBase, outBase, primaryAxisStride, gradAxisStride, outAxisStride, axisSize) ->
                        computeSoftmaxBF16(primary, grad, out, primaryBase, gradBase, outBase, primaryAxisStride, gradAxisStride, outAxisStride, axisSize)
        );
    }

    static void executeLogSoftmaxF64(Tensor logSoftmaxOut, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        validate(logSoftmaxOut, outGrad, node, dimension, context, "logSoftmaxGrad");
        double[] primary = logSoftmaxOut.getFloat64Data();
        double[] grad = outGrad.getFloat64Data();
        double[] out = node.getFloat64Data();
        SoftmaxGradTraversal.runGroups(
                logSoftmaxOut.getShapeUnsafe(),
                logSoftmaxOut.getStridesUnsafe(),
                logSoftmaxOut.getStorageOffsetUnsafe(),
                outGrad.getStridesUnsafe(),
                outGrad.getStorageOffsetUnsafe(),
                node.getStridesUnsafe(),
                node.getStorageOffsetUnsafe(),
                dimension,
                context,
                (primaryBase, gradBase, outBase, primaryAxisStride, gradAxisStride, outAxisStride, axisSize) ->
                        computeLogSoftmaxF64(primary, grad, out, primaryBase, gradBase, outBase, primaryAxisStride, gradAxisStride, outAxisStride, axisSize)
        );
    }

    static void executeLogSoftmaxF32(Tensor logSoftmaxOut, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        validate(logSoftmaxOut, outGrad, node, dimension, context, "logSoftmaxGrad");
        float[] primary = logSoftmaxOut.getFloat32Data();
        float[] grad = outGrad.getFloat32Data();
        float[] out = node.getFloat32Data();
        SoftmaxGradTraversal.runGroups(
                logSoftmaxOut.getShapeUnsafe(),
                logSoftmaxOut.getStridesUnsafe(),
                logSoftmaxOut.getStorageOffsetUnsafe(),
                outGrad.getStridesUnsafe(),
                outGrad.getStorageOffsetUnsafe(),
                node.getStridesUnsafe(),
                node.getStorageOffsetUnsafe(),
                dimension,
                context,
                (primaryBase, gradBase, outBase, primaryAxisStride, gradAxisStride, outAxisStride, axisSize) ->
                        computeLogSoftmaxF32(primary, grad, out, primaryBase, gradBase, outBase, primaryAxisStride, gradAxisStride, outAxisStride, axisSize)
        );
    }

    static void executeLogSoftmaxBF16(Tensor logSoftmaxOut, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context) {
        validate(logSoftmaxOut, outGrad, node, dimension, context, "logSoftmaxGrad");
        short[] primary = logSoftmaxOut.getBFloat16Data();
        short[] grad = outGrad.getBFloat16Data();
        short[] out = node.getBFloat16Data();
        SoftmaxGradTraversal.runGroups(
                logSoftmaxOut.getShapeUnsafe(),
                logSoftmaxOut.getStridesUnsafe(),
                logSoftmaxOut.getStorageOffsetUnsafe(),
                outGrad.getStridesUnsafe(),
                outGrad.getStorageOffsetUnsafe(),
                node.getStridesUnsafe(),
                node.getStorageOffsetUnsafe(),
                dimension,
                context,
                (primaryBase, gradBase, outBase, primaryAxisStride, gradAxisStride, outAxisStride, axisSize) ->
                        computeLogSoftmaxBF16(primary, grad, out, primaryBase, gradBase, outBase, primaryAxisStride, gradAxisStride, outAxisStride, axisSize)
        );
    }

    private static void validate(Tensor primary, Tensor outGrad, Tensor node, int dimension, CpuKernelContext context, String label) {
        if (primary == null || outGrad == null || node == null || context == null) {
            throw new IllegalArgumentException(label + " execution arguments cannot be null");
        }
        if (primary.getDataType() == DataType.BOOL || primary.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException(label + " requires floating numeric input.");
        }
        if (primary.getDataType() != outGrad.getDataType() || primary.getDataType() != node.getDataType()) {
            throw new IllegalArgumentException(label + " requires matching floating dtypes.");
        }
        SoftmaxGradTraversal.validateShapes(primary.getShapeUnsafe(), outGrad.getShapeUnsafe(), node.getShapeUnsafe(), dimension, label);
    }

    private static void computeSoftmaxF64(double[] primary, double[] grad, double[] out, int primaryBase, int gradBase, int outBase,
                                          int primaryAxisStride, int gradAxisStride, int outAxisStride, int axisSize) {
        if (canUseContiguousVectorPath(primaryAxisStride, gradAxisStride, outAxisStride, axisSize, F64_SPECIES.length())) {
            double dot = dotProductContiguousF64(primary, grad, primaryBase, gradBase, axisSize);
            transformSoftmaxContiguousF64(primary, grad, out, primaryBase, gradBase, outBase, axisSize, dot);
            return;
        }
        double dot = 0.0d;
        for (int i = 0, primaryOffset = primaryBase, gradOffset = gradBase; i < axisSize; i++, primaryOffset += primaryAxisStride, gradOffset += gradAxisStride) {
            dot += primary[primaryOffset] * grad[gradOffset];
        }
        for (int i = 0, primaryOffset = primaryBase, gradOffset = gradBase, outOffset = outBase;
             i < axisSize;
             i++, primaryOffset += primaryAxisStride, gradOffset += gradAxisStride, outOffset += outAxisStride) {
            out[outOffset] = primary[primaryOffset] * (grad[gradOffset] - dot);
        }
    }

    private static void computeSoftmaxF32(float[] primary, float[] grad, float[] out, int primaryBase, int gradBase, int outBase,
                                          int primaryAxisStride, int gradAxisStride, int outAxisStride, int axisSize) {
        if (canUseContiguousVectorPath(primaryAxisStride, gradAxisStride, outAxisStride, axisSize, F32_SPECIES.length())) {
            float dot = dotProductContiguousF32(primary, grad, primaryBase, gradBase, axisSize);
            transformSoftmaxContiguousF32(primary, grad, out, primaryBase, gradBase, outBase, axisSize, dot);
            return;
        }
        float dot = 0.0f;
        for (int i = 0, primaryOffset = primaryBase, gradOffset = gradBase; i < axisSize; i++, primaryOffset += primaryAxisStride, gradOffset += gradAxisStride) {
            dot += primary[primaryOffset] * grad[gradOffset];
        }
        for (int i = 0, primaryOffset = primaryBase, gradOffset = gradBase, outOffset = outBase;
             i < axisSize;
             i++, primaryOffset += primaryAxisStride, gradOffset += gradAxisStride, outOffset += outAxisStride) {
            out[outOffset] = primary[primaryOffset] * (grad[gradOffset] - dot);
        }
    }

    private static void computeSoftmaxBF16(short[] primary, short[] grad, short[] out, int primaryBase, int gradBase, int outBase,
                                           int primaryAxisStride, int gradAxisStride, int outAxisStride, int axisSize) {
        float dot = 0.0f;
        for (int i = 0, primaryOffset = primaryBase, gradOffset = gradBase; i < axisSize; i++, primaryOffset += primaryAxisStride, gradOffset += gradAxisStride) {
            dot += CpuDTypeOps.fromBFloat16Bits(primary[primaryOffset]) * CpuDTypeOps.fromBFloat16Bits(grad[gradOffset]);
        }
        for (int i = 0, primaryOffset = primaryBase, gradOffset = gradBase, outOffset = outBase;
             i < axisSize;
             i++, primaryOffset += primaryAxisStride, gradOffset += gradAxisStride, outOffset += outAxisStride) {
            float value = CpuDTypeOps.fromBFloat16Bits(primary[primaryOffset]) * (CpuDTypeOps.fromBFloat16Bits(grad[gradOffset]) - dot);
            out[outOffset] = CpuDTypeOps.toBFloat16Bits(value);
        }
    }

    private static void computeLogSoftmaxF64(double[] primary, double[] grad, double[] out, int primaryBase, int gradBase, int outBase,
                                             int primaryAxisStride, int gradAxisStride, int outAxisStride, int axisSize) {
        if (canUseContiguousVectorPath(primaryAxisStride, gradAxisStride, outAxisStride, axisSize, F64_SPECIES.length())) {
            double sumGrad = sumContiguousF64(grad, gradBase, axisSize);
            transformLogSoftmaxContiguousF64(primary, grad, out, primaryBase, gradBase, outBase, axisSize, sumGrad);
            return;
        }
        double sumGrad = 0.0d;
        for (int i = 0, gradOffset = gradBase; i < axisSize; i++, gradOffset += gradAxisStride) {
            sumGrad += grad[gradOffset];
        }
        for (int i = 0, primaryOffset = primaryBase, gradOffset = gradBase, outOffset = outBase;
             i < axisSize;
             i++, primaryOffset += primaryAxisStride, gradOffset += gradAxisStride, outOffset += outAxisStride) {
            out[outOffset] = grad[gradOffset] - Math.exp(primary[primaryOffset]) * sumGrad;
        }
    }

    private static void computeLogSoftmaxF32(float[] primary, float[] grad, float[] out, int primaryBase, int gradBase, int outBase,
                                             int primaryAxisStride, int gradAxisStride, int outAxisStride, int axisSize) {
        if (canUseContiguousVectorPath(primaryAxisStride, gradAxisStride, outAxisStride, axisSize, F32_SPECIES.length())) {
            float sumGrad = sumContiguousF32(grad, gradBase, axisSize);
            transformLogSoftmaxContiguousF32(primary, grad, out, primaryBase, gradBase, outBase, axisSize, sumGrad);
            return;
        }
        float sumGrad = 0.0f;
        for (int i = 0, gradOffset = gradBase; i < axisSize; i++, gradOffset += gradAxisStride) {
            sumGrad += grad[gradOffset];
        }
        for (int i = 0, primaryOffset = primaryBase, gradOffset = gradBase, outOffset = outBase;
             i < axisSize;
             i++, primaryOffset += primaryAxisStride, gradOffset += gradAxisStride, outOffset += outAxisStride) {
            out[outOffset] = grad[gradOffset] - (float) Math.exp(primary[primaryOffset]) * sumGrad;
        }
    }

    private static void computeLogSoftmaxBF16(short[] primary, short[] grad, short[] out, int primaryBase, int gradBase, int outBase,
                                              int primaryAxisStride, int gradAxisStride, int outAxisStride, int axisSize) {
        float sumGrad = 0.0f;
        for (int i = 0, gradOffset = gradBase; i < axisSize; i++, gradOffset += gradAxisStride) {
            sumGrad += CpuDTypeOps.fromBFloat16Bits(grad[gradOffset]);
        }
        for (int i = 0, primaryOffset = primaryBase, gradOffset = gradBase, outOffset = outBase;
             i < axisSize;
             i++, primaryOffset += primaryAxisStride, gradOffset += gradAxisStride, outOffset += outAxisStride) {
            float value = CpuDTypeOps.fromBFloat16Bits(grad[gradOffset])
                    - (float) Math.exp(CpuDTypeOps.fromBFloat16Bits(primary[primaryOffset])) * sumGrad;
            out[outOffset] = CpuDTypeOps.toBFloat16Bits(value);
        }
    }

    private static boolean canUseContiguousVectorPath(int primaryAxisStride, int gradAxisStride, int outAxisStride, int axisSize, int speciesLength) {
        return primaryAxisStride == 1
                && gradAxisStride == 1
                && outAxisStride == 1
                && speciesLength > 1
                && axisSize >= speciesLength * MIN_VECTOR_AXIS_MULTIPLIER;
    }

    private static double dotProductContiguousF64(double[] left, double[] right, int leftBase, int rightBase, int length) {
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector sumVector = DoubleVector.zero(F64_SPECIES);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            sumVector = sumVector.add(
                    DoubleVector.fromArray(F64_SPECIES, left, leftBase + i)
                            .mul(DoubleVector.fromArray(F64_SPECIES, right, rightBase + i))
            );
        }
        double sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            sum += left[leftBase + i] * right[rightBase + i];
        }
        return sum;
    }

    private static float dotProductContiguousF32(float[] left, float[] right, int leftBase, int rightBase, int length) {
        int upper = F32_SPECIES.loopBound(length);
        FloatVector sumVector = FloatVector.zero(F32_SPECIES);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            sumVector = sumVector.add(
                    FloatVector.fromArray(F32_SPECIES, left, leftBase + i)
                            .mul(FloatVector.fromArray(F32_SPECIES, right, rightBase + i))
            );
        }
        float sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            sum += left[leftBase + i] * right[rightBase + i];
        }
        return sum;
    }

    private static double sumContiguousF64(double[] values, int base, int length) {
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector sumVector = DoubleVector.zero(F64_SPECIES);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            sumVector = sumVector.add(DoubleVector.fromArray(F64_SPECIES, values, base + i));
        }
        double sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            sum += values[base + i];
        }
        return sum;
    }

    private static float sumContiguousF32(float[] values, int base, int length) {
        int upper = F32_SPECIES.loopBound(length);
        FloatVector sumVector = FloatVector.zero(F32_SPECIES);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            sumVector = sumVector.add(FloatVector.fromArray(F32_SPECIES, values, base + i));
        }
        float sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            sum += values[base + i];
        }
        return sum;
    }

    private static void transformSoftmaxContiguousF64(double[] primary, double[] grad, double[] out, int primaryBase, int gradBase, int outBase, int length, double dot) {
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector dotVector = DoubleVector.broadcast(F64_SPECIES, dot);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            DoubleVector primaryVector = DoubleVector.fromArray(F64_SPECIES, primary, primaryBase + i);
            DoubleVector gradVector = DoubleVector.fromArray(F64_SPECIES, grad, gradBase + i);
            primaryVector.mul(gradVector.sub(dotVector)).intoArray(out, outBase + i);
        }
        for (; i < length; i++) {
            out[outBase + i] = primary[primaryBase + i] * (grad[gradBase + i] - dot);
        }
    }

    private static void transformSoftmaxContiguousF32(float[] primary, float[] grad, float[] out, int primaryBase, int gradBase, int outBase, int length, float dot) {
        int upper = F32_SPECIES.loopBound(length);
        FloatVector dotVector = FloatVector.broadcast(F32_SPECIES, dot);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            FloatVector primaryVector = FloatVector.fromArray(F32_SPECIES, primary, primaryBase + i);
            FloatVector gradVector = FloatVector.fromArray(F32_SPECIES, grad, gradBase + i);
            primaryVector.mul(gradVector.sub(dotVector)).intoArray(out, outBase + i);
        }
        for (; i < length; i++) {
            out[outBase + i] = primary[primaryBase + i] * (grad[gradBase + i] - dot);
        }
    }

    private static void transformLogSoftmaxContiguousF64(double[] primary, double[] grad, double[] out, int primaryBase, int gradBase, int outBase, int length, double sumGrad) {
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector sumVector = DoubleVector.broadcast(F64_SPECIES, sumGrad);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            DoubleVector primaryVector = DoubleVector.fromArray(F64_SPECIES, primary, primaryBase + i);
            DoubleVector gradVector = DoubleVector.fromArray(F64_SPECIES, grad, gradBase + i);
            gradVector.sub(primaryVector.lanewise(VectorOperators.EXP).mul(sumVector)).intoArray(out, outBase + i);
        }
        for (; i < length; i++) {
            out[outBase + i] = grad[gradBase + i] - Math.exp(primary[primaryBase + i]) * sumGrad;
        }
    }

    private static void transformLogSoftmaxContiguousF32(float[] primary, float[] grad, float[] out, int primaryBase, int gradBase, int outBase, int length, float sumGrad) {
        int upper = F32_SPECIES.loopBound(length);
        FloatVector sumVector = FloatVector.broadcast(F32_SPECIES, sumGrad);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            FloatVector primaryVector = FloatVector.fromArray(F32_SPECIES, primary, primaryBase + i);
            FloatVector gradVector = FloatVector.fromArray(F32_SPECIES, grad, gradBase + i);
            gradVector.sub(primaryVector.lanewise(VectorOperators.EXP).mul(sumVector)).intoArray(out, outBase + i);
        }
        for (; i < length; i++) {
            out[outBase + i] = grad[gradBase + i] - (float) Math.exp(primary[primaryBase + i]) * sumGrad;
        }
    }
}
