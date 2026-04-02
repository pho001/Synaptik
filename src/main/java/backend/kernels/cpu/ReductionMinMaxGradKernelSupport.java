package backend.kernels.cpu;

import tensor.TensorMetadata;

final class ReductionMinMaxGradKernelSupport {
    private ReductionMinMaxGradKernelSupport() {}

    static void runF64(double[] input, int[] inputShape, int[] inputStrides, double[] reduced, int[] reducedShape, double[] outGrad, double[] out, int dimension, boolean isMax) {
        if (dimension == -1) {
            runAllF64(input, inputShape, inputStrides, reduced[0], outGrad[0], out, isMax);
            return;
        }
        runAxisF64(input, inputShape, inputStrides, reduced, reducedShape, outGrad, out, dimension, isMax);
    }

    static void runF32(float[] input, int[] inputShape, int[] inputStrides, float[] reduced, int[] reducedShape, float[] outGrad, float[] out, int dimension, boolean isMax) {
        if (dimension == -1) {
            runAllF32(input, inputShape, inputStrides, reduced[0], outGrad[0], out, isMax);
            return;
        }
        runAxisF32(input, inputShape, inputStrides, reduced, reducedShape, outGrad, out, dimension, isMax);
    }

    static void runF16(short[] input, int[] inputShape, int[] inputStrides, short[] reduced, int[] reducedShape, short[] outGrad, short[] out, int dimension, boolean isMax) {
        if (dimension == -1) {
            runAllF16(input, inputShape, inputStrides, reduced[0], outGrad[0], out, isMax);
            return;
        }
        runAxisF16(input, inputShape, inputStrides, reduced, reducedShape, outGrad, out, dimension, isMax);
    }

    private static void runAllF64(double[] input, int[] inputShape, int[] inputStrides, double reducedValue, double gradValue, double[] out, boolean isMax) {
        int logicalSize = logicalSize(inputShape);
        int[] denseStrides = TensorMetadata.computeStrides(inputShape);
        int winners = 0;
        for (int logical = 0; logical < logicalSize; logical++) {
            double value = input[logicalToOffset(logical, inputShape, inputStrides, denseStrides)];
            if (matches(value, reducedValue, isMax)) {
                winners++;
            }
        }
        double share = winners == 0 ? 0.0d : gradValue / winners;
        for (int logical = 0; logical < logicalSize; logical++) {
            double value = input[logicalToOffset(logical, inputShape, inputStrides, denseStrides)];
            out[logical] = matches(value, reducedValue, isMax) ? share : 0.0d;
        }
    }

    private static void runAllF32(float[] input, int[] inputShape, int[] inputStrides, float reducedValue, float gradValue, float[] out, boolean isMax) {
        int logicalSize = logicalSize(inputShape);
        int[] denseStrides = TensorMetadata.computeStrides(inputShape);
        int winners = 0;
        for (int logical = 0; logical < logicalSize; logical++) {
            float value = input[logicalToOffset(logical, inputShape, inputStrides, denseStrides)];
            if (matches(value, reducedValue, isMax)) {
                winners++;
            }
        }
        float share = winners == 0 ? 0.0f : gradValue / winners;
        for (int logical = 0; logical < logicalSize; logical++) {
            float value = input[logicalToOffset(logical, inputShape, inputStrides, denseStrides)];
            out[logical] = matches(value, reducedValue, isMax) ? share : 0.0f;
        }
    }

    private static void runAllF16(short[] input, int[] inputShape, int[] inputStrides, short reducedValueBits, short gradValueBits, short[] out, boolean isMax) {
        float reducedValue = CpuDTypeOps.fromHalfBits(reducedValueBits);
        float gradValue = CpuDTypeOps.fromHalfBits(gradValueBits);
        int logicalSize = logicalSize(inputShape);
        int[] denseStrides = TensorMetadata.computeStrides(inputShape);
        int winners = 0;
        for (int logical = 0; logical < logicalSize; logical++) {
            float value = CpuDTypeOps.fromHalfBits(input[logicalToOffset(logical, inputShape, inputStrides, denseStrides)]);
            if (matches(value, reducedValue, isMax)) {
                winners++;
            }
        }
        float share = winners == 0 ? 0.0f : gradValue / winners;
        for (int logical = 0; logical < logicalSize; logical++) {
            float value = CpuDTypeOps.fromHalfBits(input[logicalToOffset(logical, inputShape, inputStrides, denseStrides)]);
            out[logical] = CpuDTypeOps.toHalfBits(matches(value, reducedValue, isMax) ? share : 0.0f);
        }
    }

    private static void runAxisF64(double[] input, int[] inputShape, int[] inputStrides, double[] reduced, int[] reducedShape, double[] outGrad, double[] out, int dimension, boolean isMax) {
        int[] reducedDense = TensorMetadata.computeStrides(reducedShape);
        int[] outDense = TensorMetadata.computeStrides(inputShape);
        int groups = reduced.length;
        int reducedSize = inputShape[dimension];
        int inStep = inputStrides[dimension];
        int outStep = outDense[dimension];

        for (int group = 0; group < groups; group++) {
            int baseInput = reductionBaseOffset(group, reducedShape, reducedDense, inputStrides, dimension);
            int baseOut = reductionBaseOffset(group, reducedShape, reducedDense, outDense, dimension);
            double reducedValue = reduced[group];
            int winners = 0;
            for (int r = 0; r < reducedSize; r++) {
                if (matches(input[baseInput + r * inStep], reducedValue, isMax)) {
                    winners++;
                }
            }
            double share = winners == 0 ? 0.0d : outGrad[group] / winners;
            for (int r = 0; r < reducedSize; r++) {
                double value = input[baseInput + r * inStep];
                out[baseOut + r * outStep] = matches(value, reducedValue, isMax) ? share : 0.0d;
            }
        }
    }

    private static void runAxisF32(float[] input, int[] inputShape, int[] inputStrides, float[] reduced, int[] reducedShape, float[] outGrad, float[] out, int dimension, boolean isMax) {
        int[] reducedDense = TensorMetadata.computeStrides(reducedShape);
        int[] outDense = TensorMetadata.computeStrides(inputShape);
        int groups = reduced.length;
        int reducedSize = inputShape[dimension];
        int inStep = inputStrides[dimension];
        int outStep = outDense[dimension];

        for (int group = 0; group < groups; group++) {
            int baseInput = reductionBaseOffset(group, reducedShape, reducedDense, inputStrides, dimension);
            int baseOut = reductionBaseOffset(group, reducedShape, reducedDense, outDense, dimension);
            float reducedValue = reduced[group];
            int winners = 0;
            for (int r = 0; r < reducedSize; r++) {
                if (matches(input[baseInput + r * inStep], reducedValue, isMax)) {
                    winners++;
                }
            }
            float share = winners == 0 ? 0.0f : outGrad[group] / winners;
            for (int r = 0; r < reducedSize; r++) {
                float value = input[baseInput + r * inStep];
                out[baseOut + r * outStep] = matches(value, reducedValue, isMax) ? share : 0.0f;
            }
        }
    }

    private static void runAxisF16(short[] input, int[] inputShape, int[] inputStrides, short[] reduced, int[] reducedShape, short[] outGrad, short[] out, int dimension, boolean isMax) {
        int[] reducedDense = TensorMetadata.computeStrides(reducedShape);
        int[] outDense = TensorMetadata.computeStrides(inputShape);
        int groups = reduced.length;
        int reducedSize = inputShape[dimension];
        int inStep = inputStrides[dimension];
        int outStep = outDense[dimension];

        for (int group = 0; group < groups; group++) {
            int baseInput = reductionBaseOffset(group, reducedShape, reducedDense, inputStrides, dimension);
            int baseOut = reductionBaseOffset(group, reducedShape, reducedDense, outDense, dimension);
            float reducedValue = CpuDTypeOps.fromHalfBits(reduced[group]);
            int winners = 0;
            for (int r = 0; r < reducedSize; r++) {
                if (matches(CpuDTypeOps.fromHalfBits(input[baseInput + r * inStep]), reducedValue, isMax)) {
                    winners++;
                }
            }
            float share = winners == 0 ? 0.0f : CpuDTypeOps.fromHalfBits(outGrad[group]) / winners;
            for (int r = 0; r < reducedSize; r++) {
                float value = CpuDTypeOps.fromHalfBits(input[baseInput + r * inStep]);
                out[baseOut + r * outStep] = CpuDTypeOps.toHalfBits(matches(value, reducedValue, isMax) ? share : 0.0f);
            }
        }
    }

    private static boolean matches(double value, double reducedValue, boolean isMax) {
        return isMax ? value == reducedValue : value == reducedValue;
    }

    private static int reductionBaseOffset(int outIndex, int[] outShape, int[] outDenseStrides, int[] targetStrides, int reducedDimension) {
        int idx = outIndex;
        int baseOffset = 0;
        int targetRank = targetStrides.length;
        if (outShape.length == targetRank) {
            for (int outDim = 0; outDim < outShape.length; outDim++) {
                int coord = idx / outDenseStrides[outDim];
                idx %= outDenseStrides[outDim];
                if (outDim == reducedDimension) {
                    continue;
                }
                baseOffset += coord * targetStrides[outDim];
            }
            return baseOffset;
        }
        for (int outDim = 0; outDim < outShape.length; outDim++) {
            int coord = idx / outDenseStrides[outDim];
            idx %= outDenseStrides[outDim];
            int targetDim = outDim < reducedDimension ? outDim : outDim + 1;
            baseOffset += coord * targetStrides[targetDim];
        }
        return baseOffset;
    }

    private static int logicalToOffset(int logicalIndex, int[] shape, int[] strides, int[] denseStrides) {
        int idx = logicalIndex;
        int offset = 0;
        for (int d = 0; d < shape.length; d++) {
            int coord = idx / denseStrides[d];
            idx %= denseStrides[d];
            offset += coord * strides[d];
        }
        return offset;
    }

    private static int logicalSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }
}
