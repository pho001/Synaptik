package backend.kernels.cpu;

import tensor.TensorMetadata;

final class ReductionMinMaxGradKernelSupport {
    private ReductionMinMaxGradKernelSupport() {}

    static void runF64(double[] input, int[] inputShape, int[] inputStrides, int inputBaseOffset, double[] reduced, int[] reducedShape, int reducedBaseOffset, double[] outGrad, double[] out, int outBaseOffset, int dimension, boolean isMax) {
        if (dimension == -1) {
            runAllF64(input, inputShape, inputStrides, inputBaseOffset, reduced[reducedBaseOffset], outGrad[outBaseOffset], out, outBaseOffset, isMax);
            return;
        }
        runAxisF64(input, inputShape, inputStrides, inputBaseOffset, reduced, reducedShape, reducedBaseOffset, outGrad, out, outBaseOffset, dimension, isMax);
    }

    static void runF32(float[] input, int[] inputShape, int[] inputStrides, int inputBaseOffset, float[] reduced, int[] reducedShape, int reducedBaseOffset, float[] outGrad, float[] out, int outBaseOffset, int dimension, boolean isMax) {
        if (dimension == -1) {
            runAllF32(input, inputShape, inputStrides, inputBaseOffset, reduced[reducedBaseOffset], outGrad[outBaseOffset], out, outBaseOffset, isMax);
            return;
        }
        runAxisF32(input, inputShape, inputStrides, inputBaseOffset, reduced, reducedShape, reducedBaseOffset, outGrad, out, outBaseOffset, dimension, isMax);
    }

    static void runF16(short[] input, int[] inputShape, int[] inputStrides, int inputBaseOffset, short[] reduced, int[] reducedShape, int reducedBaseOffset, short[] outGrad, short[] out, int outBaseOffset, int dimension, boolean isMax) {
        if (dimension == -1) {
            runAllF16(input, inputShape, inputStrides, inputBaseOffset, reduced[reducedBaseOffset], outGrad[outBaseOffset], out, outBaseOffset, isMax);
            return;
        }
        runAxisF16(input, inputShape, inputStrides, inputBaseOffset, reduced, reducedShape, reducedBaseOffset, outGrad, out, outBaseOffset, dimension, isMax);
    }

    private static void runAllF64(double[] input, int[] inputShape, int[] inputStrides, int inputBaseOffset, double reducedValue, double gradValue, double[] out, int outBaseOffset, boolean isMax) {
        int logicalSize = logicalSize(inputShape);
        int[] denseStrides = TensorMetadata.computeStrides(inputShape);
        int winners = 0;
        for (int logical = 0; logical < logicalSize; logical++) {
            double value = input[logicalToOffset(logical, inputShape, inputStrides, denseStrides, inputBaseOffset)];
            if (matches(value, reducedValue, isMax)) {
                winners++;
            }
        }
        double share = winners == 0 ? 0.0d : gradValue / winners;
        for (int logical = 0; logical < logicalSize; logical++) {
            double value = input[logicalToOffset(logical, inputShape, inputStrides, denseStrides, inputBaseOffset)];
            out[outBaseOffset + logical] = matches(value, reducedValue, isMax) ? share : 0.0d;
        }
    }

    private static void runAllF32(float[] input, int[] inputShape, int[] inputStrides, int inputBaseOffset, float reducedValue, float gradValue, float[] out, int outBaseOffset, boolean isMax) {
        int logicalSize = logicalSize(inputShape);
        int[] denseStrides = TensorMetadata.computeStrides(inputShape);
        int winners = 0;
        for (int logical = 0; logical < logicalSize; logical++) {
            float value = input[logicalToOffset(logical, inputShape, inputStrides, denseStrides, inputBaseOffset)];
            if (matches(value, reducedValue, isMax)) {
                winners++;
            }
        }
        float share = winners == 0 ? 0.0f : gradValue / winners;
        for (int logical = 0; logical < logicalSize; logical++) {
            float value = input[logicalToOffset(logical, inputShape, inputStrides, denseStrides, inputBaseOffset)];
            out[outBaseOffset + logical] = matches(value, reducedValue, isMax) ? share : 0.0f;
        }
    }

    private static void runAllF16(short[] input, int[] inputShape, int[] inputStrides, int inputBaseOffset, short reducedValueBits, short gradValueBits, short[] out, int outBaseOffset, boolean isMax) {
        float reducedValue = CpuDTypeOps.fromHalfBits(reducedValueBits);
        float gradValue = CpuDTypeOps.fromHalfBits(gradValueBits);
        int logicalSize = logicalSize(inputShape);
        int[] denseStrides = TensorMetadata.computeStrides(inputShape);
        int winners = 0;
        for (int logical = 0; logical < logicalSize; logical++) {
            float value = CpuDTypeOps.fromHalfBits(input[logicalToOffset(logical, inputShape, inputStrides, denseStrides, inputBaseOffset)]);
            if (matches(value, reducedValue, isMax)) {
                winners++;
            }
        }
        float share = winners == 0 ? 0.0f : gradValue / winners;
        for (int logical = 0; logical < logicalSize; logical++) {
            float value = CpuDTypeOps.fromHalfBits(input[logicalToOffset(logical, inputShape, inputStrides, denseStrides, inputBaseOffset)]);
            out[outBaseOffset + logical] = CpuDTypeOps.toHalfBits(matches(value, reducedValue, isMax) ? share : 0.0f);
        }
    }

    private static void runAxisF64(double[] input, int[] inputShape, int[] inputStrides, int inputBaseOffset, double[] reduced, int[] reducedShape, int reducedBaseOffset, double[] outGrad, double[] out, int outBaseOffset, int dimension, boolean isMax) {
        int[] reducedDense = TensorMetadata.computeStrides(reducedShape);
        int[] outDense = TensorMetadata.computeStrides(inputShape);
        int groups = reduced.length;
        int reducedSize = inputShape[dimension];
        int inStep = inputStrides[dimension];
        int outStep = outDense[dimension];

        for (int group = 0; group < groups; group++) {
            int baseInput = reductionBaseOffset(group, reducedShape, reducedDense, inputStrides, dimension, inputBaseOffset);
            int baseOut = reductionBaseOffset(group, reducedShape, reducedDense, outDense, dimension, outBaseOffset);
            double reducedValue = reduced[reducedBaseOffset + group];
            int winners = 0;
            for (int r = 0; r < reducedSize; r++) {
                if (matches(input[baseInput + r * inStep], reducedValue, isMax)) {
                    winners++;
                }
            }
            double share = winners == 0 ? 0.0d : outGrad[outBaseOffset + group] / winners;
            for (int r = 0; r < reducedSize; r++) {
                double value = input[baseInput + r * inStep];
                out[baseOut + r * outStep] = matches(value, reducedValue, isMax) ? share : 0.0d;
            }
        }
    }

    private static void runAxisF32(float[] input, int[] inputShape, int[] inputStrides, int inputBaseOffset, float[] reduced, int[] reducedShape, int reducedBaseOffset, float[] outGrad, float[] out, int outBaseOffset, int dimension, boolean isMax) {
        int[] reducedDense = TensorMetadata.computeStrides(reducedShape);
        int[] outDense = TensorMetadata.computeStrides(inputShape);
        int groups = reduced.length;
        int reducedSize = inputShape[dimension];
        int inStep = inputStrides[dimension];
        int outStep = outDense[dimension];

        for (int group = 0; group < groups; group++) {
            int baseInput = reductionBaseOffset(group, reducedShape, reducedDense, inputStrides, dimension, inputBaseOffset);
            int baseOut = reductionBaseOffset(group, reducedShape, reducedDense, outDense, dimension, outBaseOffset);
            float reducedValue = reduced[reducedBaseOffset + group];
            int winners = 0;
            for (int r = 0; r < reducedSize; r++) {
                if (matches(input[baseInput + r * inStep], reducedValue, isMax)) {
                    winners++;
                }
            }
            float share = winners == 0 ? 0.0f : outGrad[outBaseOffset + group] / winners;
            for (int r = 0; r < reducedSize; r++) {
                float value = input[baseInput + r * inStep];
                out[baseOut + r * outStep] = matches(value, reducedValue, isMax) ? share : 0.0f;
            }
        }
    }

    private static void runAxisF16(short[] input, int[] inputShape, int[] inputStrides, int inputBaseOffset, short[] reduced, int[] reducedShape, int reducedBaseOffset, short[] outGrad, short[] out, int outBaseOffset, int dimension, boolean isMax) {
        int[] reducedDense = TensorMetadata.computeStrides(reducedShape);
        int[] outDense = TensorMetadata.computeStrides(inputShape);
        int groups = reduced.length;
        int reducedSize = inputShape[dimension];
        int inStep = inputStrides[dimension];
        int outStep = outDense[dimension];

        for (int group = 0; group < groups; group++) {
            int baseInput = reductionBaseOffset(group, reducedShape, reducedDense, inputStrides, dimension, inputBaseOffset);
            int baseOut = reductionBaseOffset(group, reducedShape, reducedDense, outDense, dimension, outBaseOffset);
            float reducedValue = CpuDTypeOps.fromHalfBits(reduced[reducedBaseOffset + group]);
            int winners = 0;
            for (int r = 0; r < reducedSize; r++) {
                if (matches(CpuDTypeOps.fromHalfBits(input[baseInput + r * inStep]), reducedValue, isMax)) {
                    winners++;
                }
            }
            float share = winners == 0 ? 0.0f : CpuDTypeOps.fromHalfBits(outGrad[outBaseOffset + group]) / winners;
            for (int r = 0; r < reducedSize; r++) {
                float value = CpuDTypeOps.fromHalfBits(input[baseInput + r * inStep]);
                out[baseOut + r * outStep] = CpuDTypeOps.toHalfBits(matches(value, reducedValue, isMax) ? share : 0.0f);
            }
        }
    }

    private static boolean matches(double value, double reducedValue, boolean isMax) {
        return isMax ? value == reducedValue : value == reducedValue;
    }

    private static int reductionBaseOffset(int outIndex, int[] outShape, int[] outDenseStrides, int[] targetStrides, int reducedDimension, int targetBaseOffset) {
        int idx = outIndex;
        int baseOffset = targetBaseOffset;
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

    private static int logicalToOffset(int logicalIndex, int[] shape, int[] strides, int[] denseStrides, int baseOffset) {
        int idx = logicalIndex;
        int offset = baseOffset;
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
