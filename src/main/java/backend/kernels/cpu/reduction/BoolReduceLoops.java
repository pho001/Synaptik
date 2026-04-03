package backend.kernels.cpu.reduction;

import tensor.Tensor;
import tensor.TensorMetadata;

final class BoolReduceLoops {
    private BoolReduceLoops() {}

    static void execute(Tensor input, Tensor node, int dimension, boolean isAll) {
        int[] shape = input.getShapeUnsafe();
        validateDimension(shape, dimension);

        byte[] in = input.getBoolData();
        byte[] out = node.getBoolData();
        if (in == null || out == null) {
            throw new IllegalStateException("BOOL storage is missing");
        }

        if (dimension == -1) {
            out[node.getStorageOffsetUnsafe()] = reduceAll(in, shape, input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), isAll);
            return;
        }
        reduceAxis(in, shape, input.getStridesUnsafe(), input.getStorageOffsetUnsafe(), out, node.getShapeUnsafe(), node.getStorageOffsetUnsafe(), dimension, isAll);
    }

    private static void reduceAxis(
            byte[] in,
            int[] inputShape,
            int[] inputStrides,
            int inputBaseOffset,
            byte[] out,
            int[] outShape,
            int outBaseOffset,
            int dimension,
            boolean isAll
    ) {
        int[] outDenseStrides = TensorMetadata.computeStrides(outShape);
        int reducedSize = inputShape[dimension];
        int reducedStride = inputStrides[dimension];
        for (int outIndex = 0; outIndex < nodeLogicalSize(outShape); outIndex++) {
            int baseOffset = reductionBaseOffset(outIndex, outShape, outDenseStrides, inputStrides, dimension, inputBaseOffset);
            boolean acc = in[baseOffset] != 0;
            for (int r = 1; r < reducedSize; r++) {
                boolean value = in[baseOffset + r * reducedStride] != 0;
                acc = isAll ? (acc && value) : (acc || value);
            }
            out[outBaseOffset + outIndex] = acc ? (byte) 1 : (byte) 0;
        }
    }

    private static byte reduceAll(byte[] in, int[] shape, int[] strides, int baseOffset, boolean isAll) {
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        int logicalSize = logicalSize(shape);
        boolean acc = in[logicalToOffset(0, shape, strides, denseStrides, baseOffset)] != 0;
        for (int logical = 1; logical < logicalSize; logical++) {
            boolean value = in[logicalToOffset(logical, shape, strides, denseStrides, baseOffset)] != 0;
            acc = isAll ? (acc && value) : (acc || value);
        }
        return acc ? (byte) 1 : (byte) 0;
    }

    private static int reductionBaseOffset(int outIndex, int[] outShape, int[] outDenseStrides, int[] inputStrides, int reducedDimension, int inputBaseOffset) {
        int idx = outIndex;
        int baseOffset = inputBaseOffset;
        int inputRank = inputStrides.length;
        if (outShape.length == inputRank) {
            for (int outDim = 0; outDim < outShape.length; outDim++) {
                int coord = idx / outDenseStrides[outDim];
                idx %= outDenseStrides[outDim];
                if (outDim == reducedDimension) {
                    continue;
                }
                baseOffset += coord * inputStrides[outDim];
            }
            return baseOffset;
        }
        for (int outDim = 0; outDim < outShape.length; outDim++) {
            int coord = idx / outDenseStrides[outDim];
            idx %= outDenseStrides[outDim];
            int inputDim = outDim < reducedDimension ? outDim : outDim + 1;
            baseOffset += coord * inputStrides[inputDim];
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

    private static int nodeLogicalSize(int[] shape) {
        return logicalSize(shape);
    }

    private static void validateDimension(int[] shape, int dimension) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Input shape must not be empty");
        }
        if (dimension < -1 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
    }
}
