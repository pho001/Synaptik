package backend.kernels.cpu.linalg.matmul.common;

public final class MatMulBatchingSupport {
    private MatMulBatchingSupport() {
    }

    public static int batchCount(int[] outShape) {
        int count = 1;
        for (int i = 0; i < outShape.length - 2; i++) {
            count *= outShape[i];
        }
        return count;
    }

    public static int[] computeBatchOffsets(int[] inputShape, int[] outShape) {
        int inputBatchRank = inputShape.length - 2;
        int outBatchRank = outShape.length - 2;
        int[] inputDenseStrides = denseStrides(inputShape);
        int[] alignedBatchShape = new int[outBatchRank];
        int[] alignedBatchStrides = new int[outBatchRank];
        int shapeOffset = outBatchRank - inputBatchRank;
        for (int d = 0; d < outBatchRank; d++) {
            if (d < shapeOffset) {
                alignedBatchShape[d] = 1;
                alignedBatchStrides[d] = 0;
                continue;
            }
            int inputDim = inputShape[d - shapeOffset];
            alignedBatchShape[d] = inputDim;
            alignedBatchStrides[d] = inputDim == 1 ? 0 : inputDenseStrides[d - shapeOffset];
        }

        int batchCount = batchCount(outShape);
        int[] offsets = new int[batchCount];
        if (outBatchRank == 0) {
            return offsets;
        }
        int[] outBatchShape = java.util.Arrays.copyOf(outShape, outBatchRank);
        int[] outBatchDenseStrides = denseStrides(outBatchShape);
        for (int batch = 0; batch < batchCount; batch++) {
            int tmp = batch;
            int offset = 0;
            for (int d = 0; d < outBatchRank; d++) {
                int coord = tmp / outBatchDenseStrides[d];
                tmp %= outBatchDenseStrides[d];
                if (alignedBatchShape[d] != 1) {
                    offset += coord * alignedBatchStrides[d];
                }
            }
            offsets[batch] = offset;
        }
        return offsets;
    }

    public static int positiveTile(int value, int fallback) {
        return value > 0 ? value : fallback;
    }

    private static int[] denseStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }
}
