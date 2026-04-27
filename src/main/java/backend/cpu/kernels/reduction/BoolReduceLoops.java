package backend.cpu.kernels.reduction;

import backend.cpu.kernels.*;

import tensor.Tensor;
import tensor.TensorMetadata;

final class BoolReduceLoops {
    private BoolReduceLoops() {}

    static void execute(Tensor input, Tensor node, int dimension, boolean isAll) {
        int[] shape = input.getShapeUnsafe();
        ReductionTraversal.validateDimension(shape, dimension);

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
        ReductionTraversal.forEachAxisGroup(inputShape, inputStrides, inputBaseOffset, outShape, dimension, (outIndex, baseOffset, reducedSize, reducedStride) -> {
            boolean acc = in[baseOffset] != 0;
            for (int r = 1; r < reducedSize; r++) {
                boolean value = in[baseOffset + r * reducedStride] != 0;
                acc = isAll ? (acc && value) : (acc || value);
            }
            out[outBaseOffset + outIndex] = acc ? (byte) 1 : (byte) 0;
        });
    }

    private static byte reduceAll(byte[] in, int[] shape, int[] strides, int baseOffset, boolean isAll) {
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        int logicalSize = ReductionTraversal.logicalSize(shape);
        boolean acc = in[ReductionTraversal.logicalToOffset(0, shape, strides, denseStrides, baseOffset)] != 0;
        for (int logical = 1; logical < logicalSize; logical++) {
            boolean value = in[ReductionTraversal.logicalToOffset(logical, shape, strides, denseStrides, baseOffset)] != 0;
            acc = isAll ? (acc && value) : (acc || value);
        }
        return acc ? (byte) 1 : (byte) 0;
    }
}
