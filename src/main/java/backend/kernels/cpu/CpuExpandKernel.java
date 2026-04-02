package backend.kernels.cpu;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuExpandKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        forward(inputs, node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        forward(inputs, node);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        forward(inputs, node);
    }

    private static void forward(List<Tensor> inputs, Tensor node) {
        if (inputs == null || inputs.isEmpty()) {
            return;
        }
        Tensor src = inputs.getFirst();
        switch (node.getDataType()) {
            case FLOAT64 -> expandF64(src, node);
            case FLOAT32 -> expandF32(src, node);
            case FLOAT16 -> expandF16(src, node);
        }
    }

    private static void expandF64(Tensor src, Tensor dst) {
        double[] in = src.getFloat64Data();
        double[] out = dst.getFloat64Data();
        copyExpanded(src, dst, (srcOffset, dstOffset) -> out[dstOffset] = in[srcOffset]);
    }

    private static void expandF32(Tensor src, Tensor dst) {
        float[] in = src.getFloat32Data();
        float[] out = dst.getFloat32Data();
        copyExpanded(src, dst, (srcOffset, dstOffset) -> out[dstOffset] = in[srcOffset]);
    }

    private static void expandF16(Tensor src, Tensor dst) {
        short[] in = src.getFloat16Data();
        short[] out = dst.getFloat16Data();
        copyExpanded(src, dst, (srcOffset, dstOffset) -> out[dstOffset] = in[srcOffset]);
    }

    private static void copyExpanded(Tensor src, Tensor dst, OffsetCopy copy) {
        int[] srcShape = src.getShapeUnsafe();
        int[] srcStrides = src.getStridesUnsafe();
        int[] dstShape = dst.getShapeUnsafe();
        int[] dstDenseStrides = denseStrides(dstShape);
        int dstRank = dstShape.length;
        int srcRank = srcShape.length;
        int rankOffset = dstRank - srcRank;
        int size = dst.getFlatDataSize();

        for (int logicalIndex = 0; logicalIndex < size; logicalIndex++) {
            int rem = logicalIndex;
            int srcOffset = 0;
            for (int d = 0; d < dstRank; d++) {
                int coord = rem / dstDenseStrides[d];
                rem %= dstDenseStrides[d];

                int srcDimIndex = d - rankOffset;
                if (srcDimIndex < 0) {
                    continue;
                }
                int srcDim = srcShape[srcDimIndex];
                if (srcDim != 1) {
                    srcOffset += coord * srcStrides[srcDimIndex];
                }
            }
            copy.copy(srcOffset, logicalIndex);
        }
    }

    private static int[] denseStrides(int[] shape) {
        int[] out = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            out[i] = stride;
            stride *= shape[i];
        }
        return out;
    }

    @FunctionalInterface
    private interface OffsetCopy {
        void copy(int srcOffset, int dstOffset);
    }
}
