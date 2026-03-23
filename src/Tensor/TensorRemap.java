package Tensor;

import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public final class TensorRemap {
    private TensorRemap() {}

    public static void apply(Tensor src, Tensor dst, int parallelThreshold) {
        if (src.getDataType() == DataType.FLOAT32
                && src.getFloat32Data() != null
                && dst.getFloat32Data() != null) {
            applyF32(src, dst, parallelThreshold);
            return;
        }
        if (src.getDataType() == DataType.FLOAT16
                && src.getFloat16Data() != null
                && dst.getFloat16Data() != null) {
            applyF16(src, dst, parallelThreshold);
            return;
        }
        int logicalSize = logicalSize(src.getShape());
        if (logicalSize > parallelThreshold) {
            parallelApply(src, dst);
        } else {
            sequentialApply(src, dst);
        }
    }

    private static void sequentialApply(Tensor src, Tensor dst) {
        int[] srcShape = src.getShape();
        int[] dstShape = dst.getShape();
        int[] srcStrides = src.getStrides();
        int[] dstStrides = dst.getStrides();

        if (!Arrays.equals(srcShape, dstShape)) {
            throw new IllegalArgumentException("Source and destination tensors must have the same shape.");
        }

        TensorStorage srcStorage = src.getStorage();
        TensorStorage dstStorage = dst.getStorage();
        int[] denseStrides = denseStrides(srcShape);
        int logicalSize = logicalSize(srcShape);

        for (int logicalIndex = 0; logicalIndex < logicalSize; logicalIndex++) {
            int srcOffset = logicalToOffset(logicalIndex, srcShape, srcStrides, denseStrides);
            int dstOffset = logicalToOffset(logicalIndex, dstShape, dstStrides, denseStrides);
            dstStorage.setAsDoubleAt(dstOffset, srcStorage.getAsDoubleAt(srcOffset));
        }
    }

    private static void parallelApply(Tensor src, Tensor dst) {
        int[] srcShape = src.getShape();
        int[] dstShape = dst.getShape();
        int[] srcStrides = src.getStrides();
        int[] dstStrides = dst.getStrides();

        if (!Arrays.equals(srcShape, dstShape)) {
            throw new IllegalArgumentException("Source and destination tensors must have the same shape.");
        }

        TensorStorage srcStorage = src.getStorage();
        TensorStorage dstStorage = dst.getStorage();
        int[] denseStrides = denseStrides(srcShape);
        int logicalSize = logicalSize(srcShape);

        ForkJoinPool customPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
        try {
            customPool.submit(() -> IntStream.range(0, logicalSize).parallel().forEach(logicalIndex -> {
                int srcOffset = logicalToOffset(logicalIndex, srcShape, srcStrides, denseStrides);
                int dstOffset = logicalToOffset(logicalIndex, dstShape, dstStrides, denseStrides);
                dstStorage.setAsDoubleAt(dstOffset, srcStorage.getAsDoubleAt(srcOffset));
            })).get();
        } catch (Exception e) {
            throw new RuntimeException("Parallel remap failed", e);
        } finally {
            customPool.shutdown();
        }
    }

    private static int logicalToOffset(int logicalIndex, int[] shape, int[] strides, int[] denseStrides) {
        int rem = logicalIndex;
        int offset = 0;
        for (int dim = 0; dim < shape.length; dim++) {
            int coord = rem / denseStrides[dim];
            rem %= denseStrides[dim];
            offset += coord * strides[dim];
        }
        return offset;
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

    private static int logicalSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    private static void applyF32(Tensor src, Tensor dst, int parallelThreshold) {
        int logicalSize = logicalSize(src.getShape());
        if (logicalSize > parallelThreshold) {
            parallelApplyF32(src, dst);
        } else {
            sequentialApplyF32(src, dst);
        }
    }

    private static void sequentialApplyF32(Tensor src, Tensor dst) {
        int[] srcShape = src.getShape();
        int[] dstShape = dst.getShape();
        int[] srcStrides = src.getStrides();
        int[] dstStrides = dst.getStrides();
        if (!Arrays.equals(srcShape, dstShape)) {
            throw new IllegalArgumentException("Source and destination tensors must have the same shape.");
        }
        float[] srcData = src.getFloat32Data();
        float[] dstData = dst.getFloat32Data();
        int[] denseStrides = denseStrides(srcShape);
        int logicalSize = logicalSize(srcShape);
        for (int logicalIndex = 0; logicalIndex < logicalSize; logicalIndex++) {
            int srcOffset = logicalToOffset(logicalIndex, srcShape, srcStrides, denseStrides);
            int dstOffset = logicalToOffset(logicalIndex, dstShape, dstStrides, denseStrides);
            dstData[dstOffset] = srcData[srcOffset];
        }
    }

    private static void parallelApplyF32(Tensor src, Tensor dst) {
        int[] srcShape = src.getShape();
        int[] dstShape = dst.getShape();
        int[] srcStrides = src.getStrides();
        int[] dstStrides = dst.getStrides();
        if (!Arrays.equals(srcShape, dstShape)) {
            throw new IllegalArgumentException("Source and destination tensors must have the same shape.");
        }
        float[] srcData = src.getFloat32Data();
        float[] dstData = dst.getFloat32Data();
        int[] denseStrides = denseStrides(srcShape);
        int logicalSize = logicalSize(srcShape);
        ForkJoinPool customPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
        try {
            customPool.submit(() -> IntStream.range(0, logicalSize).parallel().forEach(logicalIndex -> {
                int srcOffset = logicalToOffset(logicalIndex, srcShape, srcStrides, denseStrides);
                int dstOffset = logicalToOffset(logicalIndex, dstShape, dstStrides, denseStrides);
                dstData[dstOffset] = srcData[srcOffset];
            })).get();
        } catch (Exception e) {
            throw new RuntimeException("Parallel remap failed", e);
        } finally {
            customPool.shutdown();
        }
    }

    private static void applyF16(Tensor src, Tensor dst, int parallelThreshold) {
        int logicalSize = logicalSize(src.getShape());
        if (logicalSize > parallelThreshold) {
            parallelApplyF16(src, dst);
        } else {
            sequentialApplyF16(src, dst);
        }
    }

    private static void sequentialApplyF16(Tensor src, Tensor dst) {
        int[] srcShape = src.getShape();
        int[] dstShape = dst.getShape();
        int[] srcStrides = src.getStrides();
        int[] dstStrides = dst.getStrides();
        if (!Arrays.equals(srcShape, dstShape)) {
            throw new IllegalArgumentException("Source and destination tensors must have the same shape.");
        }
        short[] srcData = src.getFloat16Data();
        short[] dstData = dst.getFloat16Data();
        int[] denseStrides = denseStrides(srcShape);
        int logicalSize = logicalSize(srcShape);
        for (int logicalIndex = 0; logicalIndex < logicalSize; logicalIndex++) {
            int srcOffset = logicalToOffset(logicalIndex, srcShape, srcStrides, denseStrides);
            int dstOffset = logicalToOffset(logicalIndex, dstShape, dstStrides, denseStrides);
            dstData[dstOffset] = srcData[srcOffset];
        }
    }

    private static void parallelApplyF16(Tensor src, Tensor dst) {
        int[] srcShape = src.getShape();
        int[] dstShape = dst.getShape();
        int[] srcStrides = src.getStrides();
        int[] dstStrides = dst.getStrides();
        if (!Arrays.equals(srcShape, dstShape)) {
            throw new IllegalArgumentException("Source and destination tensors must have the same shape.");
        }
        short[] srcData = src.getFloat16Data();
        short[] dstData = dst.getFloat16Data();
        int[] denseStrides = denseStrides(srcShape);
        int logicalSize = logicalSize(srcShape);
        ForkJoinPool customPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
        try {
            customPool.submit(() -> IntStream.range(0, logicalSize).parallel().forEach(logicalIndex -> {
                int srcOffset = logicalToOffset(logicalIndex, srcShape, srcStrides, denseStrides);
                int dstOffset = logicalToOffset(logicalIndex, dstShape, dstStrides, denseStrides);
                dstData[dstOffset] = srcData[srcOffset];
            })).get();
        } catch (Exception e) {
            throw new RuntimeException("Parallel remap failed", e);
        } finally {
            customPool.shutdown();
        }
    }
}
