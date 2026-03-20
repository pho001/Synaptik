package Tensor;

import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public final class TensorRemap {
    private TensorRemap() {}

    public static void apply(Tensor src, Tensor dst, int parallelThreshold) {
        if (src.getFlatDataSize() > parallelThreshold) {
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

        double[] srcData = src.getData();
        double[] dstData = dst.getData();

        for (int i = 0; i < srcData.length; i++) {
            int flatIndexDst = 0;
            int index = i;

            for (int dim = 0; dim < srcShape.length; dim++) {
                int spatialIndex = index / srcStrides[dim];
                index %= srcStrides[dim];
                flatIndexDst += spatialIndex * dstStrides[dim];
            }
            dstData[flatIndexDst] = srcData[i];
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

        double[] srcData = src.getData();
        double[] dstData = dst.getData();

        ForkJoinPool customPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
        try {
            customPool.submit(() -> IntStream.range(0, srcData.length).parallel().forEach(i -> {
                int flatIndexDst = 0;
                int index = i;
                for (int dim = 0; dim < srcShape.length; dim++) {
                    int spatialIndex = index / srcStrides[dim];
                    index %= srcStrides[dim];
                    flatIndexDst += spatialIndex * dstStrides[dim];
                }
                dstData[flatIndexDst] = srcData[i];
            })).get();
        } catch (Exception e) {
            throw new RuntimeException("Parallel remap failed", e);
        } finally {
            customPool.shutdown();
        }
    }
}
