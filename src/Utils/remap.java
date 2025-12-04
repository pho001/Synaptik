package Utils;

import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

import Tensor.Tensor;

public class remap {
    public static void apply(Tensor src,Tensor dst,int parallelThreshold){
        // read necessary info from tensors
        if (src.getFlatDataSize()>parallelThreshold){
            parallelApply(src,dst);
        }
        else {
            sequentialApply(src,dst);
        }

    }

    private static void sequentialApply(Tensor src,Tensor dst){
        int[] srcShape = src.getShape();
        int[] dstShape = dst.getShape();
        int[] srcStrides = src.getStrides();
        int[] dstStrides = dst.getStrides();

        // check
        if (!Arrays.equals(srcShape, dstShape)) {
            throw new IllegalArgumentException("Source and destination tensors must have the same shape.");
        }

        double[] srcData = src.getData();
        double[] dstData = dst.getData();


        // Recalculate index for each value in src so we can store it in dst
        for (int i = 0; i < srcData.length; i++) {
            int flatIndexDst = 0;
            int index = i;

            // calculate spatial index for each input flat index and then save it in dst on flat index for given spatial index
            for (int dim = 0; dim < srcShape.length; dim++) {
                int spatialIndex = index / srcStrides[dim];
                index %= srcStrides[dim];

                flatIndexDst += spatialIndex * dstStrides[dim];
            }

            // Přepisujeme data z src na odpovídající místo v dst
            dstData[flatIndexDst] = srcData[i];
        }
    }

    private static void parallelApply(Tensor src,Tensor dst){
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
            })).get(); // blokuje dokud neskončí
        } catch (Exception e) {
            throw new RuntimeException("Parallel remap failed", e);
        } finally {
            customPool.shutdown();
        }
    }

}
