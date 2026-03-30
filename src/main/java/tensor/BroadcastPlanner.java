package tensor;

import java.util.Arrays;

public final class BroadcastPlanner {
    private BroadcastPlanner() {}

    public static BroadcastPlan plan(Tensor a, Tensor b) {
        return plan(a.getShape(), a.getStrides(), b.getShape(), b.getStrides());
    }

    public static BroadcastPlan plan(int[] aShape, int[] aStrides, int[] bShape, int[] bStrides) {
        int rank = Math.max(aShape.length, bShape.length);

        int[] aAlignedShape = align(aShape, rank, 1);
        int[] bAlignedShape = align(bShape, rank, 1);
        int[] aAlignedStrides = align(aStrides, rank, 0);
        int[] bAlignedStrides = align(bStrides, rank, 0);

        int[] outShape = new int[rank];
        int[] aEffStrides = new int[rank];
        int[] bEffStrides = new int[rank];
        int[] tempReduceA = new int[rank];
        int[] tempReduceB = new int[rank];
        int countA = 0;
        int countB = 0;
        boolean noBroadcast = true;

        for (int d = 0; d < rank; d++) {
            int ad = aAlignedShape[d];
            int bd = bAlignedShape[d];
            if (ad != bd && ad != 1 && bd != 1) {
                throw new IllegalArgumentException(
                        "Broadcast mismatch at dim " + d + ": " + ad + " vs " + bd
                );
            }
            int outDim = Math.max(ad, bd);
            outShape[d] = outDim;

            if (ad == 1 && outDim > 1) {
                aEffStrides[d] = 0;
                tempReduceA[countA++] = d;
                noBroadcast = false;
            } else {
                aEffStrides[d] = aAlignedStrides[d];
            }

            if (bd == 1 && outDim > 1) {
                bEffStrides[d] = 0;
                tempReduceB[countB++] = d;
                noBroadcast = false;
            } else {
                bEffStrides[d] = bAlignedStrides[d];
            }
        }

        int[] outStrides = TensorMetadata.computeStrides(outShape);
        return new BroadcastPlan(
                outShape,
                outStrides,
                aEffStrides,
                bEffStrides,
                Arrays.copyOf(tempReduceA, countA),
                Arrays.copyOf(tempReduceB, countB),
                noBroadcast
        );
    }

    private static int[] align(int[] array, int rank, int padValue) {
        if (array.length == rank) {
            return Arrays.copyOf(array, rank);
        }

        int[] out = new int[rank];
        int offset = rank - array.length;
        if (padValue != 0) {
            Arrays.fill(out, 0, offset, padValue);
        }
        System.arraycopy(array, 0, out, offset, array.length);
        return out;
    }
}
