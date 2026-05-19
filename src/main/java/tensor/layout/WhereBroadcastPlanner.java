package tensor.layout;

import tensor.Tensor;
import tensor.TensorMetadata;

import java.util.Arrays;

public final class WhereBroadcastPlanner {
    private WhereBroadcastPlanner() {}

    public static WhereBroadcastPlan plan(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        return plan(
                condition.getShape(), condition.getStrides(),
                ifTrue.getShape(), ifTrue.getStrides(),
                ifFalse.getShape(), ifFalse.getStrides()
        );
    }

    public static WhereBroadcastPlan plan(
            int[] condShape, int[] condStrides,
            int[] trueShape, int[] trueStrides,
            int[] falseShape, int[] falseStrides
    ) {
        int rank = Math.max(condShape.length, Math.max(trueShape.length, falseShape.length));

        int[] cShape = align(condShape, rank, 1);
        int[] tShape = align(trueShape, rank, 1);
        int[] fShape = align(falseShape, rank, 1);
        int[] cStrides = align(condStrides, rank, 0);
        int[] tStrides = align(trueStrides, rank, 0);
        int[] fStrides = align(falseStrides, rank, 0);

        int[] outShape = new int[rank];
        int[] cEff = new int[rank];
        int[] tEff = new int[rank];
        int[] fEff = new int[rank];
        boolean noBroadcast = true;

        for (int d = 0; d < rank; d++) {
            int outDim = commonOutDim(cShape[d], tShape[d], fShape[d], d);
            outShape[d] = outDim;

            cEff[d] = effectiveStride(cShape[d], outDim, cStrides[d]);
            tEff[d] = effectiveStride(tShape[d], outDim, tStrides[d]);
            fEff[d] = effectiveStride(fShape[d], outDim, fStrides[d]);

            if (cEff[d] == 0 || tEff[d] == 0 || fEff[d] == 0) {
                noBroadcast = false;
            }
        }

        return new WhereBroadcastPlan(
                outShape,
                TensorMetadata.computeStrides(outShape),
                cEff,
                tEff,
                fEff,
                noBroadcast
        );
    }

    private static int commonOutDim(int a, int b, int c, int dimension) {
        int out = pairwiseOutDim(a, b, dimension);
        return pairwiseOutDim(out, c, dimension);
    }

    private static int pairwiseOutDim(int a, int b, int dimension) {
        if (a != b && a != 1 && b != 1) {
            throw new IllegalArgumentException("Where broadcast mismatch at dim " + dimension + ": " + a + " vs " + b);
        }
        return Math.max(a, b);
    }

    private static int effectiveStride(int inDim, int outDim, int stride) {
        return (inDim == 1 && outDim > 1) ? 0 : stride;
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
