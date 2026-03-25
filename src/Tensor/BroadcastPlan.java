package Tensor;

import java.util.Arrays;

public final class BroadcastPlan {
    private final int[] outShape;
    private final int[] outStrides;
    private final int[] aEffStrides;
    private final int[] bEffStrides;
    private final int[] reduceAxesForA;
    private final int[] reduceAxesForB;
    private final boolean noBroadcast;

    public BroadcastPlan(
            int[] outShape,
            int[] outStrides,
            int[] aEffStrides,
            int[] bEffStrides,
            int[] reduceAxesForA,
            int[] reduceAxesForB,
            boolean noBroadcast
    ) {
        this.outShape = outShape.clone();
        this.outStrides = outStrides.clone();
        this.aEffStrides = aEffStrides.clone();
        this.bEffStrides = bEffStrides.clone();
        this.reduceAxesForA = reduceAxesForA.clone();
        this.reduceAxesForB = reduceAxesForB.clone();
        this.noBroadcast = noBroadcast;
    }

    public int[] outShape() {
        return outShape.clone();
    }

    public int[] outStrides() {
        return outStrides.clone();
    }

    public int[] aEffStrides() {
        return aEffStrides.clone();
    }

    public int[] bEffStrides() {
        return bEffStrides.clone();
    }

    public int[] reduceAxesForA() {
        return reduceAxesForA.clone();
    }

    public int[] reduceAxesForB() {
        return reduceAxesForB.clone();
    }

    public int rank() {
        return outShape.length;
    }

    public boolean isNoBroadcast() {
        return noBroadcast;
    }

    public int flatSize() {
        int size = 1;
        for (int dim : outShape) {
            size *= dim;
        }
        return size;
    }

    @Override
    public String toString() {
        return "BroadcastPlan{" +
                "outShape=" + Arrays.toString(outShape) +
                ", outStrides=" + Arrays.toString(outStrides) +
                ", aEffStrides=" + Arrays.toString(aEffStrides) +
                ", bEffStrides=" + Arrays.toString(bEffStrides) +
                ", reduceAxesForA=" + Arrays.toString(reduceAxesForA) +
                ", reduceAxesForB=" + Arrays.toString(reduceAxesForB) +
                ", noBroadcast=" + noBroadcast +
                '}';
    }
}

