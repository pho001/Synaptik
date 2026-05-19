package backend.cpu.kernels.layout.plan;

import tensor.layout.BroadcastPlan;

public final class ResolvedBroadcastPlan {
    private final int[] outShape;
    private final int[] outStrides;
    private final int[] aEffStrides;
    private final int[] bEffStrides;
    private final int[] aResets;
    private final int[] bResets;
    private final boolean noBroadcast;

    private ResolvedBroadcastPlan(
            int[] outShape,
            int[] outStrides,
            int[] aEffStrides,
            int[] bEffStrides,
            int[] aResets,
            int[] bResets,
            boolean noBroadcast
    ) {
        this.outShape = outShape;
        this.outStrides = outStrides;
        this.aEffStrides = aEffStrides;
        this.bEffStrides = bEffStrides;
        this.aResets = aResets;
        this.bResets = bResets;
        this.noBroadcast = noBroadcast;
    }

    public static ResolvedBroadcastPlan from(BroadcastPlan plan) {
        if (plan == null) {
            return null;
        }
        int[] outShape = plan.outShape();
        int[] aEffStrides = plan.aEffStrides();
        int[] bEffStrides = plan.bEffStrides();
        int[] aResets = new int[outShape.length];
        int[] bResets = new int[outShape.length];
        for (int d = 0; d < outShape.length; d++) {
            aResets[d] = outShape[d] * aEffStrides[d];
            bResets[d] = outShape[d] * bEffStrides[d];
        }
        return new ResolvedBroadcastPlan(
                outShape,
                plan.outStrides(),
                aEffStrides,
                bEffStrides,
                aResets,
                bResets,
                plan.isNoBroadcast()
        );
    }

    public int[] outShape() {
        return outShape;
    }

    public int[] outStrides() {
        return outStrides;
    }

    public int[] aEffStrides() {
        return aEffStrides;
    }

    public int[] bEffStrides() {
        return bEffStrides;
    }

    public int[] aResets() {
        return aResets;
    }

    public int[] bResets() {
        return bResets;
    }

    public boolean isNoBroadcast() {
        return noBroadcast;
    }
}
