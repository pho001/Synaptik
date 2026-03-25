package Backend.kernels.cpu;

import Tensor.BroadcastPlan;

public final class ResolvedBroadcastPlan {
    private final int[] outShape;
    private final int[] outStrides;
    private final int[] aEffStrides;
    private final int[] bEffStrides;
    private final boolean noBroadcast;

    private ResolvedBroadcastPlan(
            int[] outShape,
            int[] outStrides,
            int[] aEffStrides,
            int[] bEffStrides,
            boolean noBroadcast
    ) {
        this.outShape = outShape;
        this.outStrides = outStrides;
        this.aEffStrides = aEffStrides;
        this.bEffStrides = bEffStrides;
        this.noBroadcast = noBroadcast;
    }

    public static ResolvedBroadcastPlan from(BroadcastPlan plan) {
        if (plan == null) {
            return null;
        }
        return new ResolvedBroadcastPlan(
                plan.outShape(),
                plan.outStrides(),
                plan.aEffStrides(),
                plan.bEffStrides(),
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

    public boolean isNoBroadcast() {
        return noBroadcast;
    }
}
