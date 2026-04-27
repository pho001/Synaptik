package backend.cpu.kernels.layout.plan;

import tensor.WhereBroadcastPlan;

public final class ResolvedWhereBroadcastPlan {
    private final int[] outShape;
    private final int[] outStrides;
    private final int[] condEffStrides;
    private final int[] trueEffStrides;
    private final int[] falseEffStrides;
    private final int[] condResets;
    private final int[] trueResets;
    private final int[] falseResets;
    private final boolean noBroadcast;

    private ResolvedWhereBroadcastPlan(
            int[] outShape,
            int[] outStrides,
            int[] condEffStrides,
            int[] trueEffStrides,
            int[] falseEffStrides,
            int[] condResets,
            int[] trueResets,
            int[] falseResets,
            boolean noBroadcast
    ) {
        this.outShape = outShape;
        this.outStrides = outStrides;
        this.condEffStrides = condEffStrides;
        this.trueEffStrides = trueEffStrides;
        this.falseEffStrides = falseEffStrides;
        this.condResets = condResets;
        this.trueResets = trueResets;
        this.falseResets = falseResets;
        this.noBroadcast = noBroadcast;
    }

    public static ResolvedWhereBroadcastPlan from(WhereBroadcastPlan plan) {
        if (plan == null) {
            return null;
        }
        int[] outShape = plan.outShape();
        int[] cond = plan.condEffStrides();
        int[] t = plan.trueEffStrides();
        int[] f = plan.falseEffStrides();
        int[] condResets = new int[outShape.length];
        int[] trueResets = new int[outShape.length];
        int[] falseResets = new int[outShape.length];
        for (int d = 0; d < outShape.length; d++) {
            condResets[d] = outShape[d] * cond[d];
            trueResets[d] = outShape[d] * t[d];
            falseResets[d] = outShape[d] * f[d];
        }
        return new ResolvedWhereBroadcastPlan(
                outShape,
                plan.outStrides(),
                cond,
                t,
                f,
                condResets,
                trueResets,
                falseResets,
                plan.isNoBroadcast()
        );
    }

    public int[] outShape() { return outShape; }
    public int[] outStrides() { return outStrides; }
    public int[] condEffStrides() { return condEffStrides; }
    public int[] trueEffStrides() { return trueEffStrides; }
    public int[] falseEffStrides() { return falseEffStrides; }
    public int[] condResets() { return condResets; }
    public int[] trueResets() { return trueResets; }
    public int[] falseResets() { return falseResets; }
    public boolean isNoBroadcast() { return noBroadcast; }
}
