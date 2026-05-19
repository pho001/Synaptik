package tensor.layout;

public final class WhereBroadcastPlan {
    private final int[] outShape;
    private final int[] outStrides;
    private final int[] condEffStrides;
    private final int[] trueEffStrides;
    private final int[] falseEffStrides;
    private final boolean noBroadcast;

    public WhereBroadcastPlan(
            int[] outShape,
            int[] outStrides,
            int[] condEffStrides,
            int[] trueEffStrides,
            int[] falseEffStrides,
            boolean noBroadcast
    ) {
        this.outShape = outShape.clone();
        this.outStrides = outStrides.clone();
        this.condEffStrides = condEffStrides.clone();
        this.trueEffStrides = trueEffStrides.clone();
        this.falseEffStrides = falseEffStrides.clone();
        this.noBroadcast = noBroadcast;
    }

    public int[] outShape() {
        return outShape.clone();
    }

    public int[] outStrides() {
        return outStrides.clone();
    }

    public int[] condEffStrides() {
        return condEffStrides.clone();
    }

    public int[] trueEffStrides() {
        return trueEffStrides.clone();
    }

    public int[] falseEffStrides() {
        return falseEffStrides.clone();
    }

    public boolean isNoBroadcast() {
        return noBroadcast;
    }
}
