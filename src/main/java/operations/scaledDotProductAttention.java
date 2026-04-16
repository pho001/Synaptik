package operations;

public final class scaledDotProductAttention implements Operation {
    private final double scale;
    private final boolean hasMask;

    public scaledDotProductAttention(double scale, boolean hasMask) {
        if (!(scale > 0.0d)) {
            throw new IllegalArgumentException("scaledDotProductAttention scale must be positive.");
        }
        this.scale = scale;
        this.hasMask = hasMask;
    }

    @Override
    public OpType opType() {
        return OpType.SCALED_DOT_PRODUCT_ATTENTION;
    }

    @Override
    public String getExpression() {
        return "scaledDotProductAttention";
    }

    public double getScale() {
        return scale;
    }

    public boolean hasMask() {
        return hasMask;
    }
}
