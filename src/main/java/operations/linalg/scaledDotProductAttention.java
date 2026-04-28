package operations.linalg;
import operations.Operation;

/**
 * Computes scaled dot-product attention.
 *
 * <p>The descriptor records the positive scale applied to query-key scores and
 * whether an attention mask input is present. Query, key, value, and optional
 * mask shapes are validated by the tensor front end/backend.</p>
 */
public final class scaledDotProductAttention implements Operation {
    private final double scale;
    private final boolean hasMask;

    /**
     * Creates an attention descriptor.
     *
     * @param scale positive multiplier applied to query-key scores
     * @param hasMask whether a mask input participates in the operation
     * @throws IllegalArgumentException if {@code scale} is not positive
     */
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

    /**
     * Returns the attention score scale.
     *
     * @return positive query-key score multiplier
     */
    public double getScale() {
        return scale;
    }

    /**
     * Indicates whether this attention descriptor expects a mask input.
     *
     * @return {@code true} when a mask input is present
     */
    public boolean hasMask() {
        return hasMask;
    }
}
