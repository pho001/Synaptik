package operations.normalization;
import operations.Operation;

/**
 * Applies layer normalization over the trailing normalized dimensions.
 *
 * <p>{@code normalizedRank} determines how many final axes participate in
 * the statistic, and {@code epsilon} stabilizes the denominator.</p>
 */
public final class layerNorm implements Operation {
    private final int normalizedRank;
    private final double epsilon;

    /**
     * Creates a normalization descriptor.
     *
     * @param normalizedRank number of trailing dimensions to normalize
     * @param epsilon positive numerical stability constant
     * @throws IllegalArgumentException if {@code normalizedRank < 1} or
     *        {@code epsilon <= 0}
     */
    public layerNorm(int normalizedRank, double epsilon) {
        if (normalizedRank < 1) {
            throw new IllegalArgumentException("normalizedRank must be >= 1");
        }
        if (!(epsilon > 0.0d)) {
            throw new IllegalArgumentException("epsilon must be > 0");
        }
        this.normalizedRank = normalizedRank;
        this.epsilon = epsilon;
    }

    @Override
    public OpType opType() {
        return OpType.LAYER_NORM;
    }

    @Override
    public String getExpression() {
        return "layerNorm";
    }

    /**
     * Returns the number of trailing normalized dimensions.
     *
     * @return normalized rank
     */
    public int getNormalizedRank() {
        return normalizedRank;
    }

    /**
     * Returns the numerical stability constant.
     *
     * @return positive epsilon value
     */
    public double getEpsilon() {
        return epsilon;
    }
}
