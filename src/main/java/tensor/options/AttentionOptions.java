package tensor.options;

/**
 * Immutable configuration for scaled dot-product attention.
 *
 * @param causal whether positions may only attend to current and earlier key positions
 * @param scaleOverride optional positive multiplier applied to query-key scores;
 *                      null means use {@code 1 / sqrt(headDim)}
 */
public record AttentionOptions(
        boolean causal,
        Double scaleOverride
) {
    /**
     * Validates attention option invariants.
     *
     * @throws IllegalArgumentException if {@code scaleOverride} is non-null and
     *                                  not strictly positive
     */
    public AttentionOptions {
        if (scaleOverride != null && !(scaleOverride > 0.0d)) {
            throw new IllegalArgumentException("Attention scaleOverride must be positive.");
        }
    }

    /**
     * Returns non-causal attention using the default head-dimension scale.
     *
     * @return immutable default options
     */
    public static AttentionOptions defaults() {
        return new AttentionOptions(false, null);
    }

    /**
     * Returns causal attention using the default head-dimension scale.
     *
     * @return immutable causal options
     */
    public static AttentionOptions causalDefaults() {
        return new AttentionOptions(true, null);
    }

    /**
     * Returns a copy with updated causal masking.
     *
     * @param causal true to apply a lower-triangular causal mask
     * @return new immutable options instance
     */
    public AttentionOptions withCausal(boolean causal) {
        return new AttentionOptions(causal, scaleOverride);
    }

    /**
     * Returns a copy with an explicit score scale.
     *
     * @param scaleOverride strictly positive score scale
     * @return new immutable options instance
     * @throws IllegalArgumentException if {@code scaleOverride} is not positive
     */
    public AttentionOptions withScale(double scaleOverride) {
        return new AttentionOptions(causal, scaleOverride);
    }

    /**
     * Resolves the score scale for a given attention head dimension.
     *
     * @param headDim size of the key/query feature dimension; must be positive
     *                for the default scale to be meaningful
     * @return {@code scaleOverride} when present, otherwise {@code 1 / sqrt(headDim)}
     */
    public double resolveScale(int headDim) {
        if (scaleOverride != null) {
            return scaleOverride;
        }
        return 1.0d / Math.sqrt(headDim);
    }
}
