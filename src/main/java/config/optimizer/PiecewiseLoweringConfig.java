package config.optimizer;

/**
 * Controls lowering of piecewise scalar expressions into canonical compare/select forms.
 *
 * @param canonicalSigmoid whether sigmoid-like forms may be canonicalized
 * @param reluLikeWhere whether relu-like forms may be represented as where/select expressions
 * @param clampLikeWhere whether clamp-like forms may be represented as where/select expressions
 */
public record PiecewiseLoweringConfig(
        boolean canonicalSigmoid,
        boolean reluLikeWhere,
        boolean clampLikeWhere
) {
    /**
     * @return defaults with piecewise lowering disabled
     */
    public static PiecewiseLoweringConfig defaults() {
        return new PiecewiseLoweringConfig(false, false, false);
    }

    /**
     * @return defaults with all piecewise lowering forms enabled
     */
    public static PiecewiseLoweringConfig aggressiveDefaults() {
        return new PiecewiseLoweringConfig(true, true, true);
    }

    /**
     * @return {@code true} when any piecewise lowering option is enabled
     */
    public boolean anyEnabled() {
        return canonicalSigmoid || reluLikeWhere || clampLikeWhere;
    }
}
