package config.optimizer;

public record PiecewiseLoweringConfig(
        boolean canonicalSigmoid,
        boolean reluLikeWhere,
        boolean clampLikeWhere
) {
    public static PiecewiseLoweringConfig defaults() {
        return new PiecewiseLoweringConfig(false, false, false);
    }

    public static PiecewiseLoweringConfig aggressiveDefaults() {
        return new PiecewiseLoweringConfig(true, true, true);
    }

    public boolean anyEnabled() {
        return canonicalSigmoid || reluLikeWhere || clampLikeWhere;
    }
}
