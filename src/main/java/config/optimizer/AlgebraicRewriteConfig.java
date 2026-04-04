package config.optimizer;

public record AlgebraicRewriteConfig(
        boolean enabled
) {
    public static AlgebraicRewriteConfig defaults() {
        return new AlgebraicRewriteConfig(true);
    }

    public static AlgebraicRewriteConfig disabled() {
        return new AlgebraicRewriteConfig(false);
    }
}
