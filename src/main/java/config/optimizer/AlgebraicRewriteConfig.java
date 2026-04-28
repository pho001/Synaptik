package config.optimizer;

/**
 * Enables or disables algebraic simplification rewrites.
 *
 * <p>Algebraic rewrites are graph-semantic transformations such as replacing redundant arithmetic
 * patterns with simpler equivalent nodes. They do not execute tensors directly.</p>
 *
 * @param enabled whether algebraic rewrites should run when the AR stage is active
 */
public record AlgebraicRewriteConfig(
        boolean enabled
) {
    /**
     * @return default enabled algebraic rewrite config
     */
    public static AlgebraicRewriteConfig defaults() {
        return new AlgebraicRewriteConfig(true);
    }

    /**
     * @return config that disables algebraic rewrites
     */
    public static AlgebraicRewriteConfig disabled() {
        return new AlgebraicRewriteConfig(false);
    }
}
