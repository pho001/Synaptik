package config.optimizer;

/**
 * Configuration bundle for the optimizer rewrite stage.
 *
 * <p>Rewrite configuration is split by rewrite family so graph autotune and tests can vary one family
 * without rebuilding the entire optimizer config.</p>
 *
 * @param algebraic algebraic simplification settings; {@code null} uses defaults
 * @param linearLowering linear lowering settings; {@code null} uses defaults
 * @param conv2dLowering conv2d lowering settings; {@code null} uses defaults
 * @param piecewiseLowering piecewise/select lowering settings; {@code null} uses defaults
 */
public record RewriteConfig(
        AlgebraicRewriteConfig algebraic,
        LinearLoweringConfig linearLowering,
        Conv2dLoweringConfig conv2dLowering,
        PiecewiseLoweringConfig piecewiseLowering
) {
    public RewriteConfig {
        algebraic = algebraic == null ? AlgebraicRewriteConfig.defaults() : algebraic;
        linearLowering = linearLowering == null ? LinearLoweringConfig.defaults() : linearLowering;
        conv2dLowering = conv2dLowering == null ? Conv2dLoweringConfig.defaults() : conv2dLowering;
        piecewiseLowering = piecewiseLowering == null ? PiecewiseLoweringConfig.defaults() : piecewiseLowering;
    }

    /**
     * Creates rewrite config overriding only conv2d lowering.
     *
     * @param conv2dLowering conv2d lowering config
     */
    public RewriteConfig(
            Conv2dLoweringConfig conv2dLowering
    ) {
        this(
                AlgebraicRewriteConfig.defaults(),
                LinearLoweringConfig.defaults(),
                conv2dLowering,
                PiecewiseLoweringConfig.defaults()
        );
    }

    /**
     * @return default rewrite configuration
     */
    public static RewriteConfig defaults() {
        return new RewriteConfig(
                AlgebraicRewriteConfig.defaults(),
                LinearLoweringConfig.defaults(),
                Conv2dLoweringConfig.defaults(),
                PiecewiseLoweringConfig.defaults()
        );
    }

    /**
     * @param newAlgebraic replacement algebraic rewrite config
     * @return updated rewrite config
     */
    public RewriteConfig withAlgebraic(AlgebraicRewriteConfig newAlgebraic) {
        return new RewriteConfig(newAlgebraic, linearLowering, conv2dLowering, piecewiseLowering);
    }

    /**
     * @param newLinearLowering replacement linear lowering config
     * @return updated rewrite config
     */
    public RewriteConfig withLinearLowering(LinearLoweringConfig newLinearLowering) {
        return new RewriteConfig(algebraic, newLinearLowering, conv2dLowering, piecewiseLowering);
    }

    /**
     * @param newConv2dLowering replacement conv2d lowering config
     * @return updated rewrite config
     */
    public RewriteConfig withConv2dLowering(Conv2dLoweringConfig newConv2dLowering) {
        return new RewriteConfig(algebraic, linearLowering, newConv2dLowering, piecewiseLowering);
    }

    /**
     * @param newPiecewiseLowering replacement piecewise lowering config
     * @return updated rewrite config
     */
    public RewriteConfig withPiecewiseLowering(PiecewiseLoweringConfig newPiecewiseLowering) {
        return new RewriteConfig(algebraic, linearLowering, conv2dLowering, newPiecewiseLowering);
    }
}
