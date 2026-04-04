package config.optimizer;

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

    public static RewriteConfig defaults() {
        return new RewriteConfig(
                AlgebraicRewriteConfig.defaults(),
                LinearLoweringConfig.defaults(),
                Conv2dLoweringConfig.defaults(),
                PiecewiseLoweringConfig.defaults()
        );
    }

    public RewriteConfig withAlgebraic(AlgebraicRewriteConfig newAlgebraic) {
        return new RewriteConfig(newAlgebraic, linearLowering, conv2dLowering, piecewiseLowering);
    }

    public RewriteConfig withLinearLowering(LinearLoweringConfig newLinearLowering) {
        return new RewriteConfig(algebraic, newLinearLowering, conv2dLowering, piecewiseLowering);
    }

    public RewriteConfig withConv2dLowering(Conv2dLoweringConfig newConv2dLowering) {
        return new RewriteConfig(algebraic, linearLowering, newConv2dLowering, piecewiseLowering);
    }

    public RewriteConfig withPiecewiseLowering(PiecewiseLoweringConfig newPiecewiseLowering) {
        return new RewriteConfig(algebraic, linearLowering, conv2dLowering, newPiecewiseLowering);
    }
}
