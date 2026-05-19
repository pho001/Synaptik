package graph.optimizer.rewrite.canonical;

import config.optimizer.PiecewiseLoweringConfig;
import graph.optimizer.rewrite.LocalTensorRewriteRule;
import tensor.Tensor;

/**
 * Optional canonicalization pass for externally imported or manually decomposed graphs.
 *
 * <p>Internal Tensor builders should prefer creating the specialized surface op directly
 * instead of relying on this rewrite as a repair step.
 */
public final class PiecewiseCanonicalizationRule extends LocalTensorRewriteRule {
    private final PiecewisePatternLowerer lowerer;

    /**
     * Creates a piecewise lowering rewrite.
     *
     * @param config piecewise lowering configuration, or {@code null} for defaults
     */
    public PiecewiseCanonicalizationRule(PiecewiseLoweringConfig config) {
        this.lowerer = new PiecewisePatternLowerer(config);
    }

    @Override
    protected Tensor rewriteTensor(Tensor tensor) {
        Tensor lowered = lowerer.lower(tensor);
        return lowered == null ? tensor : lowered;
    }
}
