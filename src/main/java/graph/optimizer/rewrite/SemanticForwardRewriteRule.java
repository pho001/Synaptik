package graph.optimizer.rewrite;

import config.optimizer.RewriteConfig;
import graph.optimizer.OptimizationRule;
import graph.optimizer.state.OptimizerState;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Rewrite pipeline used by semantic forward canonicalization.
 *
 * <p>This stage runs only forward-safe rewrites before backward graph construction. Backward-specific lowering is
 * intentionally excluded because the backward graph has not been built yet.
 */
public final class SemanticForwardRewriteRule implements OptimizationRule {
    private final List<OptimizationRule> delegates;

    /**
     * Creates a forward-only rewrite pipeline.
     *
     * @param config rewrite configuration, or {@code null} for defaults
     */
    public SemanticForwardRewriteRule(RewriteConfig config) {
        RewriteConfig resolved = config == null ? RewriteConfig.defaults() : config;
        ArrayList<OptimizationRule> configured = new ArrayList<>();
        if (resolved.piecewiseLowering().anyEnabled()) {
            configured.add(new PiecewiseLoweringRewrite(resolved.piecewiseLowering()));
        }
        if (resolved.algebraic().enabled()) {
            configured.add(new AlgebraicRewrite());
        }
        if (resolved.linearLowering().enabled()) {
            configured.add(new LinearLoweringRewrite());
        }
        configured.add(new LossForwardLoweringRewrite());
        this.delegates = List.copyOf(configured);
    }

    /**
     * Applies all forward-safe rewrite delegates in order.
     *
     * @param state optimizer state to rewrite
     * @return rewritten optimizer state
     */
    @Override
    public OptimizerState apply(OptimizerState state) {
        OptimizerState current = Objects.requireNonNull(state, "state cannot be null");
        for (OptimizationRule delegate : delegates) {
            current = delegate.apply(current);
        }
        return current;
    }
}
