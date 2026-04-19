package graph.optimizer.rewrite;

import config.optimizer.RewriteConfig;
import graph.optimizer.OptimizationRule;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class SemanticForwardRewriteRule implements OptimizationRule {
    private final List<OptimizationRule> delegates;

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
        configured.add(new AttentionLoweringRewrite());
        this.delegates = List.copyOf(configured);
    }

    @Override
    public List<tensor.Tensor> apply(List<tensor.Tensor> sortedGraph) {
        List<tensor.Tensor> current = Objects.requireNonNull(sortedGraph, "sortedGraph cannot be null");
        for (OptimizationRule delegate : delegates) {
            current = delegate.apply(current);
        }
        return current;
    }
}
