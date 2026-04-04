package graph.optimizer.rewrite;

import graph.optimizer.OptimizationRule;
import tensor.Tensor;

import java.util.List;
import java.util.Objects;

public class RewriteRule implements OptimizationRule {
    private final List<OptimizationRule> delegates;

    public RewriteRule(List<OptimizationRule> delegates) {
        this.delegates = List.copyOf(Objects.requireNonNull(delegates, "delegates cannot be null"));
    }

    public static RewriteRule defaults() {
        return new RewriteRule(List.of(
                new AlgebraicRewrite(),
                new LinearLoweringRewrite()
        ));
    }

    @Override
    public List<Tensor> apply(List<Tensor> sortedGraph) {
        List<Tensor> current = sortedGraph;
        for (OptimizationRule delegate : delegates) {
            current = delegate.apply(current);
        }
        return current;
    }
}
