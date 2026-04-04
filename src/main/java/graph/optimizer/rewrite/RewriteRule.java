package graph.optimizer.rewrite;

import config.optimizer.RewriteConfig;
import graph.optimizer.OptimizationRule;
import tensor.Tensor;

import java.util.List;
import java.util.Objects;

public class RewriteRule implements OptimizationRule {
    private final List<OptimizationRule> delegates;
    private final RewriteConfig config;

    public RewriteRule(List<OptimizationRule> delegates) {
        this(RewriteConfig.defaults(), delegates);
    }

    public RewriteRule(RewriteConfig config, List<OptimizationRule> delegates) {
        this.config = config == null ? RewriteConfig.defaults() : config;
        this.delegates = List.copyOf(Objects.requireNonNull(delegates, "delegates cannot be null"));
    }

    public static RewriteRule defaults() {
        return new RewriteRule(RewriteConfig.defaults());
    }

    public RewriteRule(RewriteConfig config) {
        this(
                config,
                List.of(
                        new AlgebraicRewrite(),
                        new LinearLoweringRewrite(),
                        new Conv2dLoweringRewrite((config == null ? RewriteConfig.defaults() : config).conv2dLowering())
                )
        );
    }

    public RewriteConfig config() {
        return config;
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
