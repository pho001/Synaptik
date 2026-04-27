package graph.optimizer.rewrite;

import config.optimizer.RewriteConfig;
import config.optimizer.Conv2dLoweringMode;
import graph.optimizer.OptimizationRule;
import graph.optimizer.state.OptimizerState;
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
        this(config, createDelegates(config));
    }

    public RewriteConfig config() {
        return config;
    }

    private static List<OptimizationRule> createDelegates(RewriteConfig config) {
        RewriteConfig resolved = config == null ? RewriteConfig.defaults() : config;
        java.util.ArrayList<OptimizationRule> delegates = new java.util.ArrayList<>();
        addImportCanonicalizationDelegates(delegates, resolved);
        if (resolved.algebraic().enabled()) {
            delegates.add(new AlgebraicRewrite());
        }
        if (resolved.linearLowering().enabled()) {
            delegates.add(new LinearLoweringRewrite());
        }
        delegates.add(new LossLoweringRewrite());
        delegates.add(new ReductionLoweringRewrite());
        delegates.add(new AttentionLoweringRewrite());
        delegates.add(new AttentionBackwardLoweringRewrite());
        if (resolved.conv2dLowering().mode() != Conv2dLoweringMode.OFF) {
            delegates.add(new Conv2dLoweringRewrite(resolved.conv2dLowering()));
        }
        return List.copyOf(delegates);
    }

    private static void addImportCanonicalizationDelegates(
            java.util.ArrayList<OptimizationRule> delegates,
            RewriteConfig resolved
    ) {
        if (resolved.piecewiseLowering().anyEnabled()) {
            delegates.add(new PiecewiseLoweringRewrite(resolved.piecewiseLowering()));
        }
    }

    @Override
    public OptimizerState apply(OptimizerState state) {
        OptimizerState current = state;
        for (OptimizationRule delegate : delegates) {
            current = delegate.apply(current);
        }
        return current;
    }
}
