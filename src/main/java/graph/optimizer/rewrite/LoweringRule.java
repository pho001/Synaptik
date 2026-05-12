package graph.optimizer.rewrite;

import config.optimizer.Conv2dLoweringMode;
import config.optimizer.RewriteConfig;
import graph.optimizer.OptimizationRule;
import graph.optimizer.state.OptimizerState;

import java.util.ArrayList;
import java.util.List;

/**
 * Backend-neutral lowering stage that runs after cleanup and before partitioning.
 */
public final class LoweringRule implements OptimizationRule {
    private final List<OptimizationRule> delegates;

    public LoweringRule(RewriteConfig config) {
        RewriteConfig resolved = config == null ? RewriteConfig.defaults() : config;
        ArrayList<OptimizationRule> rules = new ArrayList<>();
        if (resolved.linearLowering().enabled()) {
            rules.add(new LinearLoweringRewrite());
        }
        rules.add(new LossLoweringRewrite());
        rules.add(new ReductionLoweringRewrite());
        if (resolved.conv2dLowering().mode() != Conv2dLoweringMode.OFF) {
            rules.add(new Conv2dLoweringRewrite(resolved.conv2dLowering()));
        }
        this.delegates = List.copyOf(rules);
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
