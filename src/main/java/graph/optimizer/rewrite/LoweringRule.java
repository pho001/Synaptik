package graph.optimizer.rewrite;

import config.optimizer.Conv2dLoweringMode;
import config.optimizer.RewriteConfig;
import graph.optimizer.OptimizationRule;
import graph.optimizer.state.OptimizerState;

import java.util.ArrayList;
import java.util.List;

/**
 * Backend-neutral lowering stage that runs after cleanup and before partitioning.
 *
 * <p>This stage must preserve the canonical Tensor DAG contract. Do not add a
 * default rule here that collapses canonical primitive backward graphs
 * (softmax, log-softmax, min/max, gather, slice, and similar gradients) back
 * into legacy {@code *_GRAD} operation descriptors. Those descriptors are kept
 * as an explicit backend/CPU specialization surface and direct-test surface;
 * they are not the semantic default produced by public Tensor APIs.</p>
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
