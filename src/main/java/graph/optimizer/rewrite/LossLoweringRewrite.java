package graph.optimizer.rewrite;

import graph.optimizer.OptimizationRule;
import graph.optimizer.state.OptimizerState;
import tensor.Tensor;

import java.util.List;

/**
 * Aggregate loss lowering stage that applies forward and backward loss lowering.
 */
public final class LossLoweringRewrite implements OptimizationRule {
    private final LossForwardLoweringRewrite forward = new LossForwardLoweringRewrite();
    private final LossBackwardLoweringRewrite backward = new LossBackwardLoweringRewrite();

    /**
     * Applies forward loss lowering followed by backward loss lowering.
     *
     * @param state optimizer state to rewrite
     * @return rewritten optimizer state
     */
    @Override
    public OptimizerState apply(OptimizerState state) {
        OptimizerState current = forward.apply(state);
        return backward.apply(current);
    }
}
