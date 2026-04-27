package graph.optimizer.rewrite;

import graph.optimizer.OptimizationRule;
import graph.optimizer.state.OptimizerState;
import tensor.Tensor;

import java.util.List;

public final class LossLoweringRewrite implements OptimizationRule {
    private final LossForwardLoweringRewrite forward = new LossForwardLoweringRewrite();
    private final LossBackwardLoweringRewrite backward = new LossBackwardLoweringRewrite();

    @Override
    public OptimizerState apply(OptimizerState state) {
        OptimizerState current = forward.apply(state);
        return backward.apply(current);
    }
}
