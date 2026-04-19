package graph.optimizer.rewrite;

import graph.optimizer.OptimizationRule;
import tensor.Tensor;

import java.util.List;

public final class LossLoweringRewrite implements OptimizationRule {
    private final LossForwardLoweringRewrite forward = new LossForwardLoweringRewrite();
    private final LossBackwardLoweringRewrite backward = new LossBackwardLoweringRewrite();

    @Override
    public List<Tensor> apply(List<Tensor> sortedGraph) {
        List<Tensor> current = forward.apply(sortedGraph);
        return backward.apply(current);
    }
}
