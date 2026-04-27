package graph.optimizer;

import graph.optimizer.state.OptimizerState;
import tensor.Tensor;

import java.util.List;

public interface OptimizationRule {
    OptimizerState apply(OptimizerState state);

    default List<Tensor> apply(List<Tensor> sortedGraph) {
        return apply(OptimizerState.ofGraph(sortedGraph)).graph();
    }
}
