package graph.optimizer;

import graph.optimizer.state.OptimizerState;
import tensor.Tensor;

import java.util.List;

/**
 * Extension point for graph optimizer stages.
 *
 * <p>A rule receives an {@link OptimizerState} and must return a non-null state. Rules may rewrite the tensor graph,
 * update graph-level compile metadata, or add trace data. Backend ownership, region planning, and memory planning live
 * in {@code planning.backend}; optimizer rules should not attach compile-planning artifacts.
 *
 * <p>Rules are expected to be deterministic for the same input graph and configuration. Implementations that keep
 * diagnostic state should document their thread-safety separately.
 */
public interface OptimizationRule {
    /**
     * Applies this rule to an optimizer state.
     *
     * @param state optimizer state to transform
     * @return transformed state; never {@code null}
     */
    OptimizerState apply(OptimizerState state);

    /**
     * Applies this rule to a graph using the graph's last tensor as the forward output.
     *
     * @param sortedGraph tensors in topological order
     * @return optimized graph in topological order
     */
    default List<Tensor> apply(List<Tensor> sortedGraph) {
        return apply(OptimizerState.ofGraph(sortedGraph)).graph();
    }
}
