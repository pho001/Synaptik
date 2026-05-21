package graph.optimizer.simplify;

import graph.optimizer.OptimizationRule;
import graph.optimizer.OptimizerGraph;
import graph.optimizer.state.OptimizerState;
import tensor.Tensor;

import java.util.LinkedHashSet;
import java.util.List;

/**
 * Removes nodes that are not reachable from compile-observable outputs.
 */
public final class DeadCodeEliminationRule implements OptimizationRule {
    @Override
    public OptimizerState apply(OptimizerState state) {
        List<Tensor> roots = roots(state);
        List<Tensor> rebuilt = OptimizerGraph.rebuildTopologicalClosureFromRoots(roots);
        return state.withGraph(rebuilt, state.forwardOutput());
    }

    private static List<Tensor> roots(OptimizerState state) {
        LinkedHashSet<Tensor> roots = new LinkedHashSet<>();
        roots.add(state.forwardOutput());
        for (Tensor tensor : state.graph()) {
            Tensor gradient = tensor.getGradient();
            if (gradient != null && gradient.getOperation() != null) {
                roots.add(gradient);
            }
        }
        return List.copyOf(roots);
    }
}
