package graph.optimizer.rewrite;

import graph.optimizer.OptimizationRule;
import graph.optimizer.OptimizerGraphSupport;
import graph.optimizer.state.OptimizerState;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

abstract class AbstractRewriteRule implements OptimizationRule {
    @Override
    public final OptimizerState apply(OptimizerState state) {
        List<Tensor> sortedGraph = state.graph();
        List<Tensor> originalRoots = OptimizerGraphSupport.observableRoots(sortedGraph);
        List<Tensor> optimized = new ArrayList<>();
        Map<Tensor, Tensor> replacements = new HashMap<>();

        for (Tensor tensor : sortedGraph) {
            OptimizerGraphSupport.rewriteInputs(tensor, replacements);
            Tensor rewritten = rewriteTensor(tensor);
            if (rewritten != tensor) {
                if (tensor.isBackward()) {
                    TensorInternalAccess.setBackward(rewritten, true);
                }
                replacements.put(tensor, rewritten);
                optimized.add(rewritten);
            } else {
                optimized.add(tensor);
            }
        }

        if (!replacements.isEmpty()) {
            for (Tensor tensor : sortedGraph) {
                Tensor resolvedGradient = OptimizerGraphSupport.resolveReplacement(tensor.getGradient(), replacements);
                if (resolvedGradient != null) {
                    TensorInternalAccess.setGradient(tensor, resolvedGradient);
                }
            }
        }

        if (!rebuildClosure()) {
            Tensor resolvedForwardOutput = OptimizerGraphSupport.resolveReplacement(state.forwardOutput(), replacements);
            return state.withGraph(optimized, resolvedForwardOutput == null ? state.forwardOutput() : resolvedForwardOutput);
        }
        Tensor resolvedForwardOutput = OptimizerGraphSupport.resolveReplacement(state.forwardOutput(), replacements);
        List<Tensor> rebuilt = OptimizerGraphSupport.rebuildTopologicalClosureFromRoots(
                OptimizerGraphSupport.resolveRoots(originalRoots, replacements)
        );
        return state.withGraph(rebuilt, resolvedForwardOutput == null ? state.forwardOutput() : resolvedForwardOutput);
    }

    protected boolean rebuildClosure() {
        return true;
    }

    protected abstract Tensor rewriteTensor(Tensor tensor);
}
