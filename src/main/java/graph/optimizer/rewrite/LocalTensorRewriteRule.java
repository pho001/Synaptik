package graph.optimizer.rewrite;

import graph.optimizer.OptimizationRule;
import graph.optimizer.OptimizerGraph;
import graph.optimizer.state.OptimizerState;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Base class for one-node-at-a-time tensor rewrites.
 *
 * <p>The template method rewrites inputs through prior replacements, delegates the local decision to
 * {@link #rewriteTensor(Tensor)}, preserves backward markings, fixes gradient references, and rebuilds the observable
 * closure by default. Subclasses only decide whether a tensor should be replaced.
 */
public abstract class LocalTensorRewriteRule implements OptimizationRule {
    /**
     * Applies the rewrite to every tensor in topological order.
     *
     * @param state optimizer state to rewrite
     * @return state with the rewritten graph and preserved graph-level optimizer metadata
     */
    @Override
    public final OptimizerState apply(OptimizerState state) {
        List<Tensor> sortedGraph = state.graph();
        List<Tensor> originalRoots = OptimizerGraph.observableRoots(sortedGraph);
        List<Tensor> optimized = new ArrayList<>();
        Map<Tensor, Tensor> replacements = new HashMap<>();

        for (Tensor tensor : sortedGraph) {
            OptimizerGraph.rewriteInputs(tensor, replacements);
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
                Tensor resolvedGradient = OptimizerGraph.resolveReplacement(tensor.getGradient(), replacements);
                if (resolvedGradient != null) {
                    TensorInternalAccess.setGradient(tensor, resolvedGradient);
                }
            }
        }

        if (!rebuildClosure()) {
            Tensor resolvedForwardOutput = OptimizerGraph.resolveReplacement(state.forwardOutput(), replacements);
            return state.withGraph(optimized, resolvedForwardOutput == null ? state.forwardOutput() : resolvedForwardOutput)
                    .withRewriteMap(state.rewriteMap().withReplacements(replacements));
        }
        Tensor resolvedForwardOutput = OptimizerGraph.resolveReplacement(state.forwardOutput(), replacements);
        List<Tensor> rebuilt = OptimizerGraph.rebuildTopologicalClosureFromRoots(
                OptimizerGraph.resolveRoots(originalRoots, replacements)
        );
        return state.withGraph(rebuilt, resolvedForwardOutput == null ? state.forwardOutput() : resolvedForwardOutput)
                .withRewriteMap(state.rewriteMap().withReplacements(replacements));
    }

    /**
     * Controls whether the optimized graph should be rebuilt from observable roots after local rewrites.
     *
     * @return {@code true} to remove unreachable intermediates after replacement
     */
    protected boolean rebuildClosure() {
        return true;
    }

    /**
     * Rewrites a single tensor after its inputs have already been rewritten.
     *
     * @param tensor tensor to inspect
     * @return either {@code tensor} to keep it or a replacement tensor with equivalent semantics
     */
    protected abstract Tensor rewriteTensor(Tensor tensor);
}
