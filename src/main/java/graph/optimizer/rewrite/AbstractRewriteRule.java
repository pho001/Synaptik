package graph.optimizer.rewrite;

import graph.optimizer.OptimizationRule;
import graph.optimizer.OptimizerGraphSupport;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

abstract class AbstractRewriteRule implements OptimizationRule {
    @Override
    public final List<Tensor> apply(List<Tensor> sortedGraph) {
        List<Tensor> originalSinks = OptimizerGraphSupport.consumerFreeSinks(sortedGraph);
        List<Tensor> optimized = new ArrayList<>();
        Map<Tensor, Tensor> replacements = new HashMap<>();

        for (Tensor tensor : sortedGraph) {
            OptimizerGraphSupport.rewriteInputs(tensor, replacements);
            Tensor rewritten = rewriteTensor(tensor);
            if (rewritten != tensor) {
                if (tensor.isBackward()) {
                    rewritten.setBackward(true);
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
                    tensor.setGradient(resolvedGradient);
                }
            }
        }

        if (!rebuildClosure()) {
            return optimized;
        }
        List<Tensor> resolvedRoots = new ArrayList<>(originalSinks.size());
        for (Tensor sink : originalSinks) {
            Tensor resolved = OptimizerGraphSupport.resolveReplacement(sink, replacements);
            resolvedRoots.add(resolved == null ? sink : resolved);
        }
        return OptimizerGraphSupport.rebuildTopologicalClosureFromRoots(resolvedRoots);
    }

    protected boolean rebuildClosure() {
        return true;
    }

    protected abstract Tensor rewriteTensor(Tensor tensor);
}
