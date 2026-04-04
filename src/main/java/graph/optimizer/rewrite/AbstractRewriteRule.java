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

        return rebuildClosure() ? OptimizerGraphSupport.rebuildTopologicalClosure(optimized) : optimized;
    }

    protected boolean rebuildClosure() {
        return true;
    }

    protected abstract Tensor rewriteTensor(Tensor tensor);
}
