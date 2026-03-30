package graph.optimizer;

import tensor.Tensor;

import java.util.List;

public interface OptimizationRule {
    List<Tensor> apply(List<Tensor> sortedGraph);
}