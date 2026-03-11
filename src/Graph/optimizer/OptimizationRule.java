package Graph.optimizer;

import Tensor.Tensor;

import java.util.List;

public interface OptimizationRule {
    List<Tensor> apply(List<Tensor> sortedGraph);
}