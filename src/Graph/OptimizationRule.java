package Graph;

import Tensor.Tensor;

import java.util.List;

public interface OptimizationRule {
    List<Tensor> apply(Tensor vertex);
}



