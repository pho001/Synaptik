package training.optimizer;

import graph.model.CompiledGradientBinding;
import graph.model.CompiledNode;
import tensor.Tensor;

/**
 * Compiled trainable parameter and its gradient binding.
 */
public record TrainableParameterRef(
        CompiledNode parameterNode,
        Tensor parameterTensor,
        CompiledGradientBinding gradientBinding
) {
}
