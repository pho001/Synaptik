package training.optimizer;

import graph.CompiledGradientBinding;
import graph.CompiledNode;

/**
 * Compiled trainable parameter and its gradient binding.
 */
public record TrainableParameterRef(
        CompiledNode parameterNode,
        CompiledGradientBinding gradientBinding
) {
}
