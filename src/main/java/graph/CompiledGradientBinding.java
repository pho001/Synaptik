package graph;

import tensor.Tensor;

import java.util.Objects;

/**
 * Immutable binding from a semantic tensor to the gradient value that should be published after a
 * run. The value can come either from a compiled runtime node or from a detached constant template
 * when optimization eliminated the explicit gradient node from the executable graph.
 */
public sealed interface CompiledGradientBinding permits CompiledGradientBinding.NodeBinding, CompiledGradientBinding.ConstantBinding {
    static CompiledGradientBinding node(int nodeId) {
        return new NodeBinding(nodeId);
    }

    static CompiledGradientBinding constant(Tensor template) {
        Objects.requireNonNull(template, "template cannot be null");
        Tensor copy = new Tensor(template.getShape(), null, template.getLabel(), template.getDataType());
        copy.copyDataFrom(template);
        return new ConstantBinding(copy);
    }

    record NodeBinding(int nodeId) implements CompiledGradientBinding {
        public NodeBinding {
            if (nodeId < 0) {
                throw new IllegalArgumentException("nodeId must be >= 0");
            }
        }
    }

    record ConstantBinding(Tensor template) implements CompiledGradientBinding {
        public ConstantBinding {
            Objects.requireNonNull(template, "template cannot be null");
        }
    }
}
