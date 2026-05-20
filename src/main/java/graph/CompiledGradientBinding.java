package graph;

import tensor.Tensor;

import java.util.Objects;

/**
 * Immutable binding from a publication tensor to the gradient value that should be attached after a run.
 * The value can come either from a compiled runtime node or from a detached constant template when
 * optimization eliminated the explicit gradient node from the executable graph.
 */
public sealed interface CompiledGradientBinding permits CompiledGradientBinding.NodeBinding, CompiledGradientBinding.ConstantBinding {
    /**
     * Creates a binding to a compiled node output.
     *
     * @param nodeId compiled node id that produces the gradient
     * @return node gradient binding
     */
    static CompiledGradientBinding node(int nodeId) {
        return new NodeBinding(nodeId);
    }

    /**
     * Creates a binding to a detached constant gradient template.
     *
     * @param template tensor template to copy after execution
     * @return constant gradient binding
     */
    static CompiledGradientBinding constant(Tensor template) {
        Objects.requireNonNull(template, "template cannot be null");
        Tensor copy = new Tensor(template.getShape(), null, template.getLabel(), template.getDataType());
        copy.copyDataFrom(template);
        return new ConstantBinding(copy);
    }

    /**
     * Gradient binding to a runtime node output.
     *
     * @param nodeId compiled node id
     */
    record NodeBinding(int nodeId) implements CompiledGradientBinding {
        public NodeBinding {
            if (nodeId < 0) {
                throw new IllegalArgumentException("nodeId must be >= 0");
            }
        }
    }

    /**
     * Gradient binding to a constant tensor template.
     *
     * @param template detached tensor template copied during publication
     */
    record ConstantBinding(Tensor template) implements CompiledGradientBinding {
        public ConstantBinding {
            Objects.requireNonNull(template, "template cannot be null");
        }
    }
}
