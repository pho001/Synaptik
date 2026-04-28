package backend.cpu.fused.codegen;

import java.util.List;
import java.util.Objects;

/**
 * Internal lowered expression graph for fused CPU code generation.
 */
public record FusedExpressionPlan(
        List<FusedNodePlan> nodes,
        List<FusedExternalInputPlan> inputs,
        int outputRef
) {
    public FusedExpressionPlan {
        Objects.requireNonNull(nodes, "nodes cannot be null");
        Objects.requireNonNull(inputs, "inputs cannot be null");
        nodes = List.copyOf(nodes);
        inputs = List.copyOf(inputs);

        if (nodes.isEmpty()) {
            throw new IllegalArgumentException("nodes cannot be empty");
        }
        if (outputRef < inputs.size()) {
            throw new IllegalArgumentException("outputRef must refer to a fused node result");
        }
    }

    public int nodeCount() {
        return nodes.size();
    }

    public int inputCount() {
        return inputs.size();
    }

    public int totalValueCount() {
        return inputs.size() + nodes.size();
    }

    public FusedNodePlan outputNode() {
        int nodeIndex = outputRef - inputs.size();
        if (nodeIndex < 0 || nodeIndex >= nodes.size()) {
            throw new IllegalStateException("outputRef does not point to a fused node");
        }
        return nodes.get(nodeIndex);
    }
}
