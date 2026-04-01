package graph.codegen;

import java.util.List;
import java.util.Objects;

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
}
