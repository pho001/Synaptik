package backend.cpu1.fused.ir;

import java.util.List;

public record Cpu1FusedExpressionPlan(
        List<Cpu1FusedNodePlan> nodes,
        List<Cpu1FusedInputPlan> inputs,
        int outputRef
) {
    public Cpu1FusedExpressionPlan {
        if (nodes == null) {
            throw new IllegalArgumentException("nodes cannot be null");
        }
        nodes = List.copyOf(nodes);
        if (inputs == null) {
            throw new IllegalArgumentException("inputs cannot be null");
        }
        inputs = List.copyOf(inputs);
        if (nodes.isEmpty()) {
            throw new IllegalArgumentException("nodes cannot be empty");
        }
    }

    public int inputCount() {
        return inputs.size();
    }

    public int nodeCount() {
        return nodes.size();
    }

    public Cpu1FusedNodePlan outputNode() {
        int nodeIndex = outputRef - inputCount();
        if (nodeIndex < 0 || nodeIndex >= nodes.size()) {
            throw new IllegalStateException("Fused outputRef does not point to an internal node: " + outputRef);
        }
        return nodes.get(nodeIndex);
    }

    public boolean usesOnlyLinearInputs() {
        for (Cpu1FusedInputPlan input : inputs) {
            if (!input.isLinearAccess()) {
                return false;
            }
        }
        return true;
    }
}
