package graph.compile.planning.memory;

import graph.CompiledNode;
import operations.Operation;
import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class RuntimeMemoryBindingPolicyPlanner {
    private RuntimeMemoryBindingPolicyPlanner() {
    }

    static Map<Tensor, RuntimeMemoryBindingPolicy> forTensors(List<Tensor> sortedGraph) {
        IdentityHashMap<Tensor, RuntimeMemoryBindingPolicy> policies = new IdentityHashMap<>();
        for (Tensor tensor : sortedGraph) {
            Operation operation = tensor.getOperation();
            if (operation == null || operation.opType() == null) {
                policies.put(tensor, RuntimeMemoryBindingPolicy.REGION_BINDING_ALLOWED);
                continue;
            }
            policies.put(tensor, policyFor(operation));
        }
        return Map.copyOf(policies);
    }

    static Map<Integer, RuntimeMemoryBindingPolicy> forNodeIds(List<CompiledNode> nodes) {
        LinkedHashMap<Integer, RuntimeMemoryBindingPolicy> policies = new LinkedHashMap<>();
        for (CompiledNode node : nodes) {
            Operation operation = node.operation();
            policies.put(node.id(), operation == null ? RuntimeMemoryBindingPolicy.REGION_BINDING_ALLOWED : policyFor(operation));
        }
        return Map.copyOf(policies);
    }

    private static RuntimeMemoryBindingPolicy policyFor(Operation operation) {
        return switch (operation.opType()) {
            case MAX_POOL2D -> RuntimeMemoryBindingPolicy.skip("workspace-sensitive-storage");
            default -> RuntimeMemoryBindingPolicy.REGION_BINDING_ALLOWED;
        };
    }
}
