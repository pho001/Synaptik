package graph.optimizer.memory;

import operations.Operation;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Collections;

public final class MemoryPlanner {
    private MemoryPlanner() {
    }

    public static MemoryPlan plan(List<Tensor> sortedGraph) {
        if (sortedGraph == null || sortedGraph.isEmpty()) {
            return new MemoryPlan(Map.of());
        }

        Map<Tensor, Integer> indexByTensor = new IdentityHashMap<>();
        for (int i = 0; i < sortedGraph.size(); i++) {
            indexByTensor.put(sortedGraph.get(i), i);
        }
        int forwardBoundaryIndex = resolveForwardBoundaryIndex(sortedGraph);

        Map<Tensor, Tensor> storageOwnerByTensor = new IdentityHashMap<>();
        for (Tensor tensor : sortedGraph) {
            storageOwnerByTensor.put(tensor, resolveStorageOwner(tensor, storageOwnerByTensor));
        }

        Set<Tensor> gradientTargets = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Tensor tensor : sortedGraph) {
            if (tensor.getOperation() == null && tensor.getRequiresGrad() && tensor.getGradient() != null) {
                gradientTargets.add(tensor.getGradient());
            }
        }

        Map<Tensor, Integer> birthIndexByOwner = new IdentityHashMap<>();
        Map<Tensor, Integer> lastReadIndexByOwner = new IdentityHashMap<>();
        Map<Tensor, Integer> consumerCountsByOwner = new IdentityHashMap<>();
        Set<Tensor> savedForwardOwners = Collections.newSetFromMap(new IdentityHashMap<>());

        for (Tensor tensor : sortedGraph) {
            Tensor owner = storageOwnerByTensor.get(tensor);
            birthIndexByOwner.merge(owner, indexByTensor.get(owner), Math::min);
            lastReadIndexByOwner.putIfAbsent(owner, -1);
            consumerCountsByOwner.putIfAbsent(owner, 0);
        }

        for (int i = 0; i < sortedGraph.size(); i++) {
            Tensor consumer = sortedGraph.get(i);
            List<Tensor> inputs = consumer.getPrevTensors();
            if (inputs == null || inputs.isEmpty()) {
                continue;
            }
            for (Tensor input : inputs) {
                Tensor owner = storageOwnerByTensor.get(input);
                if (owner == null) {
                    continue;
                }
                consumerCountsByOwner.merge(owner, 1, Integer::sum);
                lastReadIndexByOwner.merge(owner, i, Math::max);
                if (indexByTensor.get(owner) <= forwardBoundaryIndex && i > forwardBoundaryIndex) {
                    savedForwardOwners.add(owner);
                }
            }
        }

        for (Tensor owner : new ArrayList<>(lastReadIndexByOwner.keySet())) {
            if (consumerCountsByOwner.getOrDefault(owner, 0) == 0) {
                lastReadIndexByOwner.put(owner, Integer.MAX_VALUE);
            }
        }

        Map<Tensor, NodeLifetime> lifetimes = new IdentityHashMap<>();
        for (Tensor tensor : sortedGraph) {
            Tensor owner = storageOwnerByTensor.get(tensor);
            MemoryRole role = roleOf(tensor, owner, gradientTargets, savedForwardOwners);
            int birthIndex = indexByTensor.get(tensor);
            int lastReadIndex = lastReadIndexByOwner.getOrDefault(owner, Integer.MAX_VALUE);
            lifetimes.put(tensor, new NodeLifetime(birthIndex, lastReadIndex, role, owner));
        }

        return new MemoryPlan(lifetimes);
    }

    private static Tensor resolveStorageOwner(Tensor tensor, Map<Tensor, Tensor> storageOwnerByTensor) {
        if (!aliasesInput0AtRuntime(tensor)) {
            return tensor;
        }
        Tensor input0 = tensor.getPrevTensors().get(0);
        return storageOwnerByTensor.getOrDefault(input0, input0);
    }

    private static MemoryRole roleOf(
            Tensor tensor,
            Tensor owner,
            Set<Tensor> gradientTargets,
            Set<Tensor> savedForwardOwners
    ) {
        if (aliasesInput0AtRuntime(tensor)) {
            return MemoryRole.VIEW_ALIAS;
        }
        if (tensor.getOperation() == null) {
            return MemoryRole.LEAF;
        }
        if (gradientTargets.contains(tensor)) {
            return MemoryRole.GRADIENT_TARGET;
        }
        if (savedForwardOwners.contains(owner)) {
            return MemoryRole.SAVED_FORWARD;
        }
        if (tensor.isBackward()) {
            return MemoryRole.BACKWARD_TEMP;
        }
        return MemoryRole.FORWARD_TEMP;
    }

    private static int resolveForwardBoundaryIndex(List<Tensor> sortedGraph) {
        for (int i = sortedGraph.size() - 1; i >= 0; i--) {
            Tensor tensor = sortedGraph.get(i);
            if (tensor.getOperation() != null
                    && tensor.getOperation().opType() == Operation.OpType.NOOP
                    && Tensor.SYSTEM_FORWARD_OUTPUT_LABEL.equals(tensor.getLabel())) {
                return i;
            }
        }
        return sortedGraph.size() - 1;
    }

    private static boolean aliasesInput0AtRuntime(Tensor tensor) {
        if (tensor == null || tensor.getOperation() == null) {
            return false;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.isEmpty()) {
            return false;
        }
        return switch (tensor.getOperation().opType()) {
            case NOOP, PERMUTE -> true;
            case RESHAPE, EXPAND_DIMS, SQUEEZE -> inputs.get(0).isContiguous();
            default -> false;
        };
    }
}
