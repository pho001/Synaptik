package runtime.residency;

import planning.memory.MemoryPlan;
import planning.memory.PartitionMemoryBinding;
import planning.memory.PartitionMemoryBindingKind;
import planning.value.GraphValueRef;
import runtime.execution.ExecutionState;
import runtime.state.RuntimeStorageSlotKey;
import graph.model.CompiledNode;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.List;
import java.util.Objects;

public final class RuntimeMemoryBinder {
    private RuntimeMemoryBinder() {
    }

    public static void bind(
            MemoryPlan memoryPlan,
            List<CompiledNode> compiledNodes,
            ExecutionState executionState
    ) {
        if (memoryPlan == null || compiledNodes == null || compiledNodes.isEmpty()) {
            return;
        }
        Objects.requireNonNull(executionState, "executionState cannot be null");
        for (CompiledNode node : compiledNodes) {
            if (node.operation() == null) {
                continue;
            }
            if (!memoryPlan.runtimeBindingPolicyOfNodeId(node.id()).partitionBindingAllowed()) {
                continue;
            }
            Tensor runtimeTensor = executionState.runtimeTensorForNodeId(node.id());
            if (bindAliasView(node, executionState, runtimeTensor)) {
                continue;
            }
            tryBindPartitionMapped(
                    runtimeTensor,
                    node.id(),
                    memoryPlan,
                    executionState
            );
        }
    }

    private static boolean bindAliasView(
            CompiledNode node,
            ExecutionState executionState,
            Tensor runtimeTensor
    ) {
        if (node == null || node.storageOwnerId() == node.id()) {
            return false;
        }
        Tensor sourceRuntime = executionState.runtimeTensorForNodeId(node.storageOwnerId());
        if (sourceRuntime != null) {
            TensorInternalAccess.aliasRuntimeFrom(runtimeTensor, sourceRuntime);
        }
        return true;
    }

    private static boolean tryBindPartitionMapped(
            Tensor runtimeTensor,
            int nodeId,
            MemoryPlan memoryPlan,
            ExecutionState executionState
    ) {
        GraphValueRef valueRef = memoryPlan.graphValueRefOfNodeId(nodeId);
        if (valueRef == null) {
            return false;
        }
        PartitionMemoryBinding binding = memoryPlan.partitionMemoryBindingOf(valueRef);
        if (binding.kind() == PartitionMemoryBindingKind.NONE) {
            return false;
        }
        Integer slotId = memoryPlan.partitionSlotIdOf(valueRef);
        if (slotId == null) {
            return false;
        }
        int slotSize = memoryPlan.partitionSlotSize(slotId);
        if (slotSize != runtimeTensor.getFlatDataSize()) {
            return false;
        }
        RuntimeStorageSlotKey slotKey = executionState.registerPartitionRuntimeStorageSlot(
                nodeId,
                runtimeTensor.getDataType(),
                slotId,
                slotSize
        );
        if (memoryPlan.partitionSlotUseCount(slotId) < 2) {
            return false;
        }
        executionState.bindJavaStorageSlot(nodeId, slotKey);
        return true;
    }
}
