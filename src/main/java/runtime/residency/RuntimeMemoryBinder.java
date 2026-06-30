package runtime.residency;

import planning.memory.MemoryPlan;
import planning.memory.RegionMemoryBinding;
import planning.memory.RegionMemoryBindingKind;
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
            if (!memoryPlan.runtimeBindingPolicyOfNodeId(node.id()).regionBindingAllowed()) {
                continue;
            }
            Tensor runtimeTensor = executionState.runtimeTensorForNodeId(node.id());
            if (bindAliasView(node, executionState, runtimeTensor)) {
                continue;
            }
            tryBindRegionMapped(
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

    private static boolean tryBindRegionMapped(
            Tensor runtimeTensor,
            int nodeId,
            MemoryPlan memoryPlan,
            ExecutionState executionState
    ) {
        GraphValueRef valueRef = memoryPlan.graphValueRefOfNodeId(nodeId);
        if (valueRef == null) {
            return false;
        }
        RegionMemoryBinding binding = memoryPlan.regionMemoryBindingOf(valueRef);
        if (binding.kind() == RegionMemoryBindingKind.NONE) {
            return false;
        }
        Integer slotId = memoryPlan.regionSlotIdOf(valueRef);
        if (slotId == null) {
            return false;
        }
        int slotSize = memoryPlan.regionSlotSize(slotId);
        if (slotSize != runtimeTensor.getFlatDataSize()) {
            return false;
        }
        RuntimeStorageSlotKey slotKey = executionState.registerRegionRuntimeStorageSlot(
                nodeId,
                runtimeTensor.getDataType(),
                slotId,
                slotSize
        );
        if (memoryPlan.regionSlotUseCount(slotId) < 2) {
            return false;
        }
        executionState.bindJavaStorageSlot(nodeId, slotKey);
        return true;
    }
}
