package graph.execution.residency;

import graph.AliasViewPolicy;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.execution.state.ExecutionState;
import graph.execution.state.RuntimeStorageSlotKey;
import graph.compile.planning.memory.MemoryPlan;
import graph.compile.planning.memory.RegionMemoryBinding;
import graph.compile.planning.memory.RegionMemoryBindingKind;
import graph.compile.planning.value.GraphValueRef;
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
            CompiledTensorDescriptorIndex descriptorIndex,
            ExecutionState executionState
    ) {
        if (memoryPlan == null || compiledNodes == null || compiledNodes.isEmpty()) {
            return;
        }
        Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        Objects.requireNonNull(executionState, "executionState cannot be null");
        for (CompiledNode node : compiledNodes) {
            if (node.operation() == null) {
                continue;
            }
            if (!memoryPlan.runtimeBindingPolicyOfNodeId(node.id()).regionBindingAllowed()) {
                continue;
            }
            Tensor runtimeTensor = executionState.runtimeTensorForNodeId(node.id());
            if (aliasesInput0AtRuntime(node, descriptorIndex, executionState, runtimeTensor)) {
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

    private static boolean aliasesInput0AtRuntime(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            ExecutionState executionState,
            Tensor runtimeTensor
    ) {
        if (node == null || node.operation() == null || node.inputIds().isEmpty()) {
            return false;
        }
        if (!AliasViewPolicy.aliasesInput0AtRuntime(node, descriptorIndex)) {
            return false;
        }
        Tensor sourceRuntime = executionState.runtimeTensorForNodeId(node.inputIds().getFirst());
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
