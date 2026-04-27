package graph.execution;

import graph.CompiledNode;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.memory.RegionMemoryBinding;
import graph.optimizer.memory.RegionMemoryBindingKind;
import graph.optimizer.region.RegionValueRef;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

final class RuntimeMemoryBinder {
    private RuntimeMemoryBinder() {
    }

    static void bind(
            MemoryPlan memoryPlan,
            List<CompiledNode> compiledNodes,
            ExecutionState executionState
    ) {
        if (memoryPlan == null || compiledNodes == null || compiledNodes.isEmpty() || executionState == null) {
            return;
        }
        Map<Integer, double[]> regionF64Slots = new HashMap<>();
        Map<Integer, float[]> regionF32Slots = new HashMap<>();
        Map<Tensor, Tensor> runtimeTensorBySemanticTensor = new IdentityHashMap<>(compiledNodes.size());
        for (CompiledNode node : compiledNodes) {
            runtimeTensorBySemanticTensor.put(node.semanticTensor(), executionState.runtimeTensorForNodeId(node.id()));
        }

        for (CompiledNode node : compiledNodes) {
            if (node.operation() == null) {
                continue;
            }
            if (!memoryPlan.runtimeBindingPolicyOf(node.semanticTensor()).regionBindingAllowed()) {
                continue;
            }
            Tensor semanticTensor = node.semanticTensor();
            Tensor runtimeTensor = executionState.runtimeTensorForNodeId(node.id());
            if (aliasesInput0AtRuntime(node, runtimeTensorBySemanticTensor, runtimeTensor)) {
                continue;
            }
            tryBindRegionMapped(runtimeTensor, semanticTensor, memoryPlan, regionF64Slots, regionF32Slots);
        }
    }

    private static boolean aliasesInput0AtRuntime(
            CompiledNode node,
            Map<Tensor, Tensor> runtimeTensorBySemanticTensor,
            Tensor runtimeTensor
    ) {
        if (node == null || node.operation() == null || node.inputTensors().isEmpty()) {
            return false;
        }
        boolean aliases = switch (node.operation().opType()) {
            case NOOP, EXPAND, SELECT, PERMUTE, EXPAND_DIMS, SQUEEZE -> true;
            case RESHAPE -> node.inputTensors().getFirst().isContiguous();
            default -> false;
        };
        if (!aliases) {
            return false;
        }
        Tensor sourceSemantic = node.inputTensors().getFirst();
        Tensor sourceRuntime = runtimeTensorBySemanticTensor.get(sourceSemantic);
        if (sourceRuntime != null) {
            TensorInternalAccess.aliasRuntimeFrom(runtimeTensor, sourceRuntime);
        }
        return true;
    }

    private static boolean tryBindRegionMapped(
            Tensor runtimeTensor,
            Tensor semanticTensor,
            MemoryPlan memoryPlan,
            Map<Integer, double[]> f64Slots,
            Map<Integer, float[]> f32Slots
    ) {
        RegionValueRef valueRef = memoryPlan.regionValueRefOf(semanticTensor);
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
        if (memoryPlan.regionSlotUseCount(slotId) < 2) {
            return false;
        }
        int slotSize = memoryPlan.regionSlotSize(slotId);
        if (slotSize != runtimeTensor.getFlatDataSize()) {
            return false;
        }
        bindTypedStorage(runtimeTensor, slotId, slotSize, f64Slots, f32Slots);
        return true;
    }

    private static void bindTypedStorage(
            Tensor runtimeTensor,
            int slotId,
            int slotSize,
            Map<Integer, double[]> f64Slots,
            Map<Integer, float[]> f32Slots
    ) {
        switch (runtimeTensor.getDataType()) {
            case FLOAT64 -> {
                double[] buffer = f64Slots.computeIfAbsent(slotId, ignored -> new double[slotSize]);
                runtimeTensor.setData(buffer);
            }
            case FLOAT32 -> {
                float[] buffer = f32Slots.computeIfAbsent(slotId, ignored -> new float[slotSize]);
                runtimeTensor.setFloat32Data(buffer);
            }
            case BFLOAT16, INT32, BOOL -> {
                // Current runtime binding remains a no-op for these dtypes.
            }
        }
    }
}
