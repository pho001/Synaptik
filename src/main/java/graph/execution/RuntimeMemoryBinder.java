package graph.execution;

import graph.AliasViewPolicy;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.memory.RegionMemoryBinding;
import graph.optimizer.memory.RegionMemoryBindingKind;
import graph.optimizer.GraphValueRef;
import tensor.storage.BFloat16Storage;
import tensor.storage.BoolStorage;
import tensor.DataType;
import tensor.storage.Int32Storage;
import tensor.storage.Int64Storage;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class RuntimeMemoryBinder {
    private RuntimeMemoryBinder() {
    }

    static void bind(
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
        Map<Integer, double[]> regionF64Slots = new HashMap<>();
        Map<Integer, float[]> regionF32Slots = new HashMap<>();
        Map<Integer, short[]> regionBF16Slots = new HashMap<>();
        Map<Integer, int[]> regionI32Slots = new HashMap<>();
        Map<Integer, long[]> regionI64Slots = new HashMap<>();
        Map<Integer, byte[]> regionBoolSlots = new HashMap<>();
        for (CompiledNode node : compiledNodes) {
            if (node.operation() == null) {
                continue;
            }
            if (!memoryPlan.runtimeBindingPolicyOf(node.semanticTensor()).regionBindingAllowed()) {
                continue;
            }
            Tensor semanticTensor = node.semanticTensor();
            Tensor runtimeTensor = executionState.runtimeTensorForNodeId(node.id());
            if (aliasesInput0AtRuntime(node, descriptorIndex, executionState, runtimeTensor)) {
                continue;
            }
            tryBindRegionMapped(
                    runtimeTensor,
                    semanticTensor,
                    memoryPlan,
                    regionF64Slots,
                    regionF32Slots,
                    regionBF16Slots,
                    regionI32Slots,
                    regionI64Slots,
                    regionBoolSlots
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
            Tensor semanticTensor,
            MemoryPlan memoryPlan,
            Map<Integer, double[]> f64Slots,
            Map<Integer, float[]> f32Slots,
            Map<Integer, short[]> bf16Slots,
            Map<Integer, int[]> i32Slots,
            Map<Integer, long[]> i64Slots,
            Map<Integer, byte[]> boolSlots
    ) {
        GraphValueRef valueRef = memoryPlan.graphValueRefOf(semanticTensor);
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
        bindTypedStorage(runtimeTensor, slotId, slotSize, f64Slots, f32Slots, bf16Slots, i32Slots, i64Slots, boolSlots);
        return true;
    }

    private static void bindTypedStorage(
            Tensor runtimeTensor,
            int slotId,
            int slotSize,
            Map<Integer, double[]> f64Slots,
            Map<Integer, float[]> f32Slots,
            Map<Integer, short[]> bf16Slots,
            Map<Integer, int[]> i32Slots,
            Map<Integer, long[]> i64Slots,
            Map<Integer, byte[]> boolSlots
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
            case BFLOAT16 -> {
                short[] buffer = bf16Slots.computeIfAbsent(slotId, ignored -> new short[slotSize]);
                TensorInternalAccess.replaceStorage(runtimeTensor, new BFloat16Storage(buffer));
            }
            case INT32 -> {
                int[] buffer = i32Slots.computeIfAbsent(slotId, ignored -> new int[slotSize]);
                TensorInternalAccess.replaceStorage(runtimeTensor, new Int32Storage(buffer));
            }
            case INT64 -> {
                long[] buffer = i64Slots.computeIfAbsent(slotId, ignored -> new long[slotSize]);
                TensorInternalAccess.replaceStorage(runtimeTensor, new Int64Storage(buffer));
            }
            case BOOL -> {
                byte[] buffer = boolSlots.computeIfAbsent(slotId, ignored -> new byte[slotSize]);
                TensorInternalAccess.replaceStorage(runtimeTensor, new BoolStorage(buffer));
            }
        }
    }
}
