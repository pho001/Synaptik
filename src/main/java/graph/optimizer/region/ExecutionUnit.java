package graph.optimizer.region;

import graph.optimizer.partition.PartitionTarget;

import java.util.List;

public record ExecutionUnit(
        String unitId,
        ExecutionUnitKind kind,
        PartitionTarget target,
        List<RegionValueRef> inputValueRefs,
        List<RegionValueRef> outputValueRefs,
        List<RegionValueRef> materializedOutputs,
        List<RegionValueRef> virtualOutputs,
        List<Integer> orderedNodeIds,
        long estimatedWork,
        List<Integer> requiredPreparedInputNodeIds,
        RegionOptimizationTrace trace
) {
    public ExecutionUnit {
        if (unitId == null || unitId.isBlank()) {
            throw new IllegalArgumentException("unitId cannot be blank");
        }
        if (kind == null || target == null) {
            throw new IllegalArgumentException("kind and target cannot be null");
        }
        inputValueRefs = List.copyOf(inputValueRefs == null ? List.of() : inputValueRefs);
        outputValueRefs = List.copyOf(outputValueRefs == null ? List.of() : outputValueRefs);
        materializedOutputs = List.copyOf(materializedOutputs == null ? List.of() : materializedOutputs);
        virtualOutputs = List.copyOf(virtualOutputs == null ? List.of() : virtualOutputs);
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        requiredPreparedInputNodeIds = List.copyOf(requiredPreparedInputNodeIds == null ? List.of() : requiredPreparedInputNodeIds);
        estimatedWork = Math.max(0L, estimatedWork);
        trace = trace == null ? RegionOptimizationTrace.empty() : trace;
    }
}
