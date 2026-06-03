package graph.compile.planning.region;

import graph.compile.planning.value.GraphValueRef;

import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.region.specialization.RegionSpecializationCandidate;

import java.util.List;

/**
 * Executable unit inside an optimized region.
 *
 * <p>Units are the fusion-level work items handed from region optimization to memory planning and preparation. They
 * describe value dependencies, materialized and virtual outputs, node ids covered by the unit, and backend work
 * estimates.
 *
 * @param unitId stable unit id within the region
 * @param kind execution unit kind
 * @param target backend target
 * @param inputValueRefs region values consumed by the unit
 * @param outputValueRefs region values produced by the unit
 * @param materializedOutputs outputs that must have runtime storage
 * @param virtualOutputs outputs represented only structurally
 * @param orderedNodeIds compiled node ids covered by this unit
 * @param estimatedWork backend work estimate
 * @param requiredPreparedInputNodeIds external graph node ids required as prepared inputs
 * @param trace region optimization diagnostics for this unit
 * @param specialization accepted graph-level specialization, or {@code null} for structural units
 */
public record ExecutionUnit(
        String unitId,
        ExecutionUnitKind kind,
        PartitionTarget target,
        List<GraphValueRef> inputValueRefs,
        List<GraphValueRef> outputValueRefs,
        List<GraphValueRef> materializedOutputs,
        List<GraphValueRef> virtualOutputs,
        List<Integer> orderedNodeIds,
        long estimatedWork,
        List<Integer> requiredPreparedInputNodeIds,
        RegionOptimizationTrace trace,
        RegionSpecializationCandidate specialization
) {
    public ExecutionUnit(
            String unitId,
            ExecutionUnitKind kind,
            PartitionTarget target,
            List<GraphValueRef> inputValueRefs,
            List<GraphValueRef> outputValueRefs,
            List<GraphValueRef> materializedOutputs,
            List<GraphValueRef> virtualOutputs,
            List<Integer> orderedNodeIds,
            long estimatedWork,
            List<Integer> requiredPreparedInputNodeIds,
            RegionOptimizationTrace trace
    ) {
        this(
                unitId,
                kind,
                target,
                inputValueRefs,
                outputValueRefs,
                materializedOutputs,
                virtualOutputs,
                orderedNodeIds,
                estimatedWork,
                requiredPreparedInputNodeIds,
                trace,
                null
        );
    }

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
