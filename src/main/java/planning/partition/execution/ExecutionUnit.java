package planning.partition.execution;

import planning.value.GraphValueRef;

import planning.partition.PartitionTarget;
import planning.partition.specialization.PartitionSpecializationCandidate;

import java.util.List;

/**
 * Executable unit inside an planned partition.
 *
 * <p>Units are the fusion-level work items handed from partition planning to memory planning and preparation. They
 * describe value dependencies, materialized and virtual outputs, node ids covered by the unit, and backend work
 * estimates.
 *
 * @param unitId stable unit id within the partition
 * @param kind execution unit kind
 * @param target backend target
 * @param inputValueRefs partition values consumed by the unit
 * @param outputValueRefs partition values produced by the unit
 * @param materializedOutputs outputs that must have runtime storage
 * @param virtualOutputs outputs represented only structurally
 * @param orderedNodeIds compiled node ids covered by this unit
 * @param estimatedWork backend work estimate
 * @param requiredPreparedInputNodeIds external graph node ids required as prepared inputs
 * @param trace partition planning diagnostics for this unit
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
        PartitionExecutionTrace trace,
        PartitionSpecializationCandidate specialization
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
            PartitionExecutionTrace trace
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
        trace = trace == null ? PartitionExecutionTrace.empty() : trace;
    }
}
