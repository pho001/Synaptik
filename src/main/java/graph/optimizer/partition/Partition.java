package graph.optimizer.partition;

import graph.execution.trace.PartitionDecisionTrace;
import graph.optimizer.partition.cost.AcceleratorPartitionScoreModel;

import java.util.List;

/**
 * Accepted backend partition in a compiled graph.
 *
 * <p>A partition groups ordered node ids for one {@link PartitionTarget}, records value boundaries, structural scoring
 * inputs, and carries the debug decision that accepted or rejected the candidate. Region optimization consumes accepted
 * partitions to build execution units and memory planning consumes value boundary metadata.
 *
 * @param partitionId stable identifier for this partition
 * @param regionKind semantic region kind
 * @param target backend target selected for the partition
 * @param orderedNodeIds graph node ids included in execution order
 * @param values values produced inside the partition
 * @param internalEdges producer-consumer edges wholly inside the partition
 * @param externalInputNodeIds producer node ids outside the partition that feed it
 * @param outputValueRefs values that leave the partition
 * @param anchorSeedNodeId node id used to seed planning
 * @param requiredMaterializedValueRefs output values that must be materialized for graph semantics
 * @param boundaryEdges edges that cross the partition boundary
 * @param boundaryReasons reasons expansion stopped at the boundary
 * @param estimatedWork backend work estimate
 * @param structuralMetrics structural metrics used for candidate scoring
 * @param plannerStrategy planner strategy that produced this partition
 * @param debugTrace compile trace decision associated with this partition
 */
public record Partition(
        String partitionId,
        ExecutionRegionKind regionKind,
        PartitionTarget target,
        List<Integer> orderedNodeIds,
        List<PartitionValue> values,
        List<PartitionEdge> internalEdges,
        List<Integer> externalInputNodeIds,
        List<PartitionValueRef> outputValueRefs,
        int anchorSeedNodeId,
        List<PartitionValueRef> requiredMaterializedValueRefs,
        List<PartitionEdge> boundaryEdges,
        List<PartitionBoundaryReason> boundaryReasons,
        long estimatedWork,
        AcceleratorPartitionScoreModel.CandidateMetrics structuralMetrics,
        PartitionPlannerStrategy plannerStrategy,
        PartitionDecisionTrace debugTrace
) {
    public Partition {
        partitionId = partitionId == null ? "" : partitionId;
        target = target == null ? PartitionTarget.NONE : target;
        regionKind = regionKind == null ? defaultRegionKind(target) : regionKind;
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        values = List.copyOf(values == null ? List.of() : values);
        internalEdges = List.copyOf(internalEdges == null ? List.of() : internalEdges);
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        outputValueRefs = List.copyOf(outputValueRefs == null ? List.of() : outputValueRefs);
        requiredMaterializedValueRefs = List.copyOf(requiredMaterializedValueRefs == null ? List.of() : requiredMaterializedValueRefs);
        boundaryEdges = List.copyOf(boundaryEdges == null ? List.of() : boundaryEdges);
        boundaryReasons = List.copyOf(boundaryReasons == null ? List.of() : boundaryReasons);
        estimatedWork = Math.max(0L, estimatedWork);
        structuralMetrics = structuralMetrics == null
                ? new AcceleratorPartitionScoreModel.CandidateMetrics(0, 0, 0, 0, 0)
                : structuralMetrics;
        plannerStrategy = plannerStrategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : plannerStrategy;
        if (anchorSeedNodeId < 0) {
            throw new IllegalArgumentException("anchorSeedNodeId must be >= 0");
        }
    }

    /**
     * Creates an accepted partition using the default semantic region kind for its target.
     */
    public Partition(
            String partitionId,
            PartitionTarget target,
            List<Integer> orderedNodeIds,
            List<PartitionValue> values,
            List<PartitionEdge> internalEdges,
            List<Integer> externalInputNodeIds,
            List<PartitionValueRef> outputValueRefs,
            int anchorSeedNodeId,
            List<PartitionValueRef> requiredMaterializedValueRefs,
            List<PartitionEdge> boundaryEdges,
            List<PartitionBoundaryReason> boundaryReasons,
            long estimatedWork,
            AcceleratorPartitionScoreModel.CandidateMetrics structuralMetrics,
            PartitionPlannerStrategy plannerStrategy,
            PartitionDecisionTrace debugTrace
    ) {
        this(
                partitionId,
                defaultRegionKind(target),
                target,
                orderedNodeIds,
                values,
                internalEdges,
                externalInputNodeIds,
                outputValueRefs,
                anchorSeedNodeId,
                requiredMaterializedValueRefs,
                boundaryEdges,
                boundaryReasons,
                estimatedWork,
                structuralMetrics,
                plannerStrategy,
                debugTrace
        );
    }

    private static ExecutionRegionKind defaultRegionKind(PartitionTarget target) {
        return target == PartitionTarget.CPU
                ? ExecutionRegionKind.CPU_EXECUTION
                : ExecutionRegionKind.ACCELERATOR_OWNERSHIP;
    }
}
