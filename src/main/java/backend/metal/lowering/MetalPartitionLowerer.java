package backend.metal.lowering;

import backend.contract.ComputeBackend;
import backend.accelerator.lowering.GpuCompoundLoweringArtifact;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.accelerator.lowering.GpuCompoundPartitionSummary;
import backend.accelerator.lowering.GpuPartitionLoweredUnitSummary;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweredPartition;
import backend.lowering.LoweringFamily;
import backend.lowering.LoweringRequest;
import backend.lowering.LoweringResult;
import backend.lowering.PartitionLowerer;
import backend.lowering.partition.MetalPartitionPayload;
import backend.lowering.partition.PartitionCost;
import backend.lowering.partition.PartitionDecision;
import backend.lowering.partition.PartitionExecutionGroup;
import backend.lowering.partition.PartitionExecutionKind;
import backend.lowering.partition.BackendPartitionExecutionPlan;
import backend.lowering.partition.PartitionLegalityStatus;
import backend.lowering.partition.PartitionNodePlan;
import backend.lowering.partition.PartitionRole;
import backend.lowering.partition.PartitionStorageContract;
import planning.partition.PartitionPlan;
import planning.partition.PartitionTarget;
import planning.partition.execution.ExecutionUnit;
import operations.Operation;

import java.util.List;

/**
 * Partition lowerer that marks selected Metal partitions for graph execution.
 */
public final class MetalPartitionLowerer implements PartitionLowerer {
    /**
     * Lowers a Metal-targeted partition when a Metal partition plan is attached.
     */
    @Override
    public LoweringResult lowerPartition(LoweringRequest request) {
        if (request == null || request.executablePartition().partition().target() != PartitionTarget.GPU_METAL) {
            return null;
        }
        if (!request.capabilities().supports(ComputeBackend.GPU_METAL)) {
            return null;
        }
        PartitionPlan attachedPlan = request.executablePartition().backendPlan();
        if (!(attachedPlan instanceof MetalPartitionPlan metalPlan) || metalPlan.backend() != ComputeBackend.GPU_METAL) {
            return null;
        }
        GpuCompoundPartitionSummary summary = metalPlan.lowering().compoundSummary();
        GpuCompoundLoweringArtifact compoundArtifact = partitionArtifact(summary, request.executablePartition().executionPlan().executionUnits());
        BackendPartitionExecutionPlan partitionPlan = partitionPlan(request, metalPlan, compoundArtifact);
        LoweredExecutionUnit unit = new LoweredExecutionUnit(
                request.executablePartition().partition().partitionId() + "-metal-graph",
                LoweringFamily.METAL_GRAPH_PARTITION,
                request.executablePartition().partition().orderedNodeIds(),
                metalPlan.externalInputNodeIds(),
                partitionPlan
        );
        return new LoweringResult(
                new LoweredPartition(
                        request.executablePartition(),
                        List.of(unit)
                ),
                List.of()
        );
    }

    private static BackendPartitionExecutionPlan partitionPlan(
            LoweringRequest request,
            MetalPartitionPlan plan,
            GpuCompoundLoweringArtifact compoundArtifact
    ) {
        List<Integer> orderedNodeIds = request.executablePartition().partition().orderedNodeIds();
        List<Integer> outputs = plan.producedOutputNodeIds();
        List<PartitionNodePlan> nodePlans = orderedNodeIds.stream()
                .map(nodeId -> nodePlan(request, nodeId, outputs))
                .toList();
        PartitionExecutionGroup group = new PartitionExecutionGroup(
                request.executablePartition().partition().partitionId() + "-metal-graph-group-0",
                orderedNodeIds,
                PartitionExecutionKind.GRAPH_EXECUTABLE,
                "METAL_GRAPH",
                plan.externalInputNodeIds(),
                outputs,
                List.of(),
                PartitionStorageContract.DEVICE_BUFFER,
                "metal-graph-partition"
        );
        return new BackendPartitionExecutionPlan(
                request.executablePartition().partition().partitionId(),
                LoweringFamily.METAL_GRAPH_PARTITION,
                plan.anchorNodeId(),
                orderedNodeIds,
                plan.externalInputNodeIds(),
                outputs,
                nodePlans,
                List.of(group),
                PartitionCost.ofWork(plan.estimatedWork()),
                PartitionDecision.selected(LoweringFamily.METAL_GRAPH_PARTITION.id(), "metal-graph-partition"),
                new MetalPartitionPayload(compoundArtifact, plan.manifest())
        );
    }

    private static PartitionNodePlan nodePlan(LoweringRequest request, int nodeId, List<Integer> outputs) {
        var node = request.context().compiledNode(nodeId);
        Operation op = node == null ? null : node.operation();
        return new PartitionNodePlan(
                nodeId,
                op == null ? Operation.OpType.UNKNOWN : op.opType(),
                node == null ? tensor.DataType.FLOAT64 : node.dataType(),
                outputs.contains(nodeId) ? PartitionRole.BOUNDARY_OUTPUT : PartitionRole.LOCAL_KERNEL,
                PartitionExecutionKind.GRAPH_EXECUTABLE,
                "METAL_GRAPH",
                PartitionStorageContract.DEVICE_BUFFER,
                node == null ? List.of() : node.inputIds(),
                List.of(nodeId),
                PartitionLegalityStatus.SELECTED,
                "metal-graph-partition"
        );
    }

    private static GpuCompoundLoweringArtifact partitionArtifact(
            GpuCompoundPartitionSummary summary,
            List<ExecutionUnit> executionUnits
    ) {
        List<GpuPartitionLoweredUnitSummary> units = executionUnits == null
                ? List.of()
                : executionUnits.stream().map(GpuPartitionLoweredUnitSummary::fromExecutionUnit).toList();
        if ((summary == null || summary.patternType() == GpuCompoundPatternType.NONE) && units.isEmpty()) {
            return null;
        }
        GpuCompoundPartitionSummary resolvedSummary = summary == null
                ? GpuCompoundPartitionSummary.none(ComputeBackend.GPU_METAL, units.stream()
                        .flatMap(unit -> unit.orderedNodeIds().stream())
                        .distinct()
                        .toList())
                : summary;
        return new GpuCompoundLoweringArtifact(resolvedSummary, units);
    }
}
