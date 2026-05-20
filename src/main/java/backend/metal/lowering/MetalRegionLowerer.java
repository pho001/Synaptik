package backend.metal.lowering;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuCompoundLoweringArtifact;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.accelerator.lowering.GpuCompoundRegionSummary;
import backend.accelerator.lowering.GpuRegionLoweredUnitSummary;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;
import backend.lowering.LoweringRequest;
import backend.lowering.LoweringResult;
import backend.lowering.RegionLowerer;
import backend.lowering.region.MetalRegionPayload;
import backend.lowering.region.RegionCost;
import backend.lowering.region.RegionDecision;
import backend.lowering.region.RegionExecutionGroup;
import backend.lowering.region.RegionExecutionKind;
import backend.lowering.region.RegionExecutionPlan;
import backend.lowering.region.RegionLegalityStatus;
import backend.lowering.region.RegionNodePlan;
import backend.lowering.region.RegionRole;
import backend.lowering.region.RegionStorageContract;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.region.ExecutionUnit;
import operations.Operation;

import java.util.List;

/**
 * Region lowerer that marks selected Metal partitions for graph execution.
 */
public final class MetalRegionLowerer implements RegionLowerer {
    /**
     * Lowers a Metal-targeted partition region when a Metal partition plan is attached.
     */
    @Override
    public LoweringResult lower(LoweringRequest request) {
        if (request == null || request.region().target() != PartitionTarget.GPU_METAL) {
            return null;
        }
        if (!request.capabilities().supports(ComputeBackend.GPU_METAL)) {
            return null;
        }
        PartitionPlan attachedPlan = request.context().partitionPlanFor(request.region().sourcePartition().partitionId());
        if (!(attachedPlan instanceof MetalPartitionPlan metalPlan) || metalPlan.backend() != ComputeBackend.GPU_METAL) {
            return null;
        }
        GpuCompoundRegionSummary summary = metalPlan.lowering().compoundSummary();
        GpuCompoundLoweringArtifact compoundArtifact = regionArtifact(summary, request.region().executionUnits());
        RegionExecutionPlan regionPlan = regionPlan(request, metalPlan, compoundArtifact);
        LoweredExecutionUnit unit = new LoweredExecutionUnit(
                request.region().regionId() + "-metal-graph",
                LoweringFamily.METAL_GRAPH_REGION,
                request.region().sourcePartition().orderedNodeIds(),
                metalPlan.externalInputNodeIds(),
                regionPlan
        );
        return new LoweringResult(
                new LoweredRegion(
                        request.region().regionId(),
                        request.region().target(),
                        List.of(unit)
                ),
                List.of()
        );
    }

    private static RegionExecutionPlan regionPlan(
            LoweringRequest request,
            MetalPartitionPlan plan,
            GpuCompoundLoweringArtifact compoundArtifact
    ) {
        List<Integer> orderedNodeIds = request.region().sourcePartition().orderedNodeIds();
        List<Integer> outputs = plan.producedOutputNodeIds();
        List<RegionNodePlan> nodePlans = orderedNodeIds.stream()
                .map(nodeId -> nodePlan(request, nodeId, outputs))
                .toList();
        RegionExecutionGroup group = new RegionExecutionGroup(
                request.region().regionId() + "-metal-graph-group-0",
                orderedNodeIds,
                RegionExecutionKind.GRAPH_EXECUTABLE,
                "METAL_GRAPH",
                plan.externalInputNodeIds(),
                outputs,
                List.of(),
                RegionStorageContract.DEVICE_BUFFER,
                "metal-graph-region"
        );
        return new RegionExecutionPlan(
                request.region().regionId(),
                PartitionTarget.GPU_METAL,
                LoweringFamily.METAL_GRAPH_REGION,
                plan.anchorNodeId(),
                orderedNodeIds,
                plan.externalInputNodeIds(),
                outputs,
                nodePlans,
                List.of(group),
                RegionCost.ofWork(plan.estimatedWork()),
                RegionDecision.selected(LoweringFamily.METAL_GRAPH_REGION.id(), "metal-graph-region"),
                new MetalRegionPayload(compoundArtifact, plan.manifest())
        );
    }

    private static RegionNodePlan nodePlan(LoweringRequest request, int nodeId, List<Integer> outputs) {
        var node = request.context().compiledNode(nodeId);
        Operation op = node == null ? null : node.operation();
        return new RegionNodePlan(
                nodeId,
                op == null ? Operation.OpType.UNKNOWN : op.opType(),
                node == null ? tensor.DataType.FLOAT64 : node.dataType(),
                outputs.contains(nodeId) ? RegionRole.BOUNDARY_OUTPUT : RegionRole.LOCAL_KERNEL,
                RegionExecutionKind.GRAPH_EXECUTABLE,
                "METAL_GRAPH",
                RegionStorageContract.DEVICE_BUFFER,
                node == null ? List.of() : node.inputIds(),
                List.of(nodeId),
                RegionLegalityStatus.SELECTED,
                "metal-graph-region"
        );
    }

    private static GpuCompoundLoweringArtifact regionArtifact(
            GpuCompoundRegionSummary summary,
            List<ExecutionUnit> executionUnits
    ) {
        List<GpuRegionLoweredUnitSummary> units = executionUnits == null
                ? List.of()
                : executionUnits.stream().map(GpuRegionLoweredUnitSummary::fromExecutionUnit).toList();
        if ((summary == null || summary.patternType() == GpuCompoundPatternType.NONE) && units.isEmpty()) {
            return null;
        }
        GpuCompoundRegionSummary resolvedSummary = summary == null
                ? GpuCompoundRegionSummary.none(ComputeBackend.GPU_METAL, units.stream()
                        .flatMap(unit -> unit.orderedNodeIds().stream())
                        .distinct()
                        .toList())
                : summary;
        return new GpuCompoundLoweringArtifact(resolvedSummary, units);
    }
}
