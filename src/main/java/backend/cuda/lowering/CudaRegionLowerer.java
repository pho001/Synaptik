package backend.cuda.lowering;

import backend.contract.ComputeBackend;
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
import backend.lowering.region.CudaRegionPayload;
import backend.lowering.region.RegionCost;
import backend.lowering.region.RegionDecision;
import backend.lowering.region.RegionExecutionGroup;
import backend.lowering.region.RegionExecutionKind;
import backend.lowering.region.RegionExecutionPlan;
import backend.lowering.region.RegionLegalityStatus;
import backend.lowering.region.RegionNodePlan;
import backend.lowering.region.RegionRole;
import backend.lowering.region.RegionStorageContract;
import planning.partition.PartitionTarget;
import planning.region.ExecutionUnit;
import operations.Operation;

import java.util.List;

/**
 * Region lowerer that marks selected CUDA partitions for graph execution.
 */
public final class CudaRegionLowerer implements RegionLowerer {
    /**
     * Lowers a CUDA-targeted partition region when a CUDA partition plan is attached.
     */
    @Override
    public LoweringResult lower(LoweringRequest request) {
        if (request == null || request.region().target() != PartitionTarget.GPU_CUDA) {
            return null;
        }
        if (!request.capabilities().supports(ComputeBackend.GPU_CUDA)) {
            return null;
        }
        var selectedPlan = request.context().partitionPlanFor(request.region().sourcePartition().partitionId());
        if (!(selectedPlan instanceof CudaGpuPartitionPlan cudaPlan) || cudaPlan.backend() != ComputeBackend.GPU_CUDA) {
            return null;
        }
        LoweringFamily loweringFamily = LoweringFamily.CUDA_GRAPH_REGION;
        GpuCompoundRegionSummary summary = cudaPlan.compoundSummary();
        GpuCompoundLoweringArtifact compoundArtifact = regionArtifact(summary, request.region().executionUnits());
        RegionExecutionPlan regionPlan = regionPlan(request, cudaPlan, loweringFamily, compoundArtifact);
        return new LoweringResult(
                new LoweredRegion(
                        request.region().regionId(),
                        request.region().target(),
                        List.of(new LoweredExecutionUnit(
                                request.region().regionId() + "-cuda-graph",
                                loweringFamily,
                                request.region().sourcePartition().orderedNodeIds(),
                                cudaPlan.externalInputNodeIds(),
                                regionPlan
                        ))
                ),
                List.of()
        );
    }

    private static RegionExecutionPlan regionPlan(
            LoweringRequest request,
            CudaGpuPartitionPlan plan,
            LoweringFamily loweringFamily,
            GpuCompoundLoweringArtifact compoundArtifact
    ) {
        List<Integer> orderedNodeIds = request.region().sourcePartition().orderedNodeIds();
        List<Integer> outputs = plan.producedOutputNodeIds();
        List<RegionNodePlan> nodePlans = orderedNodeIds.stream()
                .map(nodeId -> nodePlan(request, nodeId, outputs, loweringFamily))
                .toList();
        RegionExecutionGroup group = new RegionExecutionGroup(
                request.region().regionId() + "-cuda-graph-group-0",
                orderedNodeIds,
                RegionExecutionKind.GRAPH_EXECUTABLE,
                "CUDA_GRAPH",
                plan.externalInputNodeIds(),
                outputs,
                List.of(),
                RegionStorageContract.DEVICE_BUFFER,
                "cuda-graph-region"
        );
        return new RegionExecutionPlan(
                request.region().regionId(),
                PartitionTarget.GPU_CUDA,
                loweringFamily,
                plan.anchorNodeId(),
                orderedNodeIds,
                plan.externalInputNodeIds(),
                outputs,
                nodePlans,
                List.of(group),
                RegionCost.ofWork(plan.estimatedWork()),
                RegionDecision.selected(loweringFamily.id(), "cuda-graph-region"),
                new CudaRegionPayload(compoundArtifact, plan.manifest())
        );
    }

    private static RegionNodePlan nodePlan(
            LoweringRequest request,
            int nodeId,
            List<Integer> outputs,
            LoweringFamily loweringFamily
    ) {
        var node = request.context().compiledNode(nodeId);
        Operation op = node == null ? null : node.operation();
        return new RegionNodePlan(
                nodeId,
                op == null ? Operation.OpType.UNKNOWN : op.opType(),
                node == null ? tensor.DataType.FLOAT64 : node.dataType(),
                outputs.contains(nodeId) ? RegionRole.BOUNDARY_OUTPUT : RegionRole.LOCAL_KERNEL,
                RegionExecutionKind.GRAPH_EXECUTABLE,
                "CUDA_GRAPH",
                RegionStorageContract.DEVICE_BUFFER,
                node == null ? List.of() : node.inputIds(),
                List.of(nodeId),
                RegionLegalityStatus.SELECTED,
                "cuda-graph-region"
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
                ? GpuCompoundRegionSummary.none(ComputeBackend.GPU_CUDA, units.stream()
                        .flatMap(unit -> unit.orderedNodeIds().stream())
                        .distinct()
                        .toList())
                : summary;
        return new GpuCompoundLoweringArtifact(resolvedSummary, units);
    }
}
