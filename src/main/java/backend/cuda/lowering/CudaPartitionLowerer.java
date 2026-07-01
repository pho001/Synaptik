package backend.cuda.lowering;

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
import backend.lowering.partition.CudaPartitionPayload;
import backend.lowering.partition.PartitionCost;
import backend.lowering.partition.PartitionDecision;
import backend.lowering.partition.PartitionExecutionGroup;
import backend.lowering.partition.PartitionExecutionKind;
import backend.lowering.partition.BackendPartitionExecutionPlan;
import backend.lowering.partition.PartitionLegalityStatus;
import backend.lowering.partition.PartitionNodePlan;
import backend.lowering.partition.PartitionRole;
import backend.lowering.partition.PartitionStorageContract;
import planning.partition.PartitionTarget;
import planning.partition.execution.ExecutionUnit;
import operations.Operation;

import java.util.List;

/**
 * Partition lowerer that marks selected CUDA partitions for graph execution.
 */
public final class CudaPartitionLowerer implements PartitionLowerer {
    /**
     * Lowers a CUDA-targeted partition when a CUDA partition plan is attached.
     */
    @Override
    public LoweringResult lowerPartition(LoweringRequest request) {
        if (request == null || request.executablePartition().partition().target() != PartitionTarget.GPU_CUDA) {
            return null;
        }
        if (!request.capabilities().supports(ComputeBackend.GPU_CUDA)) {
            return null;
        }
        var selectedPlan = request.executablePartition().backendPlan();
        if (!(selectedPlan instanceof CudaGpuPartitionPlan cudaPlan) || cudaPlan.backend() != ComputeBackend.GPU_CUDA) {
            return null;
        }
        LoweringFamily loweringFamily = LoweringFamily.CUDA_GRAPH_PARTITION;
        GpuCompoundPartitionSummary summary = cudaPlan.compoundSummary();
        GpuCompoundLoweringArtifact compoundArtifact = partitionArtifact(summary, request.executablePartition().executionPlan().executionUnits());
        BackendPartitionExecutionPlan partitionPlan = partitionPlan(request, cudaPlan, loweringFamily, compoundArtifact);
        return new LoweringResult(
                new LoweredPartition(
                        request.executablePartition(),
                        List.of(new LoweredExecutionUnit(
                                request.executablePartition().partition().partitionId() + "-cuda-graph",
                                loweringFamily,
                                request.executablePartition().partition().orderedNodeIds(),
                                cudaPlan.externalInputNodeIds(),
                                partitionPlan
                        ))
                ),
                List.of()
        );
    }

    private static BackendPartitionExecutionPlan partitionPlan(
            LoweringRequest request,
            CudaGpuPartitionPlan plan,
            LoweringFamily loweringFamily,
            GpuCompoundLoweringArtifact compoundArtifact
    ) {
        List<Integer> orderedNodeIds = request.executablePartition().partition().orderedNodeIds();
        List<Integer> outputs = plan.producedOutputNodeIds();
        List<PartitionNodePlan> nodePlans = orderedNodeIds.stream()
                .map(nodeId -> nodePlan(request, nodeId, outputs, loweringFamily))
                .toList();
        PartitionExecutionGroup group = new PartitionExecutionGroup(
                request.executablePartition().partition().partitionId() + "-cuda-graph-group-0",
                orderedNodeIds,
                PartitionExecutionKind.GRAPH_EXECUTABLE,
                "CUDA_GRAPH",
                plan.externalInputNodeIds(),
                outputs,
                List.of(),
                PartitionStorageContract.DEVICE_BUFFER,
                "cuda-graph-partition"
        );
        return new BackendPartitionExecutionPlan(
                request.executablePartition().partition().partitionId(),
                loweringFamily,
                plan.anchorNodeId(),
                orderedNodeIds,
                plan.externalInputNodeIds(),
                outputs,
                nodePlans,
                List.of(group),
                PartitionCost.ofWork(plan.estimatedWork()),
                PartitionDecision.selected(loweringFamily.id(), "cuda-graph-partition"),
                new CudaPartitionPayload(compoundArtifact, plan.manifest())
        );
    }

    private static PartitionNodePlan nodePlan(
            LoweringRequest request,
            int nodeId,
            List<Integer> outputs,
            LoweringFamily loweringFamily
    ) {
        var node = request.context().compiledNode(nodeId);
        Operation op = node == null ? null : node.operation();
        return new PartitionNodePlan(
                nodeId,
                op == null ? Operation.OpType.UNKNOWN : op.opType(),
                node == null ? tensor.DataType.FLOAT64 : node.dataType(),
                outputs.contains(nodeId) ? PartitionRole.BOUNDARY_OUTPUT : PartitionRole.LOCAL_KERNEL,
                PartitionExecutionKind.GRAPH_EXECUTABLE,
                "CUDA_GRAPH",
                PartitionStorageContract.DEVICE_BUFFER,
                node == null ? List.of() : node.inputIds(),
                List.of(nodeId),
                PartitionLegalityStatus.SELECTED,
                "cuda-graph-partition"
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
                ? GpuCompoundPartitionSummary.none(ComputeBackend.GPU_CUDA, units.stream()
                        .flatMap(unit -> unit.orderedNodeIds().stream())
                        .distinct()
                        .toList())
                : summary;
        return new GpuCompoundLoweringArtifact(resolvedSummary, units);
    }
}
