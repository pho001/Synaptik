package backend.cpu.lowering;

import backend.contract.ComputeBackend;
import config.runtime.BlasProvider;
import backend.cpu.fused.plan.FusedOperationBuilder;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweredUnitArtifact;
import backend.lowering.LoweredPartition;
import backend.lowering.LoweringFamily;
import backend.lowering.LoweringRequest;
import backend.lowering.LoweringResult;
import backend.lowering.PartitionLowerer;
import backend.lowering.partition.CpuFusedPartitionPayload;
import backend.lowering.partition.CpuSpecializedPrimitivePayload;
import backend.lowering.partition.EmptyPartitionPayload;
import backend.lowering.partition.PartitionBackendPayload;
import backend.lowering.partition.PartitionCost;
import backend.lowering.partition.PartitionDecision;
import backend.lowering.partition.PartitionExecutionGroup;
import backend.lowering.partition.PartitionExecutionKind;
import backend.lowering.partition.BackendPartitionExecutionPlan;
import backend.lowering.partition.PartitionLegalityStatus;
import backend.lowering.partition.PartitionNodePlan;
import backend.lowering.partition.PartitionRole;
import backend.lowering.partition.PartitionStorageContract;
import graph.model.CompiledNode;
import planning.partition.execution.ExecutionUnit;
import planning.partition.execution.ExecutionUnitKind;
import planning.value.GraphValueRef;
import operations.Operation;

import java.util.ArrayList;
import java.util.List;

public final class CpuPartitionLowerer implements PartitionLowerer {
    @Override
    public LoweringResult lowerPartition(LoweringRequest request) {
        if (request == null || request.executablePartition().partition().target() != planning.partition.PartitionTarget.CPU) {
            return null;
        }
        if (!request.capabilities().supports(ComputeBackend.CPU)) {
            return null;
        }
        List<LoweredExecutionUnit> loweredUnits = new ArrayList<>(request.executablePartition().executionPlan().executionUnits().size());
        for (ExecutionUnit unit : request.executablePartition().executionPlan().executionUnits()) {
            loweredUnits.add(lowerUnit(unit, request));
        }
        return new LoweringResult(
                new LoweredPartition(request.executablePartition(), loweredUnits),
                List.of()
        );
    }

    private LoweringFamily chooseSingleOpFamily(ExecutionUnit unit, LoweringRequest request) {
        CompiledNode node = unit.orderedNodeIds().isEmpty() ? null : request.context().compiledNode(unit.orderedNodeIds().getFirst());
        if (node == null || node.operation() == null) {
            return LoweringFamily.DIRECT_KERNEL;
        }
        boolean blasEnabled = request.context().runtimeConfig() != null
                && request.context().runtimeConfig().blas().provider() != BlasProvider.NONE;
        long blasMinWork = request.context().runtimeConfig() == null
                ? Long.MAX_VALUE
                : request.context().runtimeConfig().blas().matmulMinWork();
        boolean matmulFamily = node.operation().opType() == operations.Operation.OpType.MATMUL
                || node.operation().opType() == operations.Operation.OpType.LINEAR;
        if (blasEnabled && matmulFamily && unit.estimatedWork() >= blasMinWork) {
            return LoweringFamily.BLAS;
        }
        return LoweringFamily.DIRECT_KERNEL;
    }

    private LoweredExecutionUnit lowerUnit(ExecutionUnit unit, LoweringRequest request) {
        LoweringFamily family;
        LoweredUnitArtifact legacyArtifact = null;
        if (unit.kind() == ExecutionUnitKind.FUSED_ELEMENTWISE) {
            family = LoweringFamily.FUSED_NATIVE;
            legacyArtifact = FusedOperationBuilder.build(
                    unit.orderedNodeIds(),
                    request.context()::compiledNode,
                    request.context().descriptorIndex()
            );
        } else if (unit.kind() == ExecutionUnitKind.SPECIALIZED_PRIMITIVE) {
            family = LoweringFamily.DIRECT_KERNEL;
        } else {
            family = chooseSingleOpFamily(unit, request);
        }
        BackendPartitionExecutionPlan partitionPlan = partitionPlan(unit, request, family, legacyArtifact);
        return new LoweredExecutionUnit(
                unit.unitId(),
                family,
                unit.orderedNodeIds(),
                unit.inputValueRefs().stream()
                        .map(CpuPartitionLowerer::nodeIdFromRef)
                        .map(nodeId -> resolveExecutionInputNodeId(nodeId, request))
                        .filter(id -> id >= 0)
                        .distinct()
                        .toList(),
                partitionPlan
        );
    }

    private BackendPartitionExecutionPlan partitionPlan(
            ExecutionUnit unit,
            LoweringRequest request,
            LoweringFamily family,
            LoweredUnitArtifact legacyArtifact
    ) {
        List<Integer> orderedNodeIds = unit.orderedNodeIds();
        int anchorNodeId = orderedNodeIds.getLast();
        List<Integer> externalInputNodeIds = unit.inputValueRefs().stream()
                .map(CpuPartitionLowerer::nodeIdFromRef)
                .map(nodeId -> resolveExecutionInputNodeId(nodeId, request))
                .filter(id -> id >= 0)
                .distinct()
                .toList();
        List<Integer> boundaryOutputNodeIds = unit.outputValueRefs().stream()
                .map(CpuPartitionLowerer::nodeIdFromRef)
                .filter(id -> id >= 0)
                .distinct()
                .toList();
        PartitionBackendPayload payload = legacyArtifact instanceof backend.cpu.fused.plan.FusedOperationPreparation fused
                ? new CpuFusedPartitionPayload(fused)
                : unit.kind() == ExecutionUnitKind.SPECIALIZED_PRIMITIVE && unit.specialization() != null
                ? new CpuSpecializedPrimitivePayload(unit.specialization())
                : EmptyPartitionPayload.INSTANCE;
        PartitionExecutionKind executionKind = switch (family) {
            case FUSED_NATIVE -> PartitionExecutionKind.FUSED_KERNEL;
            case BLAS -> PartitionExecutionKind.PROVIDER_CALL;
            default -> PartitionExecutionKind.DIRECT_KERNEL;
        };
        PartitionStorageContract storageContract = PartitionStorageContract.CPU_ARRAY;
        String physicalKernel = unit.kind() == ExecutionUnitKind.SPECIALIZED_PRIMITIVE && unit.specialization() != null
                ? "CPU1_" + unit.specialization().kind().name()
                : family.id();
        List<PartitionNodePlan> nodePlans = orderedNodeIds.stream()
                .map(nodeId -> nodePlan(nodeId, request, family, executionKind, physicalKernel, storageContract, boundaryOutputNodeIds))
                .toList();
        PartitionExecutionGroup group = new PartitionExecutionGroup(
                unit.unitId() + "-group-0",
                orderedNodeIds,
                executionKind,
                physicalKernel,
                externalInputNodeIds,
                boundaryOutputNodeIds.isEmpty() ? List.of(anchorNodeId) : boundaryOutputNodeIds,
                List.of(),
                storageContract,
                "cpu-lowered-unit"
        );
        return new BackendPartitionExecutionPlan(
                request.executablePartition().partition().partitionId() + "/" + unit.unitId(),
                family,
                anchorNodeId,
                orderedNodeIds,
                externalInputNodeIds,
                boundaryOutputNodeIds.isEmpty() ? List.of(anchorNodeId) : boundaryOutputNodeIds,
                nodePlans,
                List.of(group),
                PartitionCost.ofWork(unit.estimatedWork()),
                PartitionDecision.selected(family.id(), "cpu-lowered-unit"),
                payload
        );
    }

    private PartitionNodePlan nodePlan(
            int nodeId,
            LoweringRequest request,
            LoweringFamily family,
            PartitionExecutionKind executionKind,
            String physicalKernel,
            PartitionStorageContract storageContract,
            List<Integer> boundaryOutputNodeIds
    ) {
        CompiledNode node = request.context().compiledNode(nodeId);
        Operation op = node == null ? null : node.operation();
        PartitionRole role = boundaryOutputNodeIds.contains(nodeId)
                ? PartitionRole.BOUNDARY_OUTPUT
                : family == LoweringFamily.BLAS ? PartitionRole.PROVIDER : PartitionRole.LOCAL_KERNEL;
        return new PartitionNodePlan(
                nodeId,
                op == null ? Operation.OpType.UNKNOWN : op.opType(),
                node == null ? tensor.DataType.FLOAT64 : node.dataType(),
                role,
                executionKind,
                physicalKernel,
                storageContract,
                node == null ? List.of() : node.inputIds(),
                List.of(nodeId),
                PartitionLegalityStatus.SELECTED,
                "cpu-lowered-unit"
        );
    }

    private static int nodeIdFromRef(GraphValueRef ref) {
        return ref == null ? -1 : ref.nodeId();
    }

    private int resolveExecutionInputNodeId(int nodeId, LoweringRequest request) {
        int current = nodeId;
        while (current >= 0) {
            CompiledNode node = request.context().compiledNode(current);
            if (node == null || node.operation() == null || node.inputIds().isEmpty()) {
                return current;
            }
            if (node.storageOwnerId() == node.id()) {
                return current;
            }
            current = node.storageOwnerId();
        }
        return nodeId;
    }
}
