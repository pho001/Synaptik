package backend.cpu.lowering;

import backend.contract.ComputeBackend;
import config.runtime.BlasProvider;
import backend.cpu.fused.plan.FusedOperationBuilder;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweredUnitArtifact;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;
import backend.lowering.LoweringRequest;
import backend.lowering.LoweringResult;
import backend.lowering.RegionLowerer;
import backend.lowering.region.CpuFusedRegionPayload;
import backend.lowering.region.CpuSpecializedPrimitivePayload;
import backend.lowering.region.EmptyRegionPayload;
import backend.lowering.region.RegionBackendPayload;
import backend.lowering.region.RegionCost;
import backend.lowering.region.RegionDecision;
import backend.lowering.region.RegionExecutionGroup;
import backend.lowering.region.RegionExecutionKind;
import backend.lowering.region.RegionExecutionPlan;
import backend.lowering.region.RegionLegalityStatus;
import backend.lowering.region.RegionNodePlan;
import backend.lowering.region.RegionRole;
import backend.lowering.region.RegionStorageContract;
import graph.model.CompiledNode;
import planning.region.ExecutionUnit;
import planning.region.ExecutionUnitKind;
import planning.value.GraphValueRef;
import operations.Operation;

import java.util.ArrayList;
import java.util.List;

public final class CpuRegionLowerer implements RegionLowerer {
    @Override
    public LoweringResult lower(LoweringRequest request) {
        if (request == null || request.region().target() != planning.partition.PartitionTarget.CPU) {
            return null;
        }
        if (!request.capabilities().supports(ComputeBackend.CPU)) {
            return null;
        }
        List<LoweredExecutionUnit> loweredUnits = new ArrayList<>(request.region().executionUnits().size());
        for (ExecutionUnit unit : request.region().executionUnits()) {
            loweredUnits.add(lowerUnit(unit, request));
        }
        return new LoweringResult(
                new LoweredRegion(request.region().regionId(), request.region().target(), loweredUnits),
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
        RegionExecutionPlan regionPlan = regionPlan(unit, request, family, legacyArtifact);
        return new LoweredExecutionUnit(
                unit.unitId(),
                family,
                unit.orderedNodeIds(),
                unit.inputValueRefs().stream()
                        .map(CpuRegionLowerer::nodeIdFromRef)
                        .map(nodeId -> resolveExecutionInputNodeId(nodeId, request))
                        .filter(id -> id >= 0)
                        .distinct()
                        .toList(),
                regionPlan
        );
    }

    private RegionExecutionPlan regionPlan(
            ExecutionUnit unit,
            LoweringRequest request,
            LoweringFamily family,
            LoweredUnitArtifact legacyArtifact
    ) {
        List<Integer> orderedNodeIds = unit.orderedNodeIds();
        int anchorNodeId = orderedNodeIds.getLast();
        List<Integer> externalInputNodeIds = unit.inputValueRefs().stream()
                .map(CpuRegionLowerer::nodeIdFromRef)
                .map(nodeId -> resolveExecutionInputNodeId(nodeId, request))
                .filter(id -> id >= 0)
                .distinct()
                .toList();
        List<Integer> boundaryOutputNodeIds = unit.outputValueRefs().stream()
                .map(CpuRegionLowerer::nodeIdFromRef)
                .filter(id -> id >= 0)
                .distinct()
                .toList();
        RegionBackendPayload payload = legacyArtifact instanceof backend.cpu.fused.plan.FusedOperationPreparation fused
                ? new CpuFusedRegionPayload(fused)
                : unit.kind() == ExecutionUnitKind.SPECIALIZED_PRIMITIVE && unit.specialization() != null
                ? new CpuSpecializedPrimitivePayload(unit.specialization())
                : EmptyRegionPayload.INSTANCE;
        RegionExecutionKind executionKind = switch (family) {
            case FUSED_NATIVE -> RegionExecutionKind.FUSED_KERNEL;
            case BLAS -> RegionExecutionKind.PROVIDER_CALL;
            default -> RegionExecutionKind.DIRECT_KERNEL;
        };
        RegionStorageContract storageContract = RegionStorageContract.CPU_ARRAY;
        String physicalKernel = unit.kind() == ExecutionUnitKind.SPECIALIZED_PRIMITIVE && unit.specialization() != null
                ? "CPU1_" + unit.specialization().kind().name()
                : family.id();
        List<RegionNodePlan> nodePlans = orderedNodeIds.stream()
                .map(nodeId -> nodePlan(nodeId, request, family, executionKind, physicalKernel, storageContract, boundaryOutputNodeIds))
                .toList();
        RegionExecutionGroup group = new RegionExecutionGroup(
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
        return new RegionExecutionPlan(
                request.region().regionId() + "/" + unit.unitId(),
                planning.partition.PartitionTarget.CPU,
                family,
                anchorNodeId,
                orderedNodeIds,
                externalInputNodeIds,
                boundaryOutputNodeIds.isEmpty() ? List.of(anchorNodeId) : boundaryOutputNodeIds,
                nodePlans,
                List.of(group),
                RegionCost.ofWork(unit.estimatedWork()),
                RegionDecision.selected(family.id(), "cpu-lowered-unit"),
                payload
        );
    }

    private RegionNodePlan nodePlan(
            int nodeId,
            LoweringRequest request,
            LoweringFamily family,
            RegionExecutionKind executionKind,
            String physicalKernel,
            RegionStorageContract storageContract,
            List<Integer> boundaryOutputNodeIds
    ) {
        CompiledNode node = request.context().compiledNode(nodeId);
        Operation op = node == null ? null : node.operation();
        RegionRole role = boundaryOutputNodeIds.contains(nodeId)
                ? RegionRole.BOUNDARY_OUTPUT
                : family == LoweringFamily.BLAS ? RegionRole.PROVIDER : RegionRole.LOCAL_KERNEL;
        return new RegionNodePlan(
                nodeId,
                op == null ? Operation.OpType.UNKNOWN : op.opType(),
                node == null ? tensor.DataType.FLOAT64 : node.dataType(),
                role,
                executionKind,
                physicalKernel,
                storageContract,
                node == null ? List.of() : node.inputIds(),
                List.of(nodeId),
                RegionLegalityStatus.SELECTED,
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
