package backend.cpu.lowering;

import backend.ComputeBackend;
import backend.blas.BlasProvider;
import backend.cpu.fused.plan.LoweredFusedOperationBuilder;
import backend.lowering.BackendWorkspaceRequirement;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweredUnitArtifact;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;
import backend.lowering.LoweringRequest;
import backend.lowering.LoweringResult;
import backend.lowering.RegionLowerer;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.optimizer.region.ExecutionUnit;
import graph.optimizer.region.ExecutionUnitKind;
import graph.optimizer.region.RegionValueRef;

import java.util.ArrayList;
import java.util.List;

public final class CpuRegionLowerer implements RegionLowerer {
    @Override
    public LoweringResult lower(LoweringRequest request) {
        if (request == null || request.region().target() != graph.optimizer.partition.PartitionTarget.CPU) {
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
        LoweredUnitArtifact artifact = null;
        if (unit.kind() == ExecutionUnitKind.FUSED_ELEMENTWISE) {
            family = LoweringFamily.FUSED_NATIVE;
            artifact = LoweredFusedOperationBuilder.build(unit.orderedNodeIds(), request.context()::compiledNode);
        } else {
            family = chooseSingleOpFamily(unit, request);
        }
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
                artifact
        );
    }

    private static int nodeIdFromRef(RegionValueRef ref) {
        if (ref == null || ref.valueId() == null || !ref.valueId().startsWith("node-")) {
            return -1;
        }
        try {
            return Integer.parseInt(ref.valueId().substring("node-".length()));
        } catch (NumberFormatException ignored) {
            return -1;
        }
    }

    private int resolveExecutionInputNodeId(int nodeId, LoweringRequest request) {
        int current = nodeId;
        while (current >= 0) {
            CompiledNode node = request.context().compiledNode(current);
            if (node == null || node.operation() == null || node.inputIds().isEmpty()) {
                return current;
            }
            boolean aliasView = switch (node.operation().opType()) {
                case NOOP, EXPAND, SELECT, PERMUTE, EXPAND_DIMS, SQUEEZE -> true;
                case RESHAPE -> {
                    CompiledTensorDescriptor input = request.context().descriptor(node.inputIds().getFirst());
                    yield input != null && input.contiguous();
                }
                default -> false;
            };
            if (!aliasView) {
                return current;
            }
            current = node.inputIds().getFirst();
        }
        return nodeId;
    }

}
