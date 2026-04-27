package backend.apple.lowering;

import backend.ComputeBackend;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;
import backend.lowering.LoweringRequest;
import backend.lowering.LoweringResult;
import backend.lowering.RegionLowerer;
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.region.ExecutionUnit;
import graph.optimizer.region.ExecutionUnitKind;

import java.util.List;

public final class AppleRegionLowerer implements RegionLowerer {
    @Override
    public LoweringResult lower(LoweringRequest request) {
        if (request == null || request.region().target() != PartitionTarget.GPU_METAL) {
            return null;
        }
        if (!request.capabilities().supports(ComputeBackend.GPU_METAL)) {
            return null;
        }
        PartitionPlan attachedPlan = request.context().partitionPlanFor(request.region().sourcePartition().partitionId());
        if (attachedPlan == null || attachedPlan.backend() != ComputeBackend.GPU_METAL) {
            return null;
        }
        LoweringFamily loweringFamily = resolveLoweringFamily(request.region().executionUnits());
        LoweredExecutionUnit unit = new LoweredExecutionUnit(
                request.region().regionId() + "-apple-graph",
                loweringFamily,
                request.region().sourcePartition().orderedNodeIds()
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

    private LoweringFamily resolveLoweringFamily(List<ExecutionUnit> units) {
        if (units == null || units.isEmpty()) {
            return LoweringFamily.APPLE_GRAPH_REGION;
        }
        if (units.size() == 1 && units.getFirst().kind() == ExecutionUnitKind.FUSED_ELEMENTWISE) {
            return LoweringFamily.APPLE_FUSED_ELEMENTWISE_GRAPH;
        }
        return LoweringFamily.APPLE_GRAPH_REGION;
    }
}
