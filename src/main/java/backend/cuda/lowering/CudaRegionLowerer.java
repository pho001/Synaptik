package backend.cuda.lowering;

import backend.ComputeBackend;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweredRegion;
import backend.lowering.LoweringFamily;
import backend.lowering.LoweringRequest;
import backend.lowering.LoweringResult;
import backend.lowering.RegionLowerer;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.region.ExecutionUnit;
import graph.optimizer.region.ExecutionUnitKind;

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
        if (selectedPlan == null || selectedPlan.backend() != ComputeBackend.GPU_CUDA) {
            return null;
        }
        LoweringFamily loweringFamily = resolveLoweringFamily(request.region().executionUnits());
        return new LoweringResult(
                new LoweredRegion(
                        request.region().regionId(),
                        request.region().target(),
                        List.of(new LoweredExecutionUnit(
                                request.region().regionId() + "-cuda-graph",
                                loweringFamily,
                                request.region().sourcePartition().orderedNodeIds()
                        ))
                ),
                List.of()
        );
    }

    private LoweringFamily resolveLoweringFamily(List<ExecutionUnit> units) {
        if (units == null || units.isEmpty()) {
            return LoweringFamily.CUDA_GRAPH_REGION;
        }
        if (units.size() == 1 && units.getFirst().kind() == ExecutionUnitKind.FUSED_ELEMENTWISE) {
            return LoweringFamily.CUDA_FUSED_ELEMENTWISE_GRAPH;
        }
        return LoweringFamily.CUDA_GRAPH_REGION;
    }
}
