package backend.metal.lowering;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuCompoundLoweringArtifact;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.accelerator.lowering.GpuCompoundRegionSummary;
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
        LoweringFamily loweringFamily = resolveLoweringFamily(request.region().executionUnits());
        GpuCompoundRegionSummary summary = metalPlan.lowering().compoundSummary();
        LoweredExecutionUnit unit = new LoweredExecutionUnit(
                request.region().regionId() + "-metal-graph",
                loweringFamily,
                request.region().sourcePartition().orderedNodeIds(),
                metalPlan.externalInputNodeIds(),
                compoundArtifact(summary)
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
            return LoweringFamily.METAL_GRAPH_REGION;
        }
        if (units.size() == 1 && units.getFirst().kind() == ExecutionUnitKind.FUSED_ELEMENTWISE) {
            return LoweringFamily.METAL_FUSED_ELEMENTWISE_GRAPH;
        }
        return LoweringFamily.METAL_GRAPH_REGION;
    }

    private static GpuCompoundLoweringArtifact compoundArtifact(GpuCompoundRegionSummary summary) {
        if (summary == null || summary.patternType() == GpuCompoundPatternType.NONE) {
            return null;
        }
        return new GpuCompoundLoweringArtifact(summary);
    }
}
