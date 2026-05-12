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
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.region.ExecutionUnit;

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
        LoweredExecutionUnit unit = new LoweredExecutionUnit(
                request.region().regionId() + "-metal-graph",
                LoweringFamily.METAL_GRAPH_REGION,
                request.region().sourcePartition().orderedNodeIds(),
                metalPlan.externalInputNodeIds(),
                regionArtifact(summary, request.region().executionUnits())
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
