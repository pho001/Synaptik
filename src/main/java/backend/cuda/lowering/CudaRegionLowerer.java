package backend.cuda.lowering;

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
        if (!(selectedPlan instanceof CudaGpuPartitionPlan cudaPlan) || cudaPlan.backend() != ComputeBackend.GPU_CUDA) {
            return null;
        }
        LoweringFamily loweringFamily = resolveLoweringFamily(request.region().executionUnits());
        GpuCompoundRegionSummary summary = cudaPlan.compoundSummary();
        return new LoweringResult(
                new LoweredRegion(
                        request.region().regionId(),
                        request.region().target(),
                        List.of(new LoweredExecutionUnit(
                                request.region().regionId() + "-cuda-graph",
                                loweringFamily,
                                request.region().sourcePartition().orderedNodeIds(),
                                cudaPlan.externalInputNodeIds(),
                                regionArtifact(summary, request.region().executionUnits())
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
