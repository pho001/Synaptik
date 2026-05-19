package graph.optimizer.region;

import graph.optimizer.GraphValueRef;

import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.ExecutionRegionKind;

import java.util.List;

/**
 * Optimized execution region derived from a partition.
 *
 * @param regionId stable region id
 * @param regionKind semantic region kind
 * @param sourcePartition partition that produced this region
 * @param target backend target
 * @param executionUnits execution units in region order
 * @param regionValues values tracked across unit and region boundaries
 * @param materializedOutputs outputs that must be materialized outside the region
 * @param trace region optimization diagnostics
 */
public record OptimizedRegion(
        String regionId,
        ExecutionRegionKind regionKind,
        Partition sourcePartition,
        PartitionTarget target,
        List<ExecutionUnit> executionUnits,
        List<RegionValue> regionValues,
        List<GraphValueRef> materializedOutputs,
        RegionOptimizationTrace trace
) {
    public OptimizedRegion {
        if (regionId == null || regionId.isBlank()) {
            throw new IllegalArgumentException("regionId cannot be blank");
        }
        if (sourcePartition == null || target == null) {
            throw new IllegalArgumentException("sourcePartition and target cannot be null");
        }
        regionKind = regionKind == null ? sourcePartition.regionKind() : regionKind;
        executionUnits = List.copyOf(executionUnits == null ? List.of() : executionUnits);
        regionValues = List.copyOf(regionValues == null ? List.of() : regionValues);
        materializedOutputs = List.copyOf(materializedOutputs == null ? List.of() : materializedOutputs);
        trace = trace == null ? RegionOptimizationTrace.empty() : trace;
    }

    /**
     * Creates an optimized region using the source partition's semantic region kind.
     */
    public OptimizedRegion(
            String regionId,
            Partition sourcePartition,
            PartitionTarget target,
            List<ExecutionUnit> executionUnits,
            List<RegionValue> regionValues,
            List<GraphValueRef> materializedOutputs,
            RegionOptimizationTrace trace
    ) {
        this(
                regionId,
                sourcePartition == null ? null : sourcePartition.regionKind(),
                sourcePartition,
                target,
                executionUnits,
                regionValues,
                materializedOutputs,
                trace
        );
    }
}
