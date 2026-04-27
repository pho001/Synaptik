package graph.optimizer.region;

import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionTarget;

import java.util.List;

public record OptimizedRegion(
        String regionId,
        Partition sourcePartition,
        PartitionTarget target,
        List<ExecutionUnit> executionUnits,
        List<RegionValue> regionValues,
        List<RegionValueRef> materializedOutputs,
        RegionOptimizationTrace trace
) {
    public OptimizedRegion {
        if (regionId == null || regionId.isBlank()) {
            throw new IllegalArgumentException("regionId cannot be blank");
        }
        if (sourcePartition == null || target == null) {
            throw new IllegalArgumentException("sourcePartition and target cannot be null");
        }
        executionUnits = List.copyOf(executionUnits == null ? List.of() : executionUnits);
        regionValues = List.copyOf(regionValues == null ? List.of() : regionValues);
        materializedOutputs = List.copyOf(materializedOutputs == null ? List.of() : materializedOutputs);
        trace = trace == null ? RegionOptimizationTrace.empty() : trace;
    }
}
