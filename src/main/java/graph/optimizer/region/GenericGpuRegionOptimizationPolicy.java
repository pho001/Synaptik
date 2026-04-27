package graph.optimizer.region;

import graph.optimizer.partition.Partition;

import java.util.List;

public final class GenericGpuRegionOptimizationPolicy implements RegionOptimizationPolicy {
    @Override
    public List<ExecutionUnit> buildUnits(Partition partition, RegionOptimizationContext context) {
        if (RegionOptimizationUnitSupport.shouldFuseWholePartition(partition, context)) {
            return List.of(RegionOptimizationUnitSupport.buildFusedUnit(partition));
        }
        return RegionOptimizationUnitSupport.buildSingleOpUnits(partition, context);
    }
}
