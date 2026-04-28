package graph.optimizer.region;

import graph.optimizer.partition.Partition;

import java.util.List;

/**
 * Generic accelerator region policy.
 *
 * <p>The current policy fuses an entire partition only when every node is fusable and there is a single output;
 * otherwise it emits single-operation units so backend-specific lowerers can remain conservative.
 */
public final class GenericGpuRegionOptimizationPolicy implements RegionOptimizationPolicy {
    /**
     * Builds generic accelerator execution units for a partition.
     *
     * @param partition accepted partition
     * @param context region optimization context
     * @return fused whole-partition unit or single-operation units
     */
    @Override
    public List<ExecutionUnit> buildUnits(Partition partition, RegionOptimizationContext context) {
        if (RegionOptimizationUnitSupport.shouldFuseWholePartition(partition, context)) {
            return List.of(RegionOptimizationUnitSupport.buildFusedUnit(partition));
        }
        return RegionOptimizationUnitSupport.buildSingleOpUnits(partition, context);
    }
}
