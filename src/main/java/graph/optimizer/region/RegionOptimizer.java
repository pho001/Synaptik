package graph.optimizer.region;

import graph.optimizer.partition.Partition;

/**
 * Converts an accepted partition into an optimized region plan.
 */
public interface RegionOptimizer {
    /**
     * Optimizes a partition for execution and value transport.
     *
     * @param partition accepted partition
     * @param context region optimization context
     * @return optimized region
     */
    OptimizedRegion optimize(Partition partition, RegionOptimizationContext context);
}
