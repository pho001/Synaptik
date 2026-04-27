package graph.optimizer.region;

import graph.optimizer.partition.Partition;

public interface RegionOptimizer {
    OptimizedRegion optimize(Partition partition, RegionOptimizationContext context);
}
