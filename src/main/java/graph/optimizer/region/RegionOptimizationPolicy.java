package graph.optimizer.region;

import graph.optimizer.partition.Partition;

import java.util.List;

public interface RegionOptimizationPolicy {
    List<ExecutionUnit> buildUnits(Partition partition, RegionOptimizationContext context);
}
