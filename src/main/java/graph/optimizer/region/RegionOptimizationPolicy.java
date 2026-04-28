package graph.optimizer.region;

import graph.optimizer.partition.Partition;

import java.util.List;

/**
 * Backend policy for splitting a partition into execution units.
 */
public interface RegionOptimizationPolicy {
    /**
     * Builds execution units for a partition.
     *
     * @param partition accepted partition
     * @param context region optimization context
     * @return execution units in region order
     */
    List<ExecutionUnit> buildUnits(Partition partition, RegionOptimizationContext context);
}
