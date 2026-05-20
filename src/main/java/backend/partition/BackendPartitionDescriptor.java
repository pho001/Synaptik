package backend.partition;

import backend.lowering.RegionLowerer;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.partition.BackendPartitionCapability;

import java.util.List;

/**
 * Backend registration descriptor for partition capability and lowering.
 */
public interface BackendPartitionDescriptor {
    /**
     * @return optimizer partition target represented by this backend descriptor
     */
    PartitionTarget target();

    /**
     * @return capability used to decide whether graph regions can run on this backend
     */
    BackendPartitionCapability partitionCapability();

    /**
     * @return region lowerers that can translate optimized regions for this backend
     */
    List<RegionLowerer> lowerers();
}
