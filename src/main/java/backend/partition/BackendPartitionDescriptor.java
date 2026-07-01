package backend.partition;

import backend.lowering.PartitionLowerer;
import planning.partition.PartitionTarget;
import planning.partition.BackendPartitionCapability;

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
     * @return capability used to decide whether graph partitions can run on this backend
     */
    BackendPartitionCapability partitionCapability();

    /**
     * @return partition lowerers that can translate optimized partitions for this backend
     */
    List<PartitionLowerer> lowerers();
}
