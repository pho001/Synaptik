package planning.backend;

import planning.partition.BackendPartitionCapability;
import planning.partition.PartitionTarget;

/**
 * Backend-neutral capability lookup used by compile-time partition planning.
 */
public interface BackendPartitionCapabilityRegistry {
    BackendPartitionCapability partitionCapabilityFor(PartitionTarget target);
}
