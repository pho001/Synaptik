package backend.partition;

import backend.lowering.RegionLowerer;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.partition.RegionLegalityAdapter;

import java.util.List;

/**
 * Backend registration descriptor for partition legality and lowering.
 */
public interface BackendPartitionDescriptor {
    /**
     * @return optimizer partition target represented by this backend descriptor
     */
    PartitionTarget target();

    /**
     * @return legality adapter used to decide whether graph regions can run on this backend
     */
    RegionLegalityAdapter legalityAdapter();

    /**
     * @return region lowerers that can translate optimized regions for this backend
     */
    List<RegionLowerer> lowerers();
}
