package backend.partition;

import backend.lowering.RegionLowerer;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.RegionLegalityAdapter;

import java.util.List;

public interface BackendPartitionDescriptor {
    PartitionTarget target();

    RegionLegalityAdapter legalityAdapter();

    List<RegionLowerer> lowerers();
}
