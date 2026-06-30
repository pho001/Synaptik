package backend.lowering;

import planning.partition.PartitionTarget;

import java.util.List;

public record LoweredRegion(
        String regionId,
        PartitionTarget target,
        List<LoweredExecutionUnit> units
) {
    public LoweredRegion {
        regionId = regionId == null ? "" : regionId;
        target = target == null ? PartitionTarget.NONE : target;
        units = List.copyOf(units == null ? List.of() : units);
    }
}
