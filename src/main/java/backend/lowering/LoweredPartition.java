package backend.lowering;

import planning.partition.ExecutablePartitionPlan;

import java.util.List;
import java.util.Objects;

public record LoweredPartition(
        ExecutablePartitionPlan source,
        List<LoweredExecutionUnit> units
) {
    public LoweredPartition {
        source = Objects.requireNonNull(source, "source cannot be null");
        units = List.copyOf(units == null ? List.of() : units);
    }
}
