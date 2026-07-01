package backend.accelerator.lowering;

import backend.lowering.LoweredUnitArtifact;

import java.util.List;
import java.util.Objects;

/**
 * Lowered-unit artifact carrying traceable compound GPU pattern and partition unit metadata.
 */
public record GpuCompoundLoweringArtifact(
        GpuCompoundPartitionSummary summary,
        List<GpuPartitionLoweredUnitSummary> units
) implements LoweredUnitArtifact {
    public GpuCompoundLoweringArtifact {
        Objects.requireNonNull(summary, "summary cannot be null");
        units = List.copyOf(units == null ? List.of() : units);
    }

    public GpuCompoundLoweringArtifact(GpuCompoundPartitionSummary summary) {
        this(summary, List.of());
    }
}
