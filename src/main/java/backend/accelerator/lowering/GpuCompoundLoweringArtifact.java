package backend.accelerator.lowering;

import backend.lowering.LoweredUnitArtifact;

import java.util.Objects;

/**
 * Lowered-unit artifact carrying traceable compound GPU pattern metadata.
 */
public record GpuCompoundLoweringArtifact(
        GpuCompoundRegionSummary summary
) implements LoweredUnitArtifact {
    public GpuCompoundLoweringArtifact {
        Objects.requireNonNull(summary, "summary cannot be null");
    }
}
