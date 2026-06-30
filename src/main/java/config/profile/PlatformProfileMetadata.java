package config.profile;

import runtime.contract.ExecutionMode;
import tensor.DataType;

import java.time.OffsetDateTime;
import java.util.Objects;

/**
 * Metadata attached to a persisted platform runtime profile.
 *
 * <p>The metadata identifies the hardware/profile target, schema versions, creation time, calibration
 * preset, dtype, and execution mode. Blank string fields are normalized to stable placeholders so JSON
 * artifacts remain self-describing even when a caller supplies partial metadata.</p>
 *
 * @param platformProfileId stable id for the platform profile
 * @param hardwareKey hardware fingerprint key used to locate platform-specific artifacts
 * @param frameworkVersion framework version that produced the profile
 * @param plannerSchemaVersion graph/runtime planner schema version
 * @param persistenceSchemaVersion persistence schema version
 * @param createdAtIso ISO timestamp for profile creation
 * @param calibrationPreset preset or seed source used to create the profile
 * @param dataType dtype for which the profile is valid
 * @param executionMode execution mode for which the profile is valid
 */
public record PlatformProfileMetadata(
        String platformProfileId,
        String hardwareKey,
        String frameworkVersion,
        String plannerSchemaVersion,
        String persistenceSchemaVersion,
        String createdAtIso,
        String calibrationPreset,
        DataType dataType,
        ExecutionMode executionMode
) {
    public PlatformProfileMetadata {
        platformProfileId = platformProfileId == null || platformProfileId.isBlank() ? "platform-profile" : platformProfileId;
        hardwareKey = hardwareKey == null || hardwareKey.isBlank() ? "unknown-hardware" : hardwareKey;
        frameworkVersion = frameworkVersion == null || frameworkVersion.isBlank() ? "dev" : frameworkVersion;
        plannerSchemaVersion = plannerSchemaVersion == null || plannerSchemaVersion.isBlank() ? "1" : plannerSchemaVersion;
        persistenceSchemaVersion = persistenceSchemaVersion == null || persistenceSchemaVersion.isBlank() ? "1" : persistenceSchemaVersion;
        createdAtIso = createdAtIso == null || createdAtIso.isBlank() ? OffsetDateTime.now().toString() : createdAtIso;
        calibrationPreset = calibrationPreset == null || calibrationPreset.isBlank() ? "UNSPECIFIED" : calibrationPreset;
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(executionMode, "executionMode cannot be null");
    }
}
