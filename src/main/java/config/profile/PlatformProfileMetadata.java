package config.profile;

import backend.runtime.ExecutionMode;
import tensor.DataType;

import java.time.OffsetDateTime;
import java.util.Objects;

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
