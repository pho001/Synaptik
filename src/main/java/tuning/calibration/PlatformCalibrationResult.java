package tuning.calibration;

import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileAssembler;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import tuning.store.HardwareFingerprint;

import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.Objects;

public record PlatformCalibrationResult(
        String platformId,
        HardwareFingerprint hardware,
        String profileName,
        GraphExecutionPolicy graphPolicy,
        PlatformRuntimeProfile seedRuntimeProfile,
        PlatformRuntimeProfile finalRuntimeProfile,
        List<PlatformCalibrationStepResult> steps,
        Path outputProfilePath,
        boolean persisted,
        OffsetDateTime createdAt
) {
    public PlatformCalibrationResult {
        platformId = (platformId == null || platformId.isBlank()) ? "platform" : platformId;
        hardware = hardware == null ? HardwareFingerprint.capture() : hardware;
        profileName = profileName == null || profileName.isBlank() ? platformId : profileName;
        Objects.requireNonNull(graphPolicy, "graphPolicy cannot be null");
        Objects.requireNonNull(seedRuntimeProfile, "seedRuntimeProfile cannot be null");
        Objects.requireNonNull(finalRuntimeProfile, "finalRuntimeProfile cannot be null");
        steps = steps == null ? List.of() : List.copyOf(steps);
        createdAt = createdAt == null ? OffsetDateTime.now() : createdAt;
    }

    public ExecutionProfile seedProfile() {
        return ExecutionProfileAssembler.assemble(
                profileName,
                "seed",
                seedRuntimeProfile.dataType(),
                seedRuntimeProfile.metadata().executionMode(),
                seedRuntimeProfile,
                graphPolicy,
                WorkloadProfile.none()
        );
    }

    public ExecutionProfile finalProfile() {
        return ExecutionProfileAssembler.assemble(
                profileName,
                "platform-calibrated",
                finalRuntimeProfile.dataType(),
                finalRuntimeProfile.metadata().executionMode(),
                finalRuntimeProfile,
                graphPolicy,
                WorkloadProfile.none()
        );
    }
}
