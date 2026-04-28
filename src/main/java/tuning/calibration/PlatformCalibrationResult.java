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

/**
 * Immutable result of a platform calibration run.
 *
 * <p>{@link #finalRuntimeProfile()} is the runtime profile selected after all
 * steps have run. {@link #seedProfile()} and {@link #finalProfile()} assemble
 * full execution profiles for comparison or downstream benchmarking; they do not
 * persist anything and use {@link WorkloadProfile#none()}.</p>
 *
 * @param platformId calibrated platform identifier
 * @param hardware captured hardware fingerprint
 * @param profileName profile namespace for assembled execution profiles
 * @param graphPolicy graph policy held fixed during calibration
 * @param seedRuntimeProfile initial runtime profile
 * @param finalRuntimeProfile selected runtime profile after all steps
 * @param steps per-step calibration results
 * @param outputProfilePath path used for final profile persistence, if any
 * @param persisted whether the final runtime profile was saved
 * @param createdAt result creation time
 */
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

    /**
     * Assembles a full execution profile from the seed runtime profile.
     *
     * @return seed execution profile for reporting or benchmark comparison
     */
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

    /**
     * Assembles a full execution profile from the calibrated runtime profile.
     *
     * @return calibrated execution profile for reporting or benchmark comparison
     */
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
