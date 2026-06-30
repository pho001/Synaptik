package tuning.calibration;

import runtime.contract.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import tensor.DataType;
import tuning.calibration.progress.PlatformCalibrationProgressListener;
import tuning.measure.MeasurementPolicy;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Immutable request for platform runtime calibration.
 *
 * <p>Platform calibration learns runtime-platform defaults, such as dispatch
 * thresholds or kernel choices, by executing ordered {@link PlatformCalibrationStep}
 * families. Unlike benchmark requests, it generates runtime-profile candidates;
 * unlike graph autotune, it keeps the graph policy fixed and mutates/selects
 * {@link PlatformRuntimeProfile} candidates. Each step starts from the profile
 * selected by the previous step.</p>
 *
 * <p>The session writes the final runtime profile only when
 * {@link #outputProfilePath()} is non-null. Progress listeners are called
 * synchronously from the calibration run thread.</p>
 *
 * <p>Example:</p>
 * <pre>{@code
 * PlatformCalibrationRequest request = new PlatformCalibrationRequest(
 *         "apple-m2",
 *         "cpu-f32",
 *         DataType.FLOAT32,
 *         ExecutionMode.CPU,
 *         graphPolicy,
 *         seedRuntimeProfile,
 *         PlatformCalibrationDefaults.defaultSteps(TuningPreset.BALANCED),
 *         Path.of("profiles/apple-m2-f32.json"),
 *         MeasurementPolicy.defaults(),
 *         PlatformCalibrationProgressListener.noop());
 * PlatformCalibrationResult result = PlatformCalibrationSession.create(request).run();
 * }</pre>
 *
 * @param platformId stable platform identifier; blank values become {@code "platform"}
 * @param profileName profile namespace; blank values become {@code platformId}
 * @param dataType dtype for execution profiles assembled during calibration; required
 * @param executionMode execution mode for assembled profiles; required
 * @param graphPolicy fixed graph policy used during calibration; required
 * @param seedRuntimeProfile initial runtime profile to refine; required
 * @param steps ordered calibration steps; {@code null} becomes empty
 * @param outputProfilePath optional path for saving the selected final runtime profile
 * @param measurement optional measurement override for all steps; {@code null} uses step presets
 * @param progressListener optional progress sink; {@code null} becomes no-op
 */
public record PlatformCalibrationRequest(
        String platformId,
        String profileName,
        DataType dataType,
        ExecutionMode executionMode,
        GraphExecutionPolicy graphPolicy,
        PlatformRuntimeProfile seedRuntimeProfile,
        List<PlatformCalibrationStep> steps,
        Path outputProfilePath,
        MeasurementPolicy measurement,
        PlatformCalibrationProgressListener progressListener
) {
    public PlatformCalibrationRequest {
        platformId = (platformId == null || platformId.isBlank()) ? "platform" : platformId;
        profileName = profileName == null || profileName.isBlank() ? platformId : profileName;
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(executionMode, "executionMode cannot be null");
        Objects.requireNonNull(graphPolicy, "graphPolicy cannot be null");
        Objects.requireNonNull(seedRuntimeProfile, "seedRuntimeProfile cannot be null");
        steps = steps == null ? List.of() : List.copyOf(steps);
        progressListener = progressListener == null ? PlatformCalibrationProgressListener.noop() : progressListener;
    }

    public PlatformCalibrationRequest(
            String platformId,
            String profileName,
            DataType dataType,
            ExecutionMode executionMode,
            GraphExecutionPolicy graphPolicy,
            PlatformRuntimeProfile seedRuntimeProfile,
            List<PlatformCalibrationStep> steps,
            Path outputProfilePath,
            PlatformCalibrationProgressListener progressListener
    ) {
        this(
                platformId,
                profileName,
                dataType,
                executionMode,
                graphPolicy,
                seedRuntimeProfile,
                steps,
                outputProfilePath,
                null,
                progressListener
        );
    }

    /**
     * Creates a calibration request from a legacy full execution profile.
     *
     * <p>The graph policy and runtime profile are extracted from the seed profile.
     * New callers should prefer the canonical constructor so graph policy and
     * platform runtime profile are explicit.</p>
     *
     * @param platformId platform identifier
     * @param seedProfile execution profile to split into graph/runtime parts
     * @param steps ordered calibration steps
     * @param outputProfilePath optional save path for the final runtime profile
     * @return calibration request
     */
    public static PlatformCalibrationRequest fromSeedExecutionProfile(
            String platformId,
            ExecutionProfile seedProfile,
            List<PlatformCalibrationStep> steps,
            Path outputProfilePath
    ) {
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        return new PlatformCalibrationRequest(
                platformId,
                seedProfile.profileName(),
                seedProfile.dataType(),
                seedProfile.mode(),
                GraphExecutionPolicy.fromExecutionProfile(seedProfile),
                PlatformRuntimeProfile.fromExecutionProfile(
                        platformId,
                        platformId,
                        "UNSPECIFIED",
                        seedProfile
                ),
                steps,
                outputProfilePath,
                null,
                null
        );
    }

    /**
     * Creates a calibration request from a legacy full execution profile with a
     * measurement-policy override.
     *
     * @param platformId platform identifier
     * @param seedProfile execution profile to split into graph/runtime parts
     * @param steps ordered calibration steps
     * @param outputProfilePath optional save path for the final runtime profile
     * @param measurement measurement policy used for every step when non-null
     * @return calibration request
     */
    public static PlatformCalibrationRequest fromSeedExecutionProfile(
            String platformId,
            ExecutionProfile seedProfile,
            List<PlatformCalibrationStep> steps,
            Path outputProfilePath,
            MeasurementPolicy measurement
    ) {
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        return new PlatformCalibrationRequest(
                platformId,
                seedProfile.profileName(),
                seedProfile.dataType(),
                seedProfile.mode(),
                GraphExecutionPolicy.fromExecutionProfile(seedProfile),
                PlatformRuntimeProfile.fromExecutionProfile(
                        platformId,
                        platformId,
                        "UNSPECIFIED",
                        seedProfile
                ),
                steps,
                outputProfilePath,
                measurement,
                null
        );
    }
}
