package tuning.calibration.runtime;

/**
 * Factory for a calibration step's runtime candidate space.
 *
 * <p>The calibration session calls this once per step with the runtime profile
 * selected by previous steps. Implementations should avoid retaining mutable
 * state unless they document their own thread-safety guarantees.</p>
 */
@FunctionalInterface
public interface PlatformCalibrationCandidateSpaceFactory {
    /**
     * Creates a candidate space from the current seed runtime profile.
     *
     * @param seedProfile runtime profile currently selected for the platform
     * @return candidate space used by the step
     */
    PlatformRuntimeCandidateSpace create(config.profile.PlatformRuntimeProfile seedProfile);
}
