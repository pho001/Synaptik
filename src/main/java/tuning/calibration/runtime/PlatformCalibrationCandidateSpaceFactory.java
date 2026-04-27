package tuning.calibration.runtime;

@FunctionalInterface
public interface PlatformCalibrationCandidateSpaceFactory {
    PlatformRuntimeCandidateSpace create(config.profile.PlatformRuntimeProfile seedProfile);
}
