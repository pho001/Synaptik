package tuning.session;

@FunctionalInterface
public interface PlatformCalibrationCandidateSpaceFactory {
    PlatformRuntimeCandidateSpace create(config.profile.PlatformRuntimeProfile seedProfile);
}
