package tuning.session;

import config.profile.ExecutionProfile;
import tuning.store.PlatformCalibrationLayout;
import tuning.store.PlatformCalibrationPaths;
import tuning.store.PlatformCalibrationSaveHelper;

import java.nio.file.Path;
import java.util.Objects;

public final class PlatformCalibrationRunner {
    private PlatformCalibrationRunner() {
    }

    public static PlatformCalibrationResult runBalancedInference(Path rootDir, ExecutionProfile seedProfile) {
        Objects.requireNonNull(rootDir, "rootDir cannot be null");
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        PlatformCalibrationLayout layout = PlatformCalibrationPaths.defaultLayout(rootDir, seedProfile);
        PlatformCalibrationRequest request = PlatformCalibrationDefaults.balancedInference(
                layout.platformId(),
                seedProfile,
                layout.profilePath()
        );
        PlatformCalibrationResult result = PlatformCalibrationSession.create(request).run();
        PlatformCalibrationSaveHelper.saveAll(
                result,
                layout.profilePath(),
                layout.jsonReportPath(),
                layout.textReportPath()
        );
        return result;
    }

    public static PlatformCalibrationResult runBalancedInferenceFull(Path rootDir, ExecutionProfile seedProfile) {
        Objects.requireNonNull(rootDir, "rootDir cannot be null");
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        PlatformCalibrationLayout layout = PlatformCalibrationPaths.defaultLayout(rootDir, seedProfile);
        PlatformCalibrationRequest request = PlatformCalibrationDefaults.balancedInferenceFull(
                layout.platformId(),
                seedProfile,
                layout.profilePath()
        );
        PlatformCalibrationResult result = PlatformCalibrationSession.create(request).run();
        PlatformCalibrationSaveHelper.saveAll(
                result,
                layout.profilePath(),
                layout.jsonReportPath(),
                layout.textReportPath()
        );
        return result;
    }

    public static PlatformCalibrationResult runThoroughInference(Path rootDir, ExecutionProfile seedProfile) {
        Objects.requireNonNull(rootDir, "rootDir cannot be null");
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        PlatformCalibrationLayout layout = PlatformCalibrationPaths.defaultLayout(rootDir, seedProfile);
        PlatformCalibrationRequest request = PlatformCalibrationDefaults.thoroughInference(
                layout.platformId(),
                seedProfile,
                layout.profilePath()
        );
        PlatformCalibrationResult result = PlatformCalibrationSession.create(request).run();
        PlatformCalibrationSaveHelper.saveAll(
                result,
                layout.profilePath(),
                layout.jsonReportPath(),
                layout.textReportPath()
        );
        return result;
    }

    public static PlatformCalibrationResult runThoroughInferenceFull(Path rootDir, ExecutionProfile seedProfile) {
        Objects.requireNonNull(rootDir, "rootDir cannot be null");
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        PlatformCalibrationLayout layout = PlatformCalibrationPaths.defaultLayout(rootDir, seedProfile);
        PlatformCalibrationRequest request = PlatformCalibrationDefaults.thoroughInferenceFull(
                layout.platformId(),
                seedProfile,
                layout.profilePath()
        );
        PlatformCalibrationResult result = PlatformCalibrationSession.create(request).run();
        PlatformCalibrationSaveHelper.saveAll(
                result,
                layout.profilePath(),
                layout.jsonReportPath(),
                layout.textReportPath()
        );
        return result;
    }

    public static PlatformCalibrationResult runQuickInference(Path rootDir, ExecutionProfile seedProfile) {
        Objects.requireNonNull(rootDir, "rootDir cannot be null");
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        PlatformCalibrationLayout layout = PlatformCalibrationPaths.defaultLayout(rootDir, seedProfile);
        PlatformCalibrationRequest request = PlatformCalibrationDefaults.quickInference(
                layout.platformId(),
                seedProfile,
                layout.profilePath()
        );
        PlatformCalibrationResult result = PlatformCalibrationSession.create(request).run();
        PlatformCalibrationSaveHelper.saveAll(
                result,
                layout.profilePath(),
                layout.jsonReportPath(),
                layout.textReportPath()
        );
        return result;
    }

    public static PlatformCalibrationResult runBalancedTraining(Path rootDir, ExecutionProfile seedProfile) {
        Objects.requireNonNull(rootDir, "rootDir cannot be null");
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        PlatformCalibrationLayout layout = PlatformCalibrationPaths.defaultLayout(rootDir, seedProfile);
        PlatformCalibrationRequest request = PlatformCalibrationDefaults.balancedTraining(
                layout.platformId(),
                seedProfile,
                layout.profilePath()
        );
        PlatformCalibrationResult result = PlatformCalibrationSession.create(request).run();
        PlatformCalibrationSaveHelper.saveAll(
                result,
                layout.profilePath(),
                layout.jsonReportPath(),
                layout.textReportPath()
        );
        return result;
    }

    public static PlatformCalibrationResult runBalancedTrainingFull(Path rootDir, ExecutionProfile seedProfile) {
        Objects.requireNonNull(rootDir, "rootDir cannot be null");
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        PlatformCalibrationLayout layout = PlatformCalibrationPaths.defaultLayout(rootDir, seedProfile);
        PlatformCalibrationRequest request = PlatformCalibrationDefaults.balancedTrainingFull(
                layout.platformId(),
                seedProfile,
                layout.profilePath()
        );
        PlatformCalibrationResult result = PlatformCalibrationSession.create(request).run();
        PlatformCalibrationSaveHelper.saveAll(
                result,
                layout.profilePath(),
                layout.jsonReportPath(),
                layout.textReportPath()
        );
        return result;
    }

    public static PlatformCalibrationResult runThoroughTraining(Path rootDir, ExecutionProfile seedProfile) {
        Objects.requireNonNull(rootDir, "rootDir cannot be null");
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        PlatformCalibrationLayout layout = PlatformCalibrationPaths.defaultLayout(rootDir, seedProfile);
        PlatformCalibrationRequest request = PlatformCalibrationDefaults.thoroughTraining(
                layout.platformId(),
                seedProfile,
                layout.profilePath()
        );
        PlatformCalibrationResult result = PlatformCalibrationSession.create(request).run();
        PlatformCalibrationSaveHelper.saveAll(
                result,
                layout.profilePath(),
                layout.jsonReportPath(),
                layout.textReportPath()
        );
        return result;
    }

    public static PlatformCalibrationResult runThoroughTrainingFull(Path rootDir, ExecutionProfile seedProfile) {
        Objects.requireNonNull(rootDir, "rootDir cannot be null");
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        PlatformCalibrationLayout layout = PlatformCalibrationPaths.defaultLayout(rootDir, seedProfile);
        PlatformCalibrationRequest request = PlatformCalibrationDefaults.thoroughTrainingFull(
                layout.platformId(),
                seedProfile,
                layout.profilePath()
        );
        PlatformCalibrationResult result = PlatformCalibrationSession.create(request).run();
        PlatformCalibrationSaveHelper.saveAll(
                result,
                layout.profilePath(),
                layout.jsonReportPath(),
                layout.textReportPath()
        );
        return result;
    }

    public static PlatformCalibrationResult runQuickTraining(Path rootDir, ExecutionProfile seedProfile) {
        Objects.requireNonNull(rootDir, "rootDir cannot be null");
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        PlatformCalibrationLayout layout = PlatformCalibrationPaths.defaultLayout(rootDir, seedProfile);
        PlatformCalibrationRequest request = PlatformCalibrationDefaults.quickTraining(
                layout.platformId(),
                seedProfile,
                layout.profilePath()
        );
        PlatformCalibrationResult result = PlatformCalibrationSession.create(request).run();
        PlatformCalibrationSaveHelper.saveAll(
                result,
                layout.profilePath(),
                layout.jsonReportPath(),
                layout.textReportPath()
        );
        return result;
    }
}
