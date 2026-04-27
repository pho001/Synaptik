import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.store.HardwareFingerprint;
import tuning.calibration.store.PlatformCalibrationLayout;
import tuning.calibration.store.PlatformCalibrationPaths;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PlatformCalibrationPathsTest {
    @Test
    void defaultLayoutBuildsExpectedPathsFromHardwareAndProfile() {
        ExecutionProfile seed = new ExecutionProfile(
                "seed",
                "seed",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD_BACKWARD,
                config.optimizer.OptimizerConfig.trainingDefaults(),
                config.runtime.RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );

        HardwareFingerprint hardware = new HardwareFingerprint(
                "macOS",
                "arm64",
                "temurin",
                "apple",
                10
        );

        PlatformCalibrationLayout layout = PlatformCalibrationPaths.defaultLayout(
                Path.of("build", "calibration"),
                seed,
                hardware
        );

        assertEquals("macos-arm64-apple-10c", layout.platformId());
        assertTrue(layout.profilePath().toString().endsWith("profiles/platform/macos-arm64-apple-10c/f64-forward-backward.json"));
        assertTrue(layout.jsonReportPath().toString().endsWith("reports/platform/macos-arm64-apple-10c/calibration-f64-forward-backward.json"));
        assertTrue(layout.textReportPath().toString().endsWith("reports/platform/macos-arm64-apple-10c/calibration-f64-forward-backward.txt"));
    }
}
