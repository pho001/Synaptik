import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.calibration.run.CalibrationCommand;
import tuning.calibration.store.CalibrationArtifactLayout;
import tuning.calibration.store.CalibrationRunManifest;
import tuning.calibration.store.CalibrationRunStore;
import tuning.preset.TuningPreset;
import tuning.store.HardwareFingerprint;

import java.nio.file.Files;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CalibrationRunStoreTest {
    @Test
    void failedRunManifestDoesNotPublishLatestProfile() throws Exception {
        var root = Files.createTempDirectory("calibration-run-store-");
        var hardware = HardwareFingerprint.capture();
        var layout = CalibrationArtifactLayout.of(root, "test-platform");
        var store = new CalibrationRunStore(layout);
        var command = new CalibrationCommand(
                List.of(DataType.FLOAT64),
                tuning.calibration.family.CalibrationFamilyId.MATMUL,
                tuning.calibration.run.CalibrationScope.SINGLE_FAMILY,
                TuningPreset.QUICK,
                ExecutionMode.FORWARD_BACKWARD,
                null,
                "never",
                "quiet",
                root,
                false
        );
        var manifest = CalibrationRunManifest.started(
                "failed-run",
                "test-platform",
                hardware,
                command,
                layout.runRoot("failed-run")
        );

        store.writeManifest(manifest);
        store.writeManifest(manifest.failed());

        assertFalse(Files.exists(layout.latestProfilePath("f64", "forward-backward")));
        assertFalse(Files.exists(layout.latestManifestPath("f64", "forward-backward")));
    }

    @Test
    void completedRunPublishesLatestProfileAndManifest() throws Exception {
        var root = Files.createTempDirectory("calibration-run-store-");
        var hardware = HardwareFingerprint.capture();
        var layout = CalibrationArtifactLayout.of(root, "test-platform");
        var store = new CalibrationRunStore(layout);
        var command = new CalibrationCommand(
                List.of(DataType.FLOAT64),
                tuning.calibration.family.CalibrationFamilyId.MATMUL,
                tuning.calibration.run.CalibrationScope.SINGLE_FAMILY,
                TuningPreset.QUICK,
                ExecutionMode.FORWARD_BACKWARD,
                null,
                "never",
                "quiet",
                root,
                false
        );
        var manifest = CalibrationRunManifest.started(
                "completed-run",
                "test-platform",
                hardware,
                command,
                layout.runRoot("completed-run")
        ).completed();

        store.writeManifest(manifest);
        store.publishLatest(manifest, "f64", "forward-backward", runtimeProfile());

        assertTrue(Files.exists(layout.latestProfilePath("f64", "forward-backward")));
        assertTrue(Files.readString(layout.latestManifestPath("f64", "forward-backward")).contains("\"status\": \"completed\""));
    }

    private static PlatformRuntimeProfile runtimeProfile() {
        return PlatformRuntimeProfile.fromExecutionProfile(
                "test-platform",
                "hardware",
                "TEST",
                new ExecutionProfile(
                        "seed",
                        "seed",
                        DataType.FLOAT64,
                        ExecutionMode.FORWARD_BACKWARD,
                        config.optimizer.OptimizerConfig.trainingDefaults(),
                        config.runtime.RuntimeConfig.trainingDefaults(),
                        WorkloadProfile.none()
                )
        );
    }
}
