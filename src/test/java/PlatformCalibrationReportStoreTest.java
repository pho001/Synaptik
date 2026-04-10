import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.report.JsonPlatformCalibrationResultRenderer;
import tuning.report.TextPlatformCalibrationResultRenderer;
import tuning.report.BenchmarkSuiteReport;
import tuning.session.PlatformCalibrationCandidateSummary;
import tuning.session.PlatformCalibrationFamily;
import tuning.session.PlatformCalibrationScore;
import tuning.session.PlatformCalibrationResult;
import tuning.session.PlatformCalibrationStepResult;
import tuning.store.PlatformCalibrationSaveHelper;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PlatformCalibrationReportStoreTest {
    @Test
    void renderersAndSaveHelperProduceArtifacts() throws Exception {
        ExecutionProfile profile = new ExecutionProfile(
                "profile",
                "profile",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        PlatformRuntimeProfile runtimeProfile = PlatformRuntimeProfile.fromExecutionProfile(
                "platform-profile",
                "hardware-key",
                "TEST",
                profile
        );

        PlatformCalibrationResult result = new PlatformCalibrationResult(
                "platform",
                tuning.store.HardwareFingerprint.capture(),
                profile.profileName(),
                GraphExecutionPolicy.fromExecutionProfile(profile),
                runtimeProfile,
                runtimeProfile,
                List.of(new PlatformCalibrationStepResult(
                        "step",
                        PlatformCalibrationFamily.MATMUL,
                        runtimeProfile,
                        new BenchmarkSuiteReport(OffsetDateTime.now(), List.of()),
                        List.of(new PlatformCalibrationCandidateSummary(
                                "profile",
                                java.util.Map.of("candidateName", "profile"),
                                new PlatformCalibrationScore(true, 1.23, 1.23, 1.23, 0.0d, "test")
                        )),
                        new PlatformCalibrationCandidateSummary(
                                "profile",
                                java.util.Map.of("candidateName", "profile"),
                                new PlatformCalibrationScore(true, 1.23, 1.23, 1.23, 0.0d, "test")
                        ),
                        runtimeProfile,
                        profile,
                        new PlatformCalibrationScore(true, 1.23, 1.23, 1.23, 0.0d, "test"),
                        "averageMedianMs"
                )),
                null,
                false,
                OffsetDateTime.now()
        );

        String text = TextPlatformCalibrationResultRenderer.render(result);
        String json = JsonPlatformCalibrationResultRenderer.render(result);
        assertTrue(text.contains("Platform Calibration Result"));
        assertTrue(json.contains("\"platformId\": \"platform\""));

        Path profilePath = Files.createTempFile("platform-profile-", ".json");
        Path jsonPath = Files.createTempFile("platform-calibration-", ".json");
        Path textPath = Files.createTempFile("platform-calibration-", ".txt");

        PlatformCalibrationSaveHelper.saveAll(result, profilePath, jsonPath, textPath);

        assertTrue(Files.size(profilePath) > 0L);
        assertTrue(Files.size(jsonPath) > 0L);
        assertTrue(Files.size(textPath) > 0L);
        assertFalse(Files.readString(textPath).isBlank());
    }
}
