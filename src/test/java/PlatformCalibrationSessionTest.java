import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.session.PlatformCalibrationFamily;
import tuning.session.PlatformCalibrationRequest;
import tuning.session.PlatformCalibrationScorePolicy;
import tuning.session.PlatformCalibrationSession;
import tuning.session.PlatformCalibrationStep;
import tuning.session.PlatformRuntimeProfileGridCandidateSpace;
import tuning.session.PlatformRuntimeProfileMutators;
import tuning.session.TuningPreset;
import tuning.workload.CalibrationWorkloads;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PlatformCalibrationSessionTest {
    @Test
    void platformCalibrationSessionProducesFinalProfileAndPersistsIt() throws Exception {
        ExecutionProfile seed = new ExecutionProfile(
                "seed",
                "seed",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        Path out = Files.createTempFile("platform-calibration-", ".json");
        PlatformCalibrationRequest request = PlatformCalibrationRequest.fromSeedExecutionProfile(
                "test-platform",
                seed,
                List.of(
                        new PlatformCalibrationStep(
                                "matmul-step",
                                PlatformCalibrationFamily.MATMUL,
                                List.of(CalibrationWorkloads.matmulSquare("matmul_step", 16)),
                                TuningPreset.QUICK,
                                base -> new PlatformRuntimeProfileGridCandidateSpace(
                                        base,
                                        List.of(
                                                PlatformRuntimeProfileMutators.blasThreads(List.of(1))
                                        )
                                ),
                                PlatformCalibrationScorePolicy.averageMedianMs()
                        )
                ),
                out
        );

        var result = PlatformCalibrationSession.create(request).run();

        assertEquals("test-platform", result.platformId());
        assertEquals(1, result.steps().size());
        assertNotNull(result.finalProfile());
        assertNotNull(result.finalRuntimeProfile());
        assertTrue(result.steps().getFirst().selectedScore().score() >= 0.0d
                || Double.isFinite(result.steps().getFirst().selectedScore().score()));
        assertTrue(result.persisted());
        assertTrue(Files.size(out) > 0L);
    }
}
