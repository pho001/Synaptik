import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.session.PlatformCalibrationProgressEvent;
import tuning.session.PlatformCalibrationProgressPhase;
import tuning.session.PlatformCalibrationRequest;
import tuning.session.PlatformCalibrationSession;
import tuning.session.PlatformCalibrationStep;
import tuning.session.PlatformCalibrationFamily;
import tuning.session.PlatformCalibrationScorePolicy;
import tuning.session.PlatformRuntimeProfileGridCandidateSpace;
import tuning.session.PlatformRuntimeProfileMutators;
import tuning.session.TuningPreset;
import tuning.workload.CalibrationWorkloads;

import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class PlatformCalibrationProgressListenerTest {
    @Test
    void platformCalibrationEmitsProgressEventsAcrossLifecycle() throws Exception {
        ExecutionProfile seed = new ExecutionProfile(
                "seed",
                "seed",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        List<PlatformCalibrationProgressEvent> events = new ArrayList<>();
        PlatformCalibrationRequest base = PlatformCalibrationRequest.fromSeedExecutionProfile(
                "test-platform",
                seed,
                List.of(
                        new PlatformCalibrationStep(
                                "matmul-step",
                                PlatformCalibrationFamily.MATMUL,
                                List.of(CalibrationWorkloads.matmulSquare("matmul_step", 16)),
                                TuningPreset.QUICK,
                                profile -> new PlatformRuntimeProfileGridCandidateSpace(
                                        profile,
                                        List.of(PlatformRuntimeProfileMutators.blasThreads(List.of(1)))
                                ),
                                PlatformCalibrationScorePolicy.averageMedianMs()
                        )
                ),
                Files.createTempFile("platform-calibration-progress-", ".json")
        );
        PlatformCalibrationRequest request = new PlatformCalibrationRequest(
                base.platformId(),
                base.profileName(),
                base.dataType(),
                base.executionMode(),
                base.graphPolicy(),
                base.seedRuntimeProfile(),
                base.steps(),
                base.outputProfilePath(),
                events::add
        );

        PlatformCalibrationSession.create(request).run();

        assertTrue(events.stream().anyMatch(e -> e.phase() == PlatformCalibrationProgressPhase.STARTED));
        assertTrue(events.stream().anyMatch(e -> e.phase() == PlatformCalibrationProgressPhase.FAMILY_STARTED));
        assertTrue(events.stream().anyMatch(e -> e.phase() == PlatformCalibrationProgressPhase.WORKLOAD_STARTED));
        assertTrue(events.stream().anyMatch(e -> e.phase() == PlatformCalibrationProgressPhase.CANDIDATE_VALIDATING));
        assertTrue(events.stream().anyMatch(e -> e.phase() == PlatformCalibrationProgressPhase.CANDIDATE_MEASURING
                || e.phase() == PlatformCalibrationProgressPhase.CANDIDATE_INVALID
                || e.phase() == PlatformCalibrationProgressPhase.CANDIDATE_FAILED));
        assertTrue(events.stream().anyMatch(e -> e.phase() == PlatformCalibrationProgressPhase.CANDIDATE_SCORED));
        assertTrue(events.stream().anyMatch(e -> e.phase() == PlatformCalibrationProgressPhase.FAMILY_COMPLETED));
        assertTrue(events.stream().anyMatch(e -> e.phase() == PlatformCalibrationProgressPhase.COMPLETED));
    }
}
