import runtime.contract.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.calibration.progress.PlatformCalibrationProgressEvent;
import tuning.calibration.progress.PlatformCalibrationProgressPhase;
import tuning.calibration.PlatformCalibrationRequest;
import tuning.calibration.PlatformCalibrationSession;
import tuning.calibration.PlatformCalibrationStep;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.PlatformCalibrationScorePolicy;
import tuning.calibration.runtime.PlatformRuntimeProfileGridCandidateSpace;
import tuning.calibration.runtime.PlatformRuntimeProfileMutators;
import tuning.preset.TuningPreset;
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
                config.compile.CompileConfig.inference(),
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
                                CalibrationFamilyId.MATMUL,
                                List.of(CalibrationWorkloads.matmulSquare("matmul_step", 16)),
                                TuningPreset.QUICK,
                                profile -> new PlatformRuntimeProfileGridCandidateSpace(
                                        profile,
                                        List.of(PlatformRuntimeProfileMutators.matmulParallelThresholds(List.of(100_000)))
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
