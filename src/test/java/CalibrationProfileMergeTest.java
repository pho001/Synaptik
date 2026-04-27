import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import graph.execution.trace.ExecutionTrace;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.calibration.PlatformCalibrationRequest;
import tuning.calibration.PlatformCalibrationScorePolicy;
import tuning.calibration.PlatformCalibrationSession;
import tuning.calibration.PlatformCalibrationStep;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.runtime.PlatformRuntimeProfileMutators;
import tuning.measure.MeasurementPolicy;
import tuning.measure.MeasurementResult;
import tuning.measure.MeasurementStatistics;
import tuning.preset.TuningPreset;
import tuning.store.PlatformRuntimeProfileStore;
import tuning.validate.ValidationResult;
import tuning.workload.CalibrationWorkloads;

import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

public class CalibrationProfileMergeTest {
    @Test
    void singleFamilyCalibrationPreservesUnrelatedRuntimeSections() {
        ExecutionProfile seedProfile = new ExecutionProfile(
                "seed",
                "seed",
                DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        PlatformRuntimeProfile seed = PlatformRuntimeProfile.fromExecutionProfile(
                "platform",
                "hardware",
                "TEST",
                seedProfile
        );
        int nextMatmulThreshold = seed.matmul().matMulParallelMinSize() + 1;
        PlatformCalibrationRequest request = new PlatformCalibrationRequest(
                "platform",
                "single-family",
                DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.profile.GraphExecutionPolicy.inferenceDefaults(),
                seed,
                List.of(new PlatformCalibrationStep(
                        "matmul-only",
                        CalibrationFamilyId.MATMUL,
                        List.of(CalibrationWorkloads.matmulSquare("merge_matmul", 8)),
                        TuningPreset.QUICK,
                        base -> workload -> PlatformRuntimeProfileMutators
                                .matmulParallelThresholds(List.of(nextMatmulThreshold))
                                .variants(base, workload),
                        PlatformCalibrationScorePolicy.averageMedianMs()
                )),
                null,
                new MeasurementPolicy(0, 1, 1, false, false, false, false, false),
                null
        );

        var result = PlatformCalibrationSession.create(
                request,
                (candidate, workload, policy) -> new MeasurementResult(
                        policy,
                        new ExecutionTrace(null, null, null),
                        new MeasurementStatistics(1.0d, 1.0d, 1.0d)
                ),
                (candidate, workloadSpec, workload, policy) -> ValidationResult.skipped(),
                new NoopRuntimeProfileStore()
        ).run();

        PlatformRuntimeProfile selected = result.finalRuntimeProfile();
        assertNotEquals(seed.matmul().matMulParallelMinSize(), selected.matmul().matMulParallelMinSize());
        assertEquals(seed.conv2d(), selected.conv2d());
        assertEquals(seed.fused(), selected.fused());
        assertEquals(seed.elementwiseDispatch(), selected.elementwiseDispatch());
        assertEquals(seed.reduction(), selected.reduction());
        assertEquals(seed.scheduler(), selected.scheduler());
        assertEquals(seed.materialization(), selected.materialization());
        assertEquals(seed.numerics(), selected.numerics());
        assertEquals(seed.accelerator(), selected.accelerator());
    }

    private static final class NoopRuntimeProfileStore implements PlatformRuntimeProfileStore {
        @Override
        public void save(Path path, PlatformRuntimeProfile profile) {
        }

        @Override
        public Optional<PlatformRuntimeProfile> load(Path path) {
            return Optional.empty();
        }
    }
}
