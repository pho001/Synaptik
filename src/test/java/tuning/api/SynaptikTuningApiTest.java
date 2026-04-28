package tuning.api;

import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.benchmark.BenchmarkEntryRole;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.run.CalibrationScope;
import tuning.preset.TuningPreset;
import tuning.workload.StandardWorkloads;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SynaptikTuningApiTest {
    @Test
    void calibrationDslBuildsCalibrationCommand() {
        var command = Synaptik.tuning()
                .calibration()
                .dtypes().single("f32")
                .families().single("matmul")
                .quick()
                .mode().forward()
                .measurement().iterations(1, 2, 3)
                .progress().quiet()
                .color().never()
                .outputRoot(Path.of("build/test-profiles"))
                .toCommand();

        assertEquals(DataType.FLOAT32, command.dataTypes().getFirst());
        assertEquals(CalibrationScope.SINGLE_FAMILY, command.scope());
        assertEquals(CalibrationFamilyId.MATMUL, command.family());
        assertEquals(TuningPreset.QUICK, command.preset());
        assertEquals(ExecutionMode.FORWARD, command.mode());
        assertEquals(1, command.measurement().warmupIters());
        assertEquals(2, command.measurement().measureIters());
        assertEquals(3, command.measurement().repeats());
        assertEquals("quiet", command.progressMode());
        assertEquals("never", command.colorMode());
        assertEquals(Path.of("build/test-profiles"), command.outputRoot());
    }

    @Test
    void calibrationDslRejectsNonCalibrationDtypeEarly() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Synaptik.tuning().calibration().dtype(DataType.INT32)
        );
    }

    @Test
    void benchmarkDslBuildsBenchmarkRequest() {
        ExecutionProfile baseline = profile("baseline", DataType.FLOAT64, RuntimeConfig.noOptNoVecNoPar());
        ExecutionProfile candidate = profile("candidate", DataType.FLOAT64, RuntimeConfig.trainingDefaults());

        var request = Synaptik.tuning()
                .benchmark()
                .workload(StandardWorkloads.matmul("matmul_test", 1, 8, 8, 8))
                .quick()
                .report().hotStepLimit(5).excludeTrace().excludeFailedCandidates().done()
                .compare()
                .baseline("baseline", baseline)
                .candidate("candidate", candidate)
                .toRequest();

        assertEquals("matmul_test", request.workload().name());
        assertEquals(2, request.entries().size());
        assertEquals(BenchmarkEntryRole.BASELINE, request.entries().get(0).role());
        assertEquals(BenchmarkEntryRole.CANDIDATE, request.entries().get(1).role());
        assertEquals(0, request.measurement().warmupIters());
        assertEquals(3, request.measurement().measureIters());
        assertEquals(1, request.measurement().repeats());
        assertEquals(5, request.report().hotStepLimit());
        assertEquals(false, request.report().includeTrace());
        assertEquals(false, request.report().includeFailedCandidates());
    }

    @Test
    void benchmarkDslRequiresWorkload() {
        assertThrows(
                IllegalStateException.class,
                () -> Synaptik.tuning().benchmark().toRequest()
        );
    }

    private static ExecutionProfile profile(String name, DataType dtype, RuntimeConfig runtime) {
        return new ExecutionProfile(
                name,
                name,
                dtype,
                ExecutionMode.FORWARD_BACKWARD,
                OptimizerConfig.trainingDefaults(),
                runtime
        );
    }
}
