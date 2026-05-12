package tuning.api;

import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
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

    @Test
    void profileDslBuildsNoOptimizationBaseline() {
        ExecutionProfile profile = Synaptik.tuning()
                .profile()
                .name("main-baseline-no-opt-f64")
                .candidate("baseline-no-opt")
                .dtype(DataType.FLOAT64)
                .mode().training()
                .compile().noGraphOptimization()
                .runtime().noOptNoVecNoPar()
                .build();

        assertEquals("main-baseline-no-opt-f64", profile.profileName());
        assertEquals("baseline-no-opt", profile.candidateName());
        assertEquals(DataType.FLOAT64, profile.dataType());
        assertEquals(ExecutionMode.FORWARD_BACKWARD, profile.mode());
        assertEquals(CompileConfig.noGraphOptimizationBaseline(), profile.compile());
        RuntimeConfig expectedRuntime = RuntimeConfig.noOptNoVecNoPar();
        assertEquals(expectedRuntime.blas().provider(), profile.runtime().blas().provider());
        assertEquals(expectedRuntime.kernel().cpu().loopUnrollFactor(), profile.runtime().kernel().cpu().loopUnrollFactor());
        assertEquals(expectedRuntime.kernel().cpu().cheapVectorMinSize(), profile.runtime().kernel().cpu().cheapVectorMinSize());
        assertEquals(expectedRuntime.kernel().cpu().cheapParallelMinSize(), profile.runtime().kernel().cpu().cheapParallelMinSize());
    }

    @Test
    void profileDslBuildsProfileFromCalibratedRuntime() {
        ExecutionProfile seed = profile("seed", DataType.FLOAT32, RuntimeConfig.trainingDefaults());
        PlatformRuntimeProfile runtimeProfile = PlatformRuntimeProfile.fromExecutionProfile(
                "test-platform",
                "test-hardware",
                "quick",
                seed
        );

        ExecutionProfile profile = Synaptik.tuning()
                .profile()
                .name("main-calibrated-runtime-f32")
                .candidate("calibrated-runtime")
                .dtype(DataType.FLOAT32)
                .mode().training()
                .compile().trainingDefaults()
                .runtime().fromPlatformProfile(runtimeProfile)
                .toExecutionProfile();

        assertEquals("main-calibrated-runtime-f32", profile.profileName());
        assertEquals("calibrated-runtime", profile.candidateName());
        assertEquals(DataType.FLOAT32, profile.dataType());
        assertEquals(CompileConfig.training(), profile.compile());
        RuntimeConfig expectedRuntime = runtimeProfile.toRuntimeConfig();
        assertEquals(expectedRuntime.blas().provider(), profile.runtime().blas().provider());
        assertEquals(expectedRuntime.kernel().cpu().matMulTileM(), profile.runtime().kernel().cpu().matMulTileM());
        assertEquals(expectedRuntime.kernel().cpu().fusedCheapVectorMinSize(), profile.runtime().kernel().cpu().fusedCheapVectorMinSize());
        assertEquals(expectedRuntime.kernel().cpu().reductionVectorMinSize(), profile.runtime().kernel().cpu().reductionVectorMinSize());
    }

    @Test
    void profileDslRequiresDtypeOptimizerAndRuntime() {
        assertThrows(
                IllegalStateException.class,
                () -> Synaptik.tuning()
                        .profile()
                        .compile().trainingDefaults()
                        .runtime().trainingDefaults()
                        .build()
        );
        assertThrows(
                IllegalStateException.class,
                () -> Synaptik.tuning()
                        .profile()
                        .dtype(DataType.FLOAT64)
                        .runtime().trainingDefaults()
                        .build()
        );
        assertThrows(
                IllegalStateException.class,
                () -> Synaptik.tuning()
                        .profile()
                        .dtype(DataType.FLOAT64)
                        .compile().trainingDefaults()
                        .build()
        );
    }

    private static ExecutionProfile profile(String name, DataType dtype, RuntimeConfig runtime) {
        return new ExecutionProfile(
                name,
                name,
                dtype,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.training(),
                runtime
        );
    }
}
