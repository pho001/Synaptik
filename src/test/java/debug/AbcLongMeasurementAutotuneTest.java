package debug;

import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileAssembler;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.candidate.ProfileGridCandidateSpace;
import tuning.candidate.ProfileMutators;
import tuning.measure.MeasurementPolicy;
import tuning.report.TextTuningResultRenderer;
import tuning.search.ExhaustiveSearchStrategy;
import tuning.session.AutotuneRequest;
import tuning.session.AutotuneSession;
import tuning.session.TuningPreset;
import tuning.store.JsonFileBestProfileStore;
import tuning.store.JsonFileTuningHistoryStore;
import tuning.store.PersistencePolicy;
import tuning.store.PlatformCalibrationPaths;
import tuning.workload.StandardWorkloads;

import java.nio.file.Path;
import java.util.List;

final class AbcLongMeasurementAutotuneTest {
    private static final MeasurementPolicy MEASUREMENT = new MeasurementPolicy(
            30,
            100,
            3,
            true,
            true,
            true,
            true,
            false
    );

    @Test
    void autotuneF64ForwardBackward() {
        run(DataType.FLOAT64, "f64");
    }

    @Test
    void autotuneF32ForwardBackward() {
        run(DataType.FLOAT32, "f32");
    }

    @Test
    void autotuneBF16ForwardBackward() {
        run(DataType.BFLOAT16, "bf16");
    }

    private static void run(DataType dataType, String dtypeId) {
        ExecutionProfile seed = calibratedAbcSeed(dataType, dtypeId);
        var request = AutotuneRequest.fromSeedExecutionProfile(
                StandardWorkloads.abcSequenceMatmulBlasBenchmark("abc_sequence_matmul_" + dtypeId),
                seed,
                new ProfileGridCandidateSpace(seed, List.of(ProfileMutators.constrainedStageOrderSpace())),
                MEASUREMENT,
                TuningPreset.BALANCED.autotuneValidation(),
                TuningPreset.BALANCED.autotuneSearch(),
                tuningPersistence(dtypeId),
                null
        );

        var result = AutotuneSession.create(
                request,
                new ExhaustiveSearchStrategy(),
                new tuning.measure.DefaultMeasurementEngine(),
                new tuning.validate.DefaultValidationEngine(),
                new JsonFileBestProfileStore(),
                new JsonFileTuningHistoryStore()
        ).run();

        System.out.println();
        System.out.println("ABC_LONG_MEASUREMENT_AUTOTUNE :: " + dtypeId);
        System.out.println(TextTuningResultRenderer.render(result));
    }

    private static ExecutionProfile calibratedAbcSeed(DataType dataType, String dtypeId) {
        ExecutionProfile trainingSeed = new ExecutionProfile(
                "platform-seed-" + dtypeId + "-training",
                "platform-seed-" + dtypeId + "-training",
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                OptimizerConfig.trainingDefaults(),
                config.runtime.RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );

        String platformId = PlatformCalibrationPaths.platformId(tuning.store.HardwareFingerprint.capture());
        Path calibrationPath = Path.of("profiles", "platform", platformId, "calibration", dtypeId + "-forward-backward.json");
        PlatformRuntimeProfile fallback = PlatformRuntimeProfile.fromExecutionProfile(
                platformId,
                tuning.store.HardwareFingerprint.capture().key(),
                "fallback",
                trainingSeed
        );
        PlatformRuntimeProfile runtimeProfile = PlatformRuntimeProfileIO.loadOrDefault(calibrationPath, fallback);
        return ExecutionProfileAssembler.assemble(
                "abc-" + dtypeId + "-calibrated",
                "abc-" + dtypeId + "-calibrated",
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                runtimeProfile,
                GraphExecutionPolicy.trainingDefaults(),
                WorkloadProfile.none()
        );
    }

    private static PersistencePolicy tuningPersistence(String dtypeId) {
        String platformId = PlatformCalibrationPaths.platformId(tuning.store.HardwareFingerprint.capture());
        Path root = Path.of("profiles", "platform", platformId, "tuning", "abc");
        return new PersistencePolicy(
                true,
                true,
                root.resolve(dtypeId + "-best-profile.json"),
                root.resolve(dtypeId + "-history.jsonl")
        );
    }
}
