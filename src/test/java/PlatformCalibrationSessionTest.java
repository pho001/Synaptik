import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfileIO;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.PlatformCalibrationDefaults;
import tuning.calibration.PlatformCalibrationRequest;
import tuning.calibration.PlatformCalibrationScorePolicy;
import tuning.calibration.PlatformCalibrationSession;
import tuning.calibration.PlatformCalibrationStep;
import tuning.calibration.runtime.PlatformRuntimeProfileGridCandidateSpace;
import tuning.calibration.runtime.PlatformRuntimeProfileMutators;
import tuning.preset.TuningPreset;
import tuning.workload.CalibrationWorkloads;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
                                CalibrationFamilyId.MATMUL,
                                List.of(CalibrationWorkloads.matmulSquare("matmul_step", 16)),
                                TuningPreset.QUICK,
                                base -> new PlatformRuntimeProfileGridCandidateSpace(
                                        base,
                                        List.of(
                                                PlatformRuntimeProfileMutators.matmulParallelThresholds(List.of(100_000))
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

    @Test
    void platformRuntimeProfilePreservesFusedAsmVectorWidths() {
        ExecutionProfile seed = new ExecutionProfile(
                "seed",
                "seed",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                new config.runtime.RuntimeConfig(
                        new config.backend.KernelTuningConfig(
                                new config.backend.CpuKernelConfig(
                                        4, 32, 32, 32,
                                        256, 256, 256, 256, 256, 256,
                                        50_000, 50_000, 50_000, 50_000, 50_000, 50_000,
                                        16_384,
                                        32_768, 4_096, 2_048, 8_192,
                                        5, 3, 2,
                                        2_048, 4_096, 8_192, 32_768,
                                        4, 2, 8, 1,
                                        config.backend.SumAccuracyMode.FAST,
                                        2_000_000,
                                        config.backend.AttentionMatMulPolicy.AUTO,
                                        config.backend.CpuMatMulMicroKernel.AUTO,
                                        config.backend.CpuMatMulMicroKernel.AUTO,
                                        32, 32, 32
                                ),
                                config.backend.CudaKernelConfig.defaultsInference(),
                                config.backend.OpenClKernelConfig.defaultsInference()
                        ),
                        config.runtime.ApproximationConfig.defaults(),
                        config.runtime.BlasConfig.disabled(),
                        config.runtime.FusedExecutionPolicy.defaultsInference(),
                        new config.runtime.AcceleratorConfig(
                                new config.runtime.AcceleratorBackendConfig(
                                        false,
                                        false,
                                        123L,
                                        new config.runtime.AcceleratorBufferConfig(
                                                config.runtime.AcceleratorBufferBindingMode.OFF,
                                                false,
                                                1_230L
                                        )
                                ),
                                new config.runtime.AcceleratorBackendConfig(
                                        true,
                                        false,
                                        456L,
                                        new config.runtime.AcceleratorBufferConfig(
                                                config.runtime.AcceleratorBufferBindingMode.AUTO,
                                                true,
                                                4_560L
                                        )
                                ),
                                new config.runtime.AcceleratorBackendConfig(
                                        true,
                                        true,
                                        789L,
                                        new config.runtime.AcceleratorBufferConfig(
                                                config.runtime.AcceleratorBufferBindingMode.REQUIRE,
                                                false,
                                                7_890L
                                        )
                                )
                        )
                ),
                WorkloadProfile.none()
        );

        PlatformRuntimeProfile profile = PlatformRuntimeProfile.fromExecutionProfile(
                "platform",
                "hardware",
                "TEST",
                seed
        );
        PlatformRuntimeProfile loaded = PlatformRuntimeProfileIO.fromJsonOrDefault(
                PlatformRuntimeProfileIO.toJson(profile),
                PlatformRuntimeProfile.fromExecutionProfile("fallback", "fallback", "TEST", defaultSeed())
        );

        assertEquals(4, profile.fused().fusedCheapContiguousAsmVectorWidth());
        assertEquals(2, profile.fused().fusedCheapStridedAsmVectorWidth());
        assertEquals(8, profile.fused().fusedNonCheapContiguousAsmVectorWidth());
        assertEquals(1, profile.fused().fusedNonCheapStridedAsmVectorWidth());
        assertEquals(4, loaded.fused().fusedCheapContiguousAsmVectorWidth());
        assertEquals(2, loaded.fused().fusedCheapStridedAsmVectorWidth());
        assertEquals(8, loaded.fused().fusedNonCheapContiguousAsmVectorWidth());
        assertEquals(1, loaded.fused().fusedNonCheapStridedAsmVectorWidth());
        assertEquals(16_384, loaded.materialization().contiguousMaterializeThreshold());
        assertEquals(32_768, loaded.materialization().cheapF64MaterializeThreshold());
        assertEquals(4_096, loaded.materialization().cheapF32MaterializeThreshold());
        assertEquals(2_048, loaded.materialization().cheapBF16MaterializeThreshold());
        assertEquals(8_192, loaded.materialization().whereMaterializeThreshold());
        assertEquals(123L, loaded.accelerator().cuda().minimumEstimatedWork());
        assertEquals(456L, loaded.accelerator().opencl().minimumEstimatedWork());
        assertEquals(789L, loaded.accelerator().metal().minimumEstimatedWork());
        assertEquals(config.runtime.AcceleratorBufferBindingMode.OFF, loaded.accelerator().cuda().buffer().bindingMode());
        assertEquals(config.runtime.AcceleratorBufferBindingMode.AUTO, loaded.accelerator().opencl().buffer().bindingMode());
        assertEquals(config.runtime.AcceleratorBufferBindingMode.REQUIRE, loaded.accelerator().metal().buffer().bindingMode());
        assertEquals(1_230L, loaded.accelerator().cuda().buffer().minimumEstimatedWork());
        assertEquals(4_560L, loaded.accelerator().opencl().buffer().minimumEstimatedWork());
        assertEquals(7_890L, loaded.accelerator().metal().buffer().minimumEstimatedWork());
        assertTrue(loaded.accelerator().metal().requireRuntimeAvailability());
        assertEquals(4, loaded.toRuntimeConfig().kernel().cpu().fusedCheapContiguousAsmVectorWidth());
        assertEquals(2, loaded.toRuntimeConfig().kernel().cpu().fusedCheapStridedAsmVectorWidth());
        assertEquals(8, loaded.toRuntimeConfig().kernel().cpu().fusedNonCheapContiguousAsmVectorWidth());
        assertEquals(1, loaded.toRuntimeConfig().kernel().cpu().fusedNonCheapStridedAsmVectorWidth());
        assertEquals(16_384, loaded.toRuntimeConfig().kernel().cpu().contiguousMaterializeThreshold());
        assertEquals(32_768, loaded.toRuntimeConfig().kernel().cpu().cheapF64MaterializeThreshold());
        assertEquals(4_096, loaded.toRuntimeConfig().kernel().cpu().cheapF32MaterializeThreshold());
        assertEquals(2_048, loaded.toRuntimeConfig().kernel().cpu().cheapBF16MaterializeThreshold());
        assertEquals(8_192, loaded.toRuntimeConfig().kernel().cpu().whereMaterializeThreshold());
        assertEquals(789L, loaded.toRuntimeConfig().accelerator().metal().minimumEstimatedWork());
        assertEquals(
                config.runtime.AcceleratorBufferBindingMode.REQUIRE,
                loaded.toRuntimeConfig().accelerator().metal().buffer().bindingMode()
        );
        assertEquals(7_890L, loaded.toRuntimeConfig().accelerator().metal().buffer().minimumEstimatedWork());
        assertTrue(loaded.toRuntimeConfig().accelerator().metal().requireRuntimeAvailability());
    }

    @Test
    void defaultCalibrationDoesNotIncludeAcceleratorMetalSelectionStep() {
        ExecutionProfile float32Seed = new ExecutionProfile(
                "seed-f32",
                "seed-f32",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        ExecutionProfile float64Seed = new ExecutionProfile(
                "seed-f64",
                "seed-f64",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var float32Request = PlatformCalibrationDefaults.quickInference("platform", float32Seed, Path.of("build", "tmp", "float32-calibration.json"));
        var float64Request = PlatformCalibrationDefaults.quickInference("platform", float64Seed, Path.of("build", "tmp", "float64-calibration.json"));

        assertFalse(float32Request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.METAL_SELECTION));
        assertFalse(float64Request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.METAL_SELECTION));
    }

    @Test
    void strictPlatformProfileLoaderRejectsInvalidAcceleratorBufferMode() {
        PlatformRuntimeProfile fallback = PlatformRuntimeProfile.fromExecutionProfile(
                "fallback",
                "fallback",
                "TEST",
                defaultSeed()
        );
        String json = PlatformRuntimeProfileIO.toJson(fallback)
                .replace("\"metalBufferBindingMode\": \"AUTO\"", "\"metalBufferBindingMode\": \"BROKEN\"");

        IllegalArgumentException thrown = assertThrows(
                IllegalArgumentException.class,
                () -> PlatformRuntimeProfileIO.fromJsonStrict(json, fallback)
        );

        assertTrue(thrown.getMessage().contains("metalBufferBindingMode"));
    }

    @Test
    void strictPlatformProfileLoaderRejectsUnsupportedPersistenceSchemaVersion() {
        PlatformRuntimeProfile fallback = PlatformRuntimeProfile.fromExecutionProfile(
                "fallback",
                "fallback",
                "TEST",
                defaultSeed()
        );
        String json = PlatformRuntimeProfileIO.toJson(fallback)
                .replace("\"persistenceSchemaVersion\": \"1\"", "\"persistenceSchemaVersion\": \"2\"");

        IllegalArgumentException thrown = assertThrows(
                IllegalArgumentException.class,
                () -> PlatformRuntimeProfileIO.fromJsonStrict(json, fallback)
        );

        assertTrue(thrown.getMessage().contains("Unsupported persistenceSchemaVersion"));
    }

    @Test
    void strictPlatformProfileLoaderKeepsLegacyMissingBufferDefaults() {
        PlatformRuntimeProfile fallback = PlatformRuntimeProfile.fromExecutionProfile(
                "fallback",
                "fallback",
                "TEST",
                defaultSeed()
        );
        String json = PlatformRuntimeProfileIO.toJson(fallback)
                .replace("    \"metalBufferBindingMode\": \"AUTO\",\n", "");

        PlatformRuntimeProfile loaded = PlatformRuntimeProfileIO.fromJsonStrict(json, fallback);

        assertEquals(
                config.runtime.AcceleratorBufferConfig.defaults().bindingMode(),
                loaded.accelerator().metal().buffer().bindingMode()
        );
    }

    private static ExecutionProfile defaultSeed() {
        return new ExecutionProfile(
                "default",
                "default",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }
}
