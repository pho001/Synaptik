import runtime.contract.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.calibration.PlatformCalibrationDefaults;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.preset.TuningPreset;

import java.nio.file.Path;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PlatformCalibrationDefaultsTest {
    @Test
    void balancedInferenceBuildsNonEmptyCalibrationPlan() {
        ExecutionProfile seed = new ExecutionProfile(
                "seed",
                "seed",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var request = PlatformCalibrationDefaults.balancedInference(
                "test-platform",
                seed,
                Path.of("build", "test-platform-profile.json")
        );

        assertEquals("test-platform", request.platformId());
        assertFalse(request.steps().isEmpty());
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.MATMUL));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.MATMUL));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.MATMUL));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.ATTENTION_MATMUL));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.FUSED_DISPATCH));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.FUSED_ASM_WIDTH));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.ELEMENTWISE_DISPATCH));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.SCHEDULER));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.MATERIALIZATION));
    }

    @Test
    void helperStepFactoriesUseRequestedPreset() {
        var step = PlatformCalibrationDefaults.matmulJavaStep("matmul", TuningPreset.THOROUGH);
        assertEquals(TuningPreset.THOROUGH, step.preset());
        assertEquals(CalibrationFamilyId.MATMUL, step.family());
    }

    @Test
    void balancedInferenceFullContainsAllCalibrationFamilies() {
        ExecutionProfile seed = new ExecutionProfile(
                "seed",
                "seed",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var request = PlatformCalibrationDefaults.balancedInferenceFull(
                "test-platform",
                seed,
                Path.of("build", "test-platform-profile.json")
        );

        assertEquals(12, request.steps().size());
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.MATMUL));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.MATMUL));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.MATMUL));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.ATTENTION_MATMUL));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.FUSED_DISPATCH));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.FUSED_ASM_WIDTH));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.ELEMENTWISE_DISPATCH));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.REDUCTION));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.SCHEDULER));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.MATERIALIZATION));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == CalibrationFamilyId.MATERIALIZATION));
        assertEquals(
                2L,
                request.steps().stream().filter(step -> step.family() == CalibrationFamilyId.MATERIALIZATION).count()
        );
    }

    @Test
    void thoroughInferenceUsesThoroughPresetAcrossSteps() {
        ExecutionProfile seed = new ExecutionProfile(
                "seed",
                "seed",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var request = PlatformCalibrationDefaults.thoroughInference(
                "test-platform",
                seed,
                Path.of("build", "test-platform-profile.json")
        );

        assertFalse(request.steps().isEmpty());
        assertTrue(request.steps().stream().allMatch(step -> step.preset() == TuningPreset.THOROUGH));
    }

    @Test
    void matmulStepGeneratesDtypeSpecificTileCandidates() {
        var step = PlatformCalibrationDefaults.matmulJavaStep("matmul", TuningPreset.QUICK);

        PlatformRuntimeProfile f64Base = PlatformRuntimeProfile.fromExecutionProfile(
                "platform-f64",
                "hw",
                "TEST",
                new ExecutionProfile(
                        "seed-f64",
                        "seed-f64",
                        tensor.DataType.FLOAT64,
                        ExecutionMode.FORWARD,
                        config.compile.CompileConfig.inference(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );
        var f64Candidates = step.candidateSpaceFactory().create(f64Base).generate(step.workloads().getFirst());
        Set<String> f64Tiles = f64Candidates.stream()
                .filter(candidate -> candidate.knobAssignments().containsKey("cpu.matMulTileM"))
                .map(candidate -> candidate.runtimeProfile().matmul().matMulTileM()
                        + "x" + candidate.runtimeProfile().matmul().matMulTileN()
                        + "x" + candidate.runtimeProfile().matmul().matMulTileK())
                .collect(Collectors.toSet());

        PlatformRuntimeProfile f32Base = PlatformRuntimeProfile.fromExecutionProfile(
                "platform-f32",
                "hw",
                "TEST",
                new ExecutionProfile(
                        "seed-f32",
                        "seed-f32",
                        tensor.DataType.FLOAT32,
                        ExecutionMode.FORWARD,
                        config.compile.CompileConfig.inference(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );
        var f32Candidates = step.candidateSpaceFactory().create(f32Base).generate(step.workloads().getFirst());
        Set<String> f32Tiles = f32Candidates.stream()
                .filter(candidate -> candidate.knobAssignments().containsKey("cpu.matMulTileM"))
                .map(candidate -> candidate.runtimeProfile().matmul().matMulTileM()
                        + "x" + candidate.runtimeProfile().matmul().matMulTileN()
                        + "x" + candidate.runtimeProfile().matmul().matMulTileK())
                .collect(Collectors.toSet());

        PlatformRuntimeProfile bf16Base = PlatformRuntimeProfile.fromExecutionProfile(
                "platform-bf16",
                "hw",
                "TEST",
                new ExecutionProfile(
                        "seed-bf16",
                        "seed-bf16",
                        tensor.DataType.BFLOAT16,
                        ExecutionMode.FORWARD,
                        config.compile.CompileConfig.inference(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );
        var bf16Candidates = step.candidateSpaceFactory().create(bf16Base).generate(step.workloads().getFirst());
        Set<String> bf16Tiles = bf16Candidates.stream()
                .filter(candidate -> candidate.knobAssignments().containsKey("cpu.matMulTileM"))
                .map(candidate -> candidate.runtimeProfile().matmul().matMulTileM()
                        + "x" + candidate.runtimeProfile().matmul().matMulTileN()
                        + "x" + candidate.runtimeProfile().matmul().matMulTileK())
                .collect(Collectors.toSet());

        assertTrue(f64Tiles.contains("16x64x32"));
        assertTrue(f64Tiles.contains("32x128x64"));
        assertFalse(f64Tiles.contains("64x256x128"));

        assertTrue(bf16Tiles.contains("16x64x64"));
        assertTrue(bf16Tiles.contains("16x128x64"));
        assertTrue(bf16Tiles.contains("64x128x64"));
        assertFalse(bf16Tiles.contains("64x128x128"));

        assertTrue(f32Tiles.contains("32x64x64"));
        assertTrue(f32Tiles.contains("64x256x128"));
        assertFalse(f32Tiles.contains("16x64x32"));
    }

    @Test
    void matmulStepIncludesProjectionWorkloadsCloseToAbcHotPath() {
        var step = PlatformCalibrationDefaults.matmulJavaStep("matmul", TuningPreset.QUICK);

        assertTrue(step.workloads().size() >= 5);
        assertTrue(step.workloads().stream().anyMatch(workload -> workload.name().contains("projection_wide")));
        assertTrue(step.workloads().stream().anyMatch(workload -> workload.name().contains("projection_tall")));
        assertTrue(step.workloads().stream().anyMatch(workload -> workload.name().contains("workload_large")));
    }

    @Test
    void matmulStepGeneratesBlasProviderAndMinWorkCandidates() {
        var step = PlatformCalibrationDefaults.matmulBlasDispatchStep("matmul", TuningPreset.QUICK);

        PlatformRuntimeProfile base = PlatformRuntimeProfile.fromExecutionProfile(
                "platform-f64",
                "hw",
                "TEST",
                new ExecutionProfile(
                        "seed-f64",
                        "seed-f64",
                        tensor.DataType.FLOAT64,
                        ExecutionMode.FORWARD,
                        config.compile.CompileConfig.inference(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        var candidates = step.candidateSpaceFactory().create(base).generate(step.workloads().getFirst());

        assertTrue(candidates.stream().anyMatch(candidate ->
                "NONE".equals(candidate.runtimeProfile().matmul().blasProvider().name())));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "OPENBLAS_FFM".equals(candidate.runtimeProfile().matmul().blasProvider().name())
                        && candidate.runtimeProfile().matmul().blasMatmulMinWork() == 1_000_000L));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "OPENBLAS_FFM".equals(candidate.runtimeProfile().matmul().blasProvider().name())
                        && candidate.runtimeProfile().matmul().blasMatmulMinWork() == 4_000_000L));
    }

    @Test
    void bfloat16MatmulBlasWideStepGeneratesWideShapeRatioCandidates() {
        var step = PlatformCalibrationDefaults.matmulBlasWideDispatchStep("matmul-wide", TuningPreset.QUICK);

        PlatformRuntimeProfile base = PlatformRuntimeProfile.fromExecutionProfile(
                "platform-bf16",
                "hw",
                "TEST",
                new ExecutionProfile(
                        "seed-bf16",
                        "seed-bf16",
                        tensor.DataType.BFLOAT16,
                        ExecutionMode.FORWARD,
                        config.compile.CompileConfig.inference(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        var candidates = step.candidateSpaceFactory().create(base).generate(step.workloads().getFirst());

        assertEquals(CalibrationFamilyId.MATMUL, step.family());
        assertTrue(step.workloads().size() >= 4);
        assertTrue(step.workloads().stream().allMatch(workload -> workload.name().contains("wide")));
        assertEquals("weightedGeometricMeanWithWorstBucketPenalty", step.scorePolicy().metricName());
        assertTrue(candidates.stream().anyMatch(candidate ->
                Double.compare(candidate.runtimeProfile().matmul().f32WideMaxNOverK(), 8.0d) == 0));
        assertTrue(candidates.stream().anyMatch(candidate ->
                Double.compare(candidate.runtimeProfile().matmul().f32WideMaxNOverK(), 12.0d) == 0));
    }

    @Test
    void fusedAsmWidthCalibrationStepGeneratesCanonicalWidthCandidates() {
        var step = PlatformCalibrationDefaults.fusedAsmWidthStep(
                "fused-asm-width",
                TuningPreset.QUICK,
                tensor.DataType.FLOAT32
        );

        PlatformRuntimeProfile base = PlatformRuntimeProfile.fromExecutionProfile(
                "platform-f32",
                "hw",
                "TEST",
                new ExecutionProfile(
                        "seed-f32",
                        "seed-f32",
                        tensor.DataType.FLOAT32,
                        ExecutionMode.FORWARD,
                        config.compile.CompileConfig.inference(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        var candidates = step.candidateSpaceFactory().create(base).generate(step.workloads().getFirst());
        var width8 = candidates.stream()
                .filter(candidate -> "8".equals(candidate.knobAssignments().get("cpu.fusedAsmVectorWidth")))
                .findFirst()
                .orElseThrow(() -> new AssertionError("Expected fused width candidate with width 8."));

        assertEquals(8, width8.runtimeProfile().fused().fusedAsmVectorWidth());
    }

    @Test
    void attentionMatmulStepGeneratesAttentionSpecificTileCandidates() {
        var step = PlatformCalibrationDefaults.attentionMatmulStep("attention-matmul", TuningPreset.QUICK);

        PlatformRuntimeProfile base = PlatformRuntimeProfile.fromExecutionProfile(
                "platform-f32",
                "hw",
                "TEST",
                new ExecutionProfile(
                        "seed-f32",
                        "seed-f32",
                        tensor.DataType.FLOAT32,
                        ExecutionMode.FORWARD_BACKWARD,
                        config.compile.CompileConfig.training(),
                        config.runtime.RuntimeConfig.trainingDefaults(),
                        WorkloadProfile.none()
                )
        );

        var candidates = step.candidateSpaceFactory().create(base).generate(step.workloads().getFirst());
        assertTrue(candidates.stream().anyMatch(candidate -> candidate.knobAssignments().containsKey("cpu.attentionMatMulTileM")));
    }

    @Test
    void fusedAsmWidthCalibrationStepKeepsSingleCanonicalKnob() {
        var step = PlatformCalibrationDefaults.fusedAsmWidthStep(
                "fused-asm-width",
                TuningPreset.QUICK,
                tensor.DataType.FLOAT32
        );

        PlatformRuntimeProfile base = PlatformRuntimeProfile.fromExecutionProfile(
                "platform-f32",
                "hw",
                "TEST",
                new ExecutionProfile(
                        "seed-f32",
                        "seed-f32",
                        tensor.DataType.FLOAT32,
                        ExecutionMode.FORWARD,
                        config.compile.CompileConfig.inference(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        var candidates = step.candidateSpaceFactory().create(base).generate(step.workloads().getFirst());
        assertTrue(candidates.stream()
                .allMatch(candidate -> candidate.knobAssignments().keySet().equals(Set.of("cpu.fusedAsmVectorWidth"))));
    }

    @Test
    void fusedAsmWidthCalibrationStepCoversContiguousStridedAndRationalPatterns() {
        var step = PlatformCalibrationDefaults.fusedAsmWidthStep(
                "fused-asm-width",
                TuningPreset.QUICK,
                tensor.DataType.BFLOAT16
        );

        assertEquals(5, step.workloads().size());
        assertTrue(step.workloads().stream().anyMatch(workload -> workload.name().contains("cheap_contiguous")));
        assertTrue(step.workloads().stream().anyMatch(workload -> workload.name().contains("cheap_strided")));
        assertTrue(step.workloads().stream().anyMatch(workload -> workload.name().contains("transcendental")));
        assertTrue(step.workloads().stream().anyMatch(workload -> workload.name().contains("affine_rational")));
    }
}
