import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.session.PlatformCalibrationDefaults;
import tuning.session.PlatformCalibrationFamily;
import tuning.session.TuningPreset;

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
                config.optimizer.OptimizerConfig.inferenceDefaults(),
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
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.MATMUL));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.ATTENTION_MATMUL));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.CONV2D));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.FUSED_THRESHOLDS));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.FUSED_CHEAP_CONTIGUOUS));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.FUSED_CHEAP_STRIDED));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.FUSED_NON_CHEAP_CONTIGUOUS));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.FUSED_NON_CHEAP_STRIDED));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.ELEMENTWISE_DISPATCH));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.SCHEDULER));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.MATERIALIZATION));
    }

    @Test
    void helperStepFactoriesUseRequestedPreset() {
        var step = PlatformCalibrationDefaults.matmulStep("matmul", TuningPreset.THOROUGH);
        assertEquals(TuningPreset.THOROUGH, step.preset());
        assertEquals(PlatformCalibrationFamily.MATMUL, step.family());
    }

    @Test
    void balancedInferenceFullContainsAllCalibrationFamilies() {
        ExecutionProfile seed = new ExecutionProfile(
                "seed",
                "seed",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var request = PlatformCalibrationDefaults.balancedInferenceFull(
                "test-platform",
                seed,
                Path.of("build", "test-platform-profile.json")
        );

        assertEquals(14, request.steps().size());
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.MATMUL));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.ATTENTION_MATMUL));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.CONV2D));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.FUSED_THRESHOLDS));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.FUSED_CHEAP_CONTIGUOUS));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.FUSED_CHEAP_STRIDED));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.FUSED_NON_CHEAP_CONTIGUOUS));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.FUSED_NON_CHEAP_STRIDED));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.ELEMENTWISE_DISPATCH));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.REDUCTION));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.SCHEDULER));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.MATERIALIZATION));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.NUMERICS));
    }

    @Test
    void thoroughInferenceUsesThoroughPresetAcrossSteps() {
        ExecutionProfile seed = new ExecutionProfile(
                "seed",
                "seed",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
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
        var step = PlatformCalibrationDefaults.matmulStep("matmul", TuningPreset.QUICK);

        PlatformRuntimeProfile f64Base = PlatformRuntimeProfile.fromExecutionProfile(
                "platform-f64",
                "hw",
                "TEST",
                new ExecutionProfile(
                        "seed-f64",
                        "seed-f64",
                        tensor.DataType.FLOAT64,
                        ExecutionMode.FORWARD,
                        config.optimizer.OptimizerConfig.inferenceDefaults(),
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
                        config.optimizer.OptimizerConfig.inferenceDefaults(),
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

        assertTrue(f64Tiles.contains("16x64x32"));
        assertTrue(f64Tiles.contains("32x128x64"));
        assertFalse(f64Tiles.contains("64x256x128"));

        assertTrue(f32Tiles.contains("32x64x64"));
        assertTrue(f32Tiles.contains("64x256x128"));
        assertFalse(f32Tiles.contains("16x64x32"));
    }

    @Test
    void matmulStepGeneratesBlasProviderAndMinWorkCandidates() {
        var step = PlatformCalibrationDefaults.matmulStep("matmul", TuningPreset.QUICK);

        PlatformRuntimeProfile base = PlatformRuntimeProfile.fromExecutionProfile(
                "platform-f64",
                "hw",
                "TEST",
                new ExecutionProfile(
                        "seed-f64",
                        "seed-f64",
                        tensor.DataType.FLOAT64,
                        ExecutionMode.FORWARD,
                        config.optimizer.OptimizerConfig.inferenceDefaults(),
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
    void conv2dStepBuildsMultiWorkloadBlasCalibrationSpace() {
        var step = PlatformCalibrationDefaults.conv2dStep("conv2d", TuningPreset.QUICK);

        PlatformRuntimeProfile base = PlatformRuntimeProfile.fromExecutionProfile(
                "platform-f32",
                "hw",
                "TEST",
                new ExecutionProfile(
                        "seed-f32",
                        "seed-f32",
                        tensor.DataType.FLOAT32,
                        ExecutionMode.FORWARD,
                        config.optimizer.OptimizerConfig.inferenceDefaults(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        var candidates = step.candidateSpaceFactory().create(base).generate(step.workloads().getFirst());

        assertTrue(step.workloads().size() >= 4);
        assertTrue(candidates.stream().anyMatch(candidate ->
                "NONE".equals(candidate.runtimeProfile().matmul().blasProvider().name())));
        assertTrue(candidates.stream().anyMatch(candidate ->
                "OPENBLAS_FFM".equals(candidate.runtimeProfile().matmul().blasProvider().name())
                        && candidate.runtimeProfile().matmul().blasMatmulMinWork() == 50_000L));
    }

    @Test
    void fusedCheapContiguousCalibrationStepGeneratesFamilySpecificAsmWidthCandidates() {
        var step = PlatformCalibrationDefaults.fusedCheapContiguousStep(
                "fused-cheap-contig",
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
                        config.optimizer.OptimizerConfig.inferenceDefaults(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        var candidates = step.candidateSpaceFactory().create(base).generate(step.workloads().getFirst());
        var width8 = candidates.stream()
                .filter(candidate -> "8".equals(candidate.knobAssignments().get("cpu.fusedCheapContiguousAsmVectorWidth")))
                .findFirst()
                .orElseThrow(() -> new AssertionError("Expected cheap-contiguous fused width candidate with width 8."));

        assertEquals(8, width8.runtimeProfile().fused().fusedCheapContiguousAsmVectorWidth());
        assertEquals(base.fused().fusedCheapStridedAsmVectorWidth(), width8.runtimeProfile().fused().fusedCheapStridedAsmVectorWidth());
        assertEquals(base.fused().fusedNonCheapContiguousAsmVectorWidth(), width8.runtimeProfile().fused().fusedNonCheapContiguousAsmVectorWidth());
        assertEquals(base.fused().fusedNonCheapStridedAsmVectorWidth(), width8.runtimeProfile().fused().fusedNonCheapStridedAsmVectorWidth());
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
                        config.optimizer.OptimizerConfig.trainingDefaults(),
                        config.runtime.RuntimeConfig.trainingDefaults(),
                        WorkloadProfile.none()
                )
        );

        var candidates = step.candidateSpaceFactory().create(base).generate(step.workloads().getFirst());
        assertTrue(candidates.stream().anyMatch(candidate -> candidate.knobAssignments().containsKey("cpu.attentionMatMulTileM")));
    }

    @Test
    void fusedCheapStridedCalibrationStepDoesNotGenerateExtendedAsmWidthCandidate() {
        var step = PlatformCalibrationDefaults.fusedCheapStridedStep(
                "fused-cheap-strided",
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
                        config.optimizer.OptimizerConfig.inferenceDefaults(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        var candidates = step.candidateSpaceFactory().create(base).generate(step.workloads().getFirst());
        assertFalse(candidates.stream()
                .anyMatch(candidate -> "8".equals(candidate.knobAssignments().get("cpu.fusedCheapStridedAsmVectorWidth"))));
    }
}
