import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.session.PlatformCalibrationDefaults;
import tuning.session.PlatformCalibrationFamily;
import tuning.session.TuningPreset;

import java.nio.file.Path;

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
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.FUSED_ARITHMETIC));
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

        assertEquals(7, request.steps().size());
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.MATMUL));
        assertTrue(request.steps().stream().anyMatch(step -> step.family() == PlatformCalibrationFamily.FUSED_ARITHMETIC));
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
}
