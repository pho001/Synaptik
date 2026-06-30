import runtime.contract.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.candidate.ListCandidateSpace;
import tuning.benchmark.BenchmarkEntry;
import tuning.autotune.TuningDefaults;
import tuning.preset.TuningPreset;
import tuning.autotune.WorkloadPresetFamily;
import tuning.validate.ValidationToleranceProfile;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TuningDefaultsTest {
    @Test
    void quickAutotuneBuildsExpectedPolicies() {
        var workload = new tuning.workload.TensorRootWorkloadSpec(
                "defaults",
                tuning.workload.WorkloadKind.GENERIC,
                environment -> tensor.Tensor.scalar(1.0)
        );
        var candidate = new tuning.candidate.Candidate("base", profile("base"));

        var request = TuningDefaults.quickAutotune(
                workload,
                profile("seed"),
                new ListCandidateSpace(List.of(candidate))
        );

        assertEquals(0, request.measurement().warmupIters());
        assertEquals(3, request.measurement().measureIters());
        assertEquals(16, request.search().maxCandidates());
        assertTrue(!request.persistence().persistBestProfile());
    }

    @Test
    void thoroughAutotuneEnablesGradientValidationAndTrace() {
        var workload = new tuning.workload.TensorRootWorkloadSpec(
                "defaults-thorough",
                tuning.workload.WorkloadKind.GENERIC,
                environment -> tensor.Tensor.scalar(1.0)
        );
        var request = TuningDefaults.thoroughAutotune(
                workload,
                profile("seed"),
                new ListCandidateSpace(List.of()),
                tuning.store.PersistencePolicy.disabled()
        );

        assertTrue(request.validation().requireGradientMatch());
        assertTrue(request.measurement().captureStepTrace());
        assertEquals(96, request.search().maxCandidates());
    }

    @Test
    void presetBenchmarkUsesBalancedPolicies() {
        var workload = new tuning.workload.TensorRootWorkloadSpec(
                "defaults-benchmark",
                tuning.workload.WorkloadKind.GENERIC,
                environment -> tensor.Tensor.scalar(1.0)
        );
        var candidate = BenchmarkEntry.candidate("base", profile("base"));

        var request = TuningDefaults.benchmark(TuningPreset.BALANCED, workload, List.of(candidate));

        assertEquals(4, request.measurement().warmupIters());
        assertEquals(8, request.measurement().measureIters());
        assertEquals(ValidationToleranceProfile.BALANCED_DTYPE_AWARE, request.validation().toleranceProfile());
        assertEquals(3e-6, request.validation().absTolerance(tensor.DataType.FLOAT32));
        assertEquals(2e-3, request.validation().absTolerance(tensor.DataType.BFLOAT16));
        assertEquals(1, request.entries().size());
    }

    @Test
    void quickAndThoroughValidationUseDifferentDTypeAwareProfiles() {
        assertEquals(ValidationToleranceProfile.QUICK_DTYPE_AWARE, TuningDefaults.quickValidation().toleranceProfile());
        assertEquals(ValidationToleranceProfile.THOROUGH_DTYPE_AWARE, TuningDefaults.thoroughValidation().toleranceProfile());
        assertEquals(1e-5, TuningDefaults.quickValidation().absTolerance(tensor.DataType.FLOAT32));
        assertEquals(5e-7, TuningDefaults.thoroughValidation().absTolerance(tensor.DataType.FLOAT32));
        assertEquals(5e-3, TuningDefaults.quickValidation().absTolerance(tensor.DataType.BFLOAT16));
        assertEquals(1e-3, TuningDefaults.thoroughValidation().absTolerance(tensor.DataType.BFLOAT16));
    }

    @Test
    void workloadPresetFamilyChoosesThoroughForLossAutotune() {
        var workload = new tuning.workload.TensorRootWorkloadSpec(
                "defaults-loss",
                tuning.workload.WorkloadKind.LOSS,
                environment -> tensor.Tensor.scalar(1.0)
        );

        assertEquals(TuningPreset.THOROUGH, WorkloadPresetFamily.autotunePresetFor(workload));
        var request = TuningDefaults.recommendedAutotune(
                workload,
                profile("seed"),
                new ListCandidateSpace(List.of()),
                tuning.store.PersistencePolicy.disabled()
        );
        assertEquals(96, request.search().maxCandidates());
        assertTrue(request.validation().requireGradientMatch());
    }

    private static ExecutionProfile profile(String name) {
        return new ExecutionProfile(
                name,
                name,
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }
}
