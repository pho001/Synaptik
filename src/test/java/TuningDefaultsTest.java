import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.candidate.ListCandidateSpace;
import tuning.session.TuningDefaults;

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
                new ListCandidateSpace(List.of()),
                tuning.store.PersistencePolicy.disabled()
        );

        assertTrue(request.validation().requireGradientMatch());
        assertTrue(request.measurement().captureStepTrace());
        assertEquals(96, request.search().maxCandidates());
    }

    private static ExecutionProfile profile(String name) {
        return new ExecutionProfile(
                name,
                name,
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }
}
