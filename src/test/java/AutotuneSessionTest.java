import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.candidate.Candidate;
import tuning.candidate.ListCandidateSpace;
import tuning.measure.MeasurementPolicy;
import tuning.autotune.AutotuneRequest;
import tuning.autotune.AutotuneSession;
import tuning.autotune.TuningResult;
import tuning.validate.TensorSnapshot;
import tuning.validate.ValidationPolicy;
import tuning.validate.ValidationReference;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadMetadata;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AutotuneSessionTest {
    @Test
    void autotuneSessionReturnsBestValidProfile() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "autotune_add",
                WorkloadKind.GENERIC,
                environment -> Tensor.scalar(2.0).add(Tensor.scalar(3.0)),
                environment -> ValidationReference.snapshot(
                        TensorSnapshot.capture("out", Tensor.scalar(5.0)),
                        Map.of(),
                        List.of()
                ),
                environment -> WorkloadMetadata.of("autotune_add", WorkloadKind.GENERIC)
        );

        Candidate valid = new Candidate("valid", profile("valid"));
        Candidate alsoValid = new Candidate("also_valid", profile("also_valid"));

        TuningResult result = AutotuneSession.create(new AutotuneRequest(
                workload,
                new ListCandidateSpace(List.of(valid, alsoValid)),
                new MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                ValidationPolicy.defaults(),
                new tuning.search.SearchPolicy(10, 2, 1, false),
                tuning.store.PersistencePolicy.disabled()
        )).run();

        assertNotNull(result.bestProfile());
        assertEquals(2, result.finalists().size());
        assertTrue(result.summary().contains("evaluated=2"));
    }

    @Test
    void autotuneSessionSkipsInvalidCandidates() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "autotune_invalid",
                WorkloadKind.GENERIC,
                environment -> Tensor.scalar(2.0).add(Tensor.scalar(3.0)),
                environment -> ValidationReference.snapshot(
                        TensorSnapshot.capture("out", Tensor.scalar(99.0)),
                        Map.of(),
                        List.of()
                ),
                environment -> WorkloadMetadata.of("autotune_invalid", WorkloadKind.GENERIC)
        );

        Candidate invalid = new Candidate("invalid", profile("invalid"));

        TuningResult result = AutotuneSession.create(new AutotuneRequest(
                workload,
                new ListCandidateSpace(List.of(invalid)),
                new MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                ValidationPolicy.defaults(),
                new tuning.search.SearchPolicy(10, 1, 1, false),
                tuning.store.PersistencePolicy.disabled()
        )).run();

        assertTrue(result.bestProfile() == null);
        assertEquals(0, result.finalists().size());
        assertTrue(result.summary().contains("evaluated=1"));
    }

    private static ExecutionProfile profile(String name) {
        return new ExecutionProfile(
                name,
                name,
                DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }
}
