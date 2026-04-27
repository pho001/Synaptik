import backend.runtime.ExecutionMode;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.autotune.GraphAutotuneMode;
import tuning.autotune.GraphAutotuneRequest;
import tuning.candidate.graph.GraphAutotuneCandidateSpace;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

public class AutotuneRequestBoundaryTest {
    @Test
    void graphAutotuneRequiresExplicitGraphAndRuntimeProfiles() {
        var seed = new config.profile.ExecutionProfile(
                "seed",
                "seed",
                DataType.FLOAT64,
                ExecutionMode.FORWARD_BACKWARD,
                config.optimizer.OptimizerConfig.trainingDefaults(),
                config.runtime.RuntimeConfig.trainingDefaults()
        );

        GraphAutotuneRequest request = new GraphAutotuneRequest(
                new TensorRootWorkloadSpec("boundary", WorkloadKind.GENERIC, environment -> tensor.Tensor.scalar(1.0)),
                "boundary",
                seed.dataType(),
                seed.mode(),
                GraphExecutionPolicy.fromExecutionProfile(seed),
                PlatformRuntimeProfile.fromExecutionProfile("platform", "hardware", "TEST", seed),
                GraphAutotuneMode.STANDARD,
                tuning.measure.MeasurementPolicy.defaults(),
                tuning.validate.ValidationPolicy.disabled(),
                new tuning.search.SearchPolicy(1, 1, 1, false),
                tuning.store.PersistencePolicy.disabled(),
                null
        );

        var autotuneRequest = request.toAutotuneRequest();

        assertEquals(GraphExecutionPolicy.fromExecutionProfile(seed), autotuneRequest.graphPolicy());
        assertInstanceOf(GraphAutotuneCandidateSpace.class, autotuneRequest.candidateSpace());
    }
}
