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

    @Test
    void graphAutotuneNullSearchUsesModeAwareDefaults() {
        var seed = new config.profile.ExecutionProfile(
                "seed",
                "seed",
                DataType.FLOAT64,
                ExecutionMode.FORWARD_BACKWARD,
                config.optimizer.OptimizerConfig.trainingDefaults(),
                config.runtime.RuntimeConfig.trainingDefaults()
        );

        GraphAutotuneRequest standard = new GraphAutotuneRequest(
                new TensorRootWorkloadSpec("standard", WorkloadKind.GENERIC, environment -> tensor.Tensor.scalar(1.0)),
                "standard",
                seed.dataType(),
                seed.mode(),
                GraphExecutionPolicy.fromExecutionProfile(seed),
                PlatformRuntimeProfile.fromExecutionProfile("platform", "hardware", "TEST", seed),
                GraphAutotuneMode.STANDARD,
                tuning.measure.MeasurementPolicy.defaults(),
                tuning.validate.ValidationPolicy.disabled(),
                null,
                tuning.store.PersistencePolicy.disabled(),
                null
        );
        GraphAutotuneRequest research = new GraphAutotuneRequest(
                new TensorRootWorkloadSpec("research", WorkloadKind.GENERIC, environment -> tensor.Tensor.scalar(1.0)),
                "research",
                seed.dataType(),
                seed.mode(),
                GraphExecutionPolicy.fromExecutionProfile(seed),
                PlatformRuntimeProfile.fromExecutionProfile("platform", "hardware", "TEST", seed),
                GraphAutotuneMode.RESEARCH,
                tuning.measure.MeasurementPolicy.defaults(),
                tuning.validate.ValidationPolicy.disabled(),
                null,
                tuning.store.PersistencePolicy.disabled(),
                null
        );

        assertEquals(16, standard.search().maxCandidates());
        assertEquals(4, standard.search().beamWidth());
        assertEquals(1, standard.search().maxRounds());
        assertEquals(32, research.search().maxCandidates());
        assertEquals(4, research.search().beamWidth());
        assertEquals(4, research.search().maxRounds());
    }
}
