import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CompiledGraphTraceTest {
    @Test
    void compiledGraphExposesCompilePrepareAndRunTrace() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).mul(a);

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        var runTrace = compiled.executeTraced(config.runtime.RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertTrue(compiled.compileTrace().measured());
        assertTrue(compiled.compileTrace().totalNodeCount() > 0);
        assertTrue(runTrace.durationNs() >= 0L);
        assertTrue(runTrace.steps().size() > 0);
        assertEquals("FORWARD", runTrace.mode().name());
    }

    @Test
    void autotuneSessionCanRunWithoutCompiledGraphBackReference() {
        Tensor out = Tensor.scalar(2.0).add(Tensor.scalar(3.0));

        ExecutionProfile profile = new ExecutionProfile(
                "delegate",
                "delegate",
                DataType.FLOAT64,
                ExecutionMode.FORWARD,
                OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        tuning.session.TuningResult result = tuning.session.AutotuneSession.create(new tuning.session.AutotuneRequest(
                new tuning.workload.TensorRootWorkloadSpec(
                        "delegate_workload",
                        tuning.workload.WorkloadKind.GENERIC,
                        environment -> Tensor.scalar(2.0).add(Tensor.scalar(3.0)),
                        environment -> tuning.validate.ValidationReference.snapshot(
                                tuning.validate.TensorSnapshot.capture("out", Tensor.scalar(5.0)),
                                java.util.Map.of(),
                                java.util.List.of()
                        ),
                        environment -> tuning.workload.WorkloadMetadata.of("delegate_workload", tuning.workload.WorkloadKind.GENERIC)
                ),
                new tuning.candidate.ListCandidateSpace(java.util.List.of(new tuning.candidate.Candidate("delegate", profile))),
                new tuning.measure.MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.defaults(),
                new tuning.search.SearchPolicy(4, 1, 1, false),
                tuning.store.PersistencePolicy.disabled()
        )).run();

        assertTrue(result.bestProfile() != null);
    }
}
