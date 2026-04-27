import backend.runtime.ExecutionMode;
import backend.ComputeBackend;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import graph.optimizer.partition.PartitionTarget;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

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
        assertTrue(compiled.compileTrace().partitionPlanning() != null);
        assertTrue(runTrace.durationNs() >= 0L);
        assertTrue(runTrace.steps().size() > 0);
        assertEquals("FORWARD", runTrace.mode().name());
    }

    @Test
    void fusedHotPathPublishesPrepareAndRunTraceMetadata() {
        int size = 4096;
        float[] av = new float[size];
        float[] bv = new float[size];
        float[] cv = new float[size];
        for (int i = 0; i < size; i++) {
            av[i] = i * 0.01f;
            bv[i] = 1.0f + i * 0.02f;
            cv[i] = -0.5f + i * 0.03f;
        }
        Tensor a = new Tensor(av, new int[]{size}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(bv, new int[]{size}, null, "b", DataType.FLOAT32);
        Tensor c = new Tensor(cv, new int[]{size}, null, "c", DataType.FLOAT32);
        Tensor out = a.add(b).mul(c).relu().exp();

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        var prepared = compiled.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        var runTrace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertTrue(prepared.prepareTrace().measured());
        assertTrue(prepared.prepareTrace().durationNs() >= 0L);
        assertTrue(runTrace.durationNs() >= 0L);
        var fusedStep = runTrace.steps().stream()
                .filter(step -> step.metadata().fused() != null)
                .findFirst()
                .orElseThrow();
        assertTrue(fusedStep.metadata().fused().fusedNodeCount() > 1);
        assertTrue(!fusedStep.metadata().fused().executionBackend().isBlank());
        assertTrue(fusedStep.metadata().fused().schedulerSignature() != null
                && !fusedStep.metadata().fused().schedulerSignature().isBlank());
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

    @Test
    void applePartitionTraceCapturesLargestStructuralDagCandidateWhenLowererRejectsIt() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor abs = matmul.abs();
        Tensor out = relu.add(abs);

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(abs, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, OptimizerConfig.noOptimization());
        var decisions = compiled.compileTrace().partitionPlanning().decisions();

        assertTrue(decisions.stream().anyMatch(decision ->
                decision.structuralNodeIds().size() >= 4
                        && decision.opTypes().contains("RELU")
                        && decision.opTypes().contains("ABS")
                        && decision.opTypes().contains("ADD")
        ));
        assertTrue(decisions.stream().allMatch(decision -> decision.exploredCandidates() >= 0));
        assertTrue(decisions.stream().allMatch(decision -> !decision.searchBudgetHit()));
        assertTrue(decisions.stream().allMatch(decision -> decision.reason() != null && !decision.reason().isBlank()));
    }

    @Test
    void cpuOnlyGraphUsesCpuPartitionTargetInAutoMode() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).mul(a);

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());

        assertEquals(PartitionTarget.CPU, compiled.compileTrace().partitionPlanning().target());
        assertTrue(compiled.compileTrace().partitionPlanning().decisions().size() > 0);
    }

    @Test
    void autoPartitionTargetPrefersGpuOverCpuWhenGpuNodesExist() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());

        assertEquals(PartitionTarget.GPU_METAL, compiled.compileTrace().partitionPlanning().target());
    }
}
