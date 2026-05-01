import config.optimizer.OffloadConfig;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import operations.Operation;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class OptimizerFuseTest {

    @Test
    public void testFuseElementWiseOptimization() {
        // Baseline graph size without optimization.
        Tensor a0 = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "a0");
        Tensor b0 = new Tensor(new double[]{3.0, 4.0}, new int[]{2}, null, "b0");
        Tensor c0 = new Tensor(new double[]{5.0, 6.0}, new int[]{2}, null, "c0");
        a0.setRequiresGrad(true);
        b0.setRequiresGrad(true);
       c0.setRequiresGrad(true);
        Tensor e0 = a0.add(b0).add(c0);
        CompiledGraph baselineGraph = CompiledGraph.compile(e0, OptimizerConfig.noOptimization());
        PreparedExecution baselinePrepared = baselineGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        int baselineForwardSteps = baselinePrepared.forwardSteps().size();

        Tensor a = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "a");
        Tensor b = new Tensor(new double[]{3.0, 4.0}, new int[]{2}, null, "b");
        Tensor c = new Tensor(new double[]{5.0, 6.0}, new int[]{2}, null, "c");
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);
        c.setRequiresGrad(true);

        Tensor d = a.add(b);
        Tensor e = d.add(c);

        CompiledGraph compiledGraph = CompiledGraph.compile(e, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        prepared.execute(backend.runtime.ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{9.0, 12.0}, e.toDoubleArrayCopy(), 1e-9);

        assertNotNull(e.getOperation(), "Final tensor should have an operation");
        long fusedPreparedSteps = prepared.forwardSteps().stream()
                .filter(step -> step.metadata().fusedExecutable() != null)
                .count();
        assertTrue(fusedPreparedSteps > 0, "Expected prepared fused execution metadata");
        assertTrue(prepared.forwardSteps().size() < baselineForwardSteps,
                "Prepared execution should have fewer forward steps after region fusion");

        compiledGraph.execute(config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{1.0, 1.0}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1.0, 1.0}, b.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1.0, 1.0}, c.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    public void cpuFusedExecutionStillUsesOperationOpTypeFused() {
        Tensor a = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "cpuFusedA");
        Tensor b = new Tensor(new double[]{3.0, 4.0}, new int[]{2}, null, "cpuFusedB");
        Tensor out = a.add(b).relu().exp();

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        prepared.execute(backend.runtime.ExecutionMode.FORWARD);

        var fusedStep = prepared.forwardSteps().stream()
                .filter(step -> step.metadata().fusedExecutable() != null)
                .findFirst()
                .orElseThrow();

        assertEquals(Operation.OpType.FUSED, fusedStep.metadata().executionOperation().opType());
        assertEquals("CPU", fusedStep.metadata().backend().name());
        assertArrayEquals(
                new double[]{Math.exp(Math.max(0.0, 4.0)), Math.exp(Math.max(0.0, 6.0))},
                out.toDoubleArrayCopy(),
                1e-5
        );
    }

    @Test
    public void fusedViewChainUsesBackingTensorAsRuntimeInput() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base");
        Tensor out = base.select(0, 1).relu().exp();

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        prepared.execute(backend.runtime.ExecutionMode.FORWARD);

        long fusedPreparedSteps = prepared.forwardSteps().stream()
                .filter(step -> step.metadata().fusedExecutable() != null)
                .count();
        assertTrue(fusedPreparedSteps > 0, "Expected prepared fused execution metadata for view-fed chain");

        assertArrayEquals(
                new double[]{Math.exp(4.0), Math.exp(5.0), Math.exp(6.0)},
                out.toDoubleArrayCopy(),
                1e-5
        );
    }

    @Test
    public void gatherRemainsFusionBarrierAndIsNotAbsorbedAsAccessChain() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base");
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", tensor.DataType.INT32);
        Tensor out = base.gather(indices, 1).relu().exp();

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        prepared.execute(backend.runtime.ExecutionMode.FORWARD);

        var fusedStep = prepared.forwardSteps().stream()
                .filter(step -> step.metadata().fusedExecutable() != null)
                .findFirst()
                .orElseThrow();

        assertEquals(1, fusedStep.metadata().executionInputNodeIds().size());
        Tensor fusedInput = compiledGraph.getCompiledGraphAsList().get(fusedStep.metadata().executionInputNodeIds().getFirst());
        assertNotNull(fusedInput.getOperation());
        assertEquals(Operation.OpType.GATHER, fusedInput.getOperation().opType());
    }

    @Test
    public void takeAlongAxisRemainsFusionBarrier() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base");
        Tensor indices = new Tensor(new int[]{2, 1, 0, 0}, new int[]{2, 2}, null, "indices", tensor.DataType.INT32);
        Tensor out = base.takeAlongAxis(indices, 1).relu().exp();

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        prepared.execute(backend.runtime.ExecutionMode.FORWARD);

        var fusedStep = prepared.forwardSteps().stream()
                .filter(step -> step.metadata().fusedExecutable() != null)
                .findFirst()
                .orElseThrow();

        assertEquals(1, fusedStep.metadata().executionInputNodeIds().size());
        Tensor fusedInput = compiledGraph.getCompiledGraphAsList().get(fusedStep.metadata().executionInputNodeIds().getFirst());
        assertNotNull(fusedInput.getOperation());
        assertEquals(Operation.OpType.TAKE_ALONG_AXIS, fusedInput.getOperation().opType());
    }

    @Test
    public void scatterAddRemainsFusionBarrier() {
        Tensor base = new Tensor(new double[]{10, 20, 30, 40, 50, 60}, new int[]{2, 3}, null, "base");
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", tensor.DataType.INT32);
        Tensor src = new Tensor(new double[]{1, 5}, new int[]{2}, null, "src");
        Tensor out = base.scatterAdd(indices, src, 1).relu().exp();

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        prepared.execute(backend.runtime.ExecutionMode.FORWARD);

        var fusedStep = prepared.forwardSteps().stream()
                .filter(step -> step.metadata().fusedExecutable() != null)
                .findFirst()
                .orElseThrow();

        assertEquals(1, fusedStep.metadata().executionInputNodeIds().size());
        Tensor fusedInput = compiledGraph.getCompiledGraphAsList().get(fusedStep.metadata().executionInputNodeIds().getFirst());
        assertNotNull(fusedInput.getOperation());
        assertEquals(Operation.OpType.SCATTER_ADD, fusedInput.getOperation().opType());
    }

    @Test
    public void reductionRemainsFusionBarrier() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base");
        Tensor out = base.sum(1).relu().exp();

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        prepared.execute(backend.runtime.ExecutionMode.FORWARD);

        var fusedStep = prepared.forwardSteps().stream()
                .filter(step -> step.metadata().fusedExecutable() != null)
                .findFirst()
                .orElseThrow();

        assertEquals(1, fusedStep.metadata().executionInputNodeIds().size());
        Tensor fusedInput = compiledGraph.getCompiledGraphAsList().get(fusedStep.metadata().executionInputNodeIds().getFirst());
        assertNotNull(fusedInput.getOperation());
        assertEquals(Operation.OpType.SUM, fusedInput.getOperation().opType());
    }

    @Test
    public void matmulRemainsFusionBarrier() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "a");
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "b");
        Tensor out = a.matmul(b).relu().exp();

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        prepared.execute(backend.runtime.ExecutionMode.FORWARD);

        var fusedStep = prepared.forwardSteps().stream()
                .filter(step -> step.metadata().fusedExecutable() != null)
                .findFirst()
                .orElseThrow();

        assertEquals(1, fusedStep.metadata().executionInputNodeIds().size());
        Tensor fusedInput = compiledGraph.getCompiledGraphAsList().get(fusedStep.metadata().executionInputNodeIds().getFirst());
        assertNotNull(fusedInput.getOperation());
        assertEquals(Operation.OpType.MATMUL, fusedInput.getOperation().opType());
    }

    @Test
    public void cpuFusionRemainsAvailableWithAcceleratorOffloadPolicy() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "a");
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{4}, null, "b");
        Tensor out = a.add(b).relu().exp();

        OptimizerConfig optimizer = fuseOnlyInferenceConfig().withOffload(OffloadConfig.acceleratorScored());
        PreparedExecution prepared = CompiledGraph.compile(out, optimizer)
                .prepare(config.runtime.RuntimeConfig.inferenceDefaults()
                        .withAccelerator(config.runtime.AcceleratorConfig.disabled()));

        assertTrue(prepared.forwardSteps().stream()
                .anyMatch(step -> step.metadata().fusedExecutable() != null));
    }

    private static OptimizerConfig fuseOnlyInferenceConfig() {
        return OptimizerConfig.inferenceDefaults().withStageOrder(java.util.List.of(OptimizerStage.PART, OptimizerStage.FUSE, OptimizerStage.MEM));
    }
}
