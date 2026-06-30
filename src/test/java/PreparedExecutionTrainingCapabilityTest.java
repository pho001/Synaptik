import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import runtime.execution.PreparedExecution;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PreparedExecutionTrainingCapabilityTest {
    @Test
    void inferenceOnlyPreparedExecutionRejectsTrainingMode() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{4.0, 5.0, 6.0}, new int[]{3}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).tanh();

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults());
        assertThrows(IllegalStateException.class, () -> execution.execute(ExecutionMode.FORWARD_BACKWARD));
    }

    @Test
    void trainingCapablePreparedExecutionComputesGradients() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{4.0, 5.0, 6.0}, new int[]{3}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.mul(b).add(a);
        RuntimeConfig runtimeConfig = RuntimeConfig.trainingDefaults();
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline()).prepare(runtimeConfig);

        assertTrue(execution.supportsBackward());
        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertNotNull(a.getGradient());
        assertNotNull(b.getGradient());
        assertArrayEquals(new double[]{5.0, 6.0, 7.0}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void compileAndPrepareDoNotPublishSemanticGradientsBeforeExecute() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{4.0, 5.0, 6.0}, new int[]{3}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.mul(b).add(a);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());

        assertNull(a.getGradient());
        assertNull(b.getGradient());
        assertNull(out.getGradient());

        PreparedExecution execution = compiled.prepare(RuntimeConfig.trainingDefaults());

        assertNull(a.getGradient());
        assertNull(b.getGradient());
        assertNull(out.getGradient());

        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertNotNull(a.getGradient());
        assertNotNull(b.getGradient());
    }

    @Test
    void repeatedRunsDoNotReusePreviouslyPublishedSemanticGradientObjects() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{4.0, 5.0, 6.0}, new int[]{3}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.mul(b).add(a);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.trainingDefaults());

        execution.execute(ExecutionMode.FORWARD_BACKWARD);
        Tensor firstGradient = a.getGradient();
        firstGradient.setDataAt(0, 999.0);

        execution.execute(ExecutionMode.FORWARD_BACKWARD);
        Tensor secondGradient = a.getGradient();

        assertNotNull(firstGradient);
        assertNotNull(secondGradient);
        assertTrue(firstGradient != secondGradient);
        assertArrayEquals(new double[]{5.0, 6.0, 7.0}, secondGradient.toDoubleArrayCopy(), 1e-9);
    }
}
