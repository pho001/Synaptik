import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import graph.optimizer.GraphOptimizer;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PreparedExecutionTrainingCapabilityTest {
    @Test
    void inferenceOnlyPreparedExecutionRejectsTrainingMode() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{4.0, 5.0, 6.0}, new int[]{3}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).tanh();

        PreparedExecution execution = CompiledGraph.compile(out, new GraphOptimizer()).prepare(RuntimeConfig.inferenceDefaults());
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
        PreparedExecution execution = CompiledGraph.compile(out, new GraphOptimizer()).prepare(runtimeConfig);

        assertTrue(execution.supportsBackward());
        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertNotNull(a.getGradient());
        assertNotNull(b.getGradient());
        assertArrayEquals(new double[]{5.0, 6.0, 7.0}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }
}
