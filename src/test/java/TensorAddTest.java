import org.junit.jupiter.api.Test;

import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.*;

public class TensorAddTest {

    @Test
    public void testAdditionForward() {
        Tensor a = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "a");
        Tensor b = new Tensor(new double[]{3.0, 4.0}, new int[]{2}, null, "b");
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor c = a.add(b);
        CompiledGraph.compile(c, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{4.0, 6.0}, c.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    public void testAdditionTrainingComputeBuildsGradients() {
        Tensor a = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "a");
        Tensor b = new Tensor(new double[]{3.0, 4.0}, new int[]{2}, null, "b");
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor c = a.add(b);
        CompiledGraph.compile(c, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertNotNull(a.getGradient());
        assertNotNull(b.getGradient());
        assertArrayEquals(new double[]{1.0, 1.0}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1.0, 1.0}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }
}
