import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import graph.execution.PublicationPolicy;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import training.optimizer.SgdOptimizer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class PublicationPolicyTest {
    @Test
    void outputOnlyPublishesRootValueAndClearsGradients() {
        Tensor x = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        x.setRequiresGrad(true);
        Tensor loss = x.mul(3.0).sum();
        PreparedExecution prepared = CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults());

        prepared.execute(ExecutionMode.FORWARD_BACKWARD, PublicationPolicy.OUTPUT_ONLY);

        assertArrayEquals(new double[]{9.0}, loss.toDoubleArrayCopy(), 1.0e-9);
        assertNull(x.getGradient());
    }

    @Test
    void noneDoesNotPublishForwardOutput() {
        Tensor x = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor out = x.mul(5.0);
        PreparedExecution prepared = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults());

        prepared.execute(ExecutionMode.FORWARD, PublicationPolicy.NONE);

        assertArrayEquals(new double[]{0.0, 0.0}, out.toDoubleArrayCopy(), 1.0e-9);
    }

    @Test
    void allPublishesIntermediateForwardValues() {
        Tensor x = new Tensor(new double[]{2.0}, new int[]{1}, null, "x", DataType.FLOAT64);
        Tensor intermediate = x.mul(3.0);
        Tensor out = intermediate.mul(2.0);
        PreparedExecution prepared = CompiledGraph.compile(out, CompileConfig.cpuOnlyBaseline())
                .prepare(RuntimeConfig.inferenceDefaults());

        prepared.execute(ExecutionMode.FORWARD, PublicationPolicy.ALL);

        assertArrayEquals(new double[]{6.0}, intermediate.toDoubleArrayCopy(), 1.0e-9);
        assertArrayEquals(new double[]{12.0}, out.toDoubleArrayCopy(), 1.0e-9);
    }

    @Test
    void optimizerStepCanPublishGradientsWhenRequested() {
        Tensor w = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "w", DataType.FLOAT32);
        Tensor x = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "x", DataType.FLOAT32);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        PreparedExecution prepared = CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults());

        prepared.executeOptimizerStep(new SgdOptimizer(0.1f), PublicationPolicy.OUTPUT_AND_GRADIENTS);

        assertArrayEquals(new float[]{0.8f, 1.7f}, w.getFloat32Data(), 1.0e-6f);
        assertArrayEquals(new float[]{2.0f, 3.0f}, w.getGradient().getFloat32Data(), 1.0e-6f);
        assertArrayEquals(new float[]{1.0f, 2.0f}, x.getGradient().getFloat32Data(), 1.0e-6f);
    }
}
