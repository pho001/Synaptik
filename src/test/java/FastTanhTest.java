import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class FastTanhTest {

    @Test
    void fastTanhForwardIsCloseToTanh() {
        Tensor a = new Tensor(new double[]{-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0}, new int[]{7}, null, "a", DataType.FLOAT64);
        Tensor y = a.fastTanh();
        CompiledGraph.compile(y, CompileConfig.training())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        double[] x = a.toDoubleArrayCopy();
        double[] actual = y.toDoubleArrayCopy();
        for (int i = 0; i < actual.length; i++) {
            double expected = Math.tanh(x[i]);
            double abs = Math.abs(actual[i] - expected);
            assertTrue(abs <= 0.04, "abs error too high at index " + i + ": " + abs);
        }
    }

    @Test
    void fastTanhGradientUsesFastTanhOutput() {
        Tensor a = new Tensor(new double[]{-2.0, -0.5, 0.0, 0.5, 2.0}, new int[]{5}, null, "a", DataType.FLOAT64);
        a.setRequiresGrad(true);
        Tensor y = a.fastTanh();
        CompiledGraph graph = CompiledGraph.compile(y, CompileConfig.training());

        graph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        double[] forward = y.toDoubleArrayCopy();

        graph.execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        double[] grad = a.getGradient().toDoubleArrayCopy();

        for (int i = 0; i < grad.length; i++) {
            double expected = 1.0 - forward[i] * forward[i];
            double abs = Math.abs(grad[i] - expected);
            assertTrue(abs <= 1e-9, "gradient mismatch at index " + i + ": " + abs);
        }
    }
}
