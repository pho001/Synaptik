import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class FastExpTest {

    @Test
    void fastExpForwardIsCloseToExp() {
        Tensor a = new Tensor(new double[]{-5.0, -1.0, 0.0, 1.0, 5.0}, new int[]{5}, null, "a", DataType.FLOAT64);
        Tensor y = a.fastExp();
        y.compute();

        double[] actual = y.toDoubleArrayCopy();
        double[] expected = new double[actual.length];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = Math.exp(a.toDoubleArrayCopy()[i]);
        }

        for (int i = 0; i < actual.length; i++) {
            double relErr = Math.abs(actual[i] - expected[i]) / expected[i];
            assertTrue(relErr <= 0.05, "relative error too high at index " + i + ": " + relErr);
        }
    }

    @Test
    void fastExpGradientUsesFastExpOutput() {
        Tensor a = new Tensor(new double[]{-2.0, 0.0, 2.0}, new int[]{3}, null, "a", DataType.FLOAT64);
        a.setRequiresGrad(true);
        Tensor y = a.fastExp();

        y.compute();
        y.getCompiledGraph().setTrainingModeOff();
        y.compute();
        double[] forward = y.toDoubleArrayCopy();

        y.getCompiledGraph().setTrainingModeOn();
        y.compute();
        double[] grad = a.getGradient().toDoubleArrayCopy();

        for (int i = 0; i < grad.length; i++) {
            double relErr = Math.abs(grad[i] - forward[i]) / Math.max(1e-12, Math.abs(forward[i]));
            assertTrue(relErr <= 1e-9, "gradient mismatch at index " + i + ": " + relErr);
        }
    }
}

