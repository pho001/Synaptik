import backend.ApproxMode;
import backend.ComputeEngine;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class ApproxModePolicyTest {
    @AfterEach
    void resetMode() {
        ComputeEngine.setApproxMode(ApproxMode.OFF);
    }

    @Test
    void expUsesApproxWhenModeAlways() {
        Tensor x = new Tensor(new double[]{-2.0, -1.0, 0.5, 2.0}, new int[]{4}, null, "x", DataType.FLOAT64);

        ComputeEngine.setApproxMode(ApproxMode.OFF);
        Tensor exact = x.exp();
        exact.compute();
        double[] yExact = exact.toDoubleArrayCopy();

        ComputeEngine.setApproxMode(ApproxMode.ALWAYS);
        Tensor approx = x.exp();
        approx.compute();
        double[] yApprox = approx.toDoubleArrayCopy();

        double delta = 0.0;
        for (int i = 0; i < yExact.length; i++) {
            delta = Math.max(delta, Math.abs(yExact[i] - yApprox[i]));
        }
        assertTrue(delta > 1e-4, "ALWAYS mode should alter exp outputs via fast approximation");
    }

    @Test
    void tanhUsesApproxWhenModeAlways() {
        Tensor x = new Tensor(new double[]{-3.0, -1.0, 0.5, 3.0}, new int[]{4}, null, "x", DataType.FLOAT64);

        ComputeEngine.setApproxMode(ApproxMode.OFF);
        Tensor exact = x.tanh();
        exact.compute();
        double[] yExact = exact.toDoubleArrayCopy();

        ComputeEngine.setApproxMode(ApproxMode.ALWAYS);
        Tensor approx = x.tanh();
        approx.compute();
        double[] yApprox = approx.toDoubleArrayCopy();

        double delta = 0.0;
        for (int i = 0; i < yExact.length; i++) {
            delta = Math.max(delta, Math.abs(yExact[i] - yApprox[i]));
        }
        assertTrue(delta > 1e-4, "ALWAYS mode should alter tanh outputs via fast approximation");
    }
}

