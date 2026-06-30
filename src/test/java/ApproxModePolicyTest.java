import backend.ApproxMode;
import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.profile.ExecutionProfile;
import config.runtime.ApproximationConfig;
import config.runtime.RuntimeConfig;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class ApproxModePolicyTest {
    @Test
    void expUsesApproxWhenModeAlways() {
        Tensor x = new Tensor(new double[]{-2.0, -1.0, 0.5, 2.0}, new int[]{4}, null, "x", DataType.FLOAT64);

        Tensor exact = x.exp();
        exact.compute(profile(exact, runtimeConfig(ApproxMode.OFF)));
        double[] yExact = exact.toDoubleArrayCopy();

        Tensor approx = x.exp();
        approx.compute(profile(approx, runtimeConfig(ApproxMode.ALWAYS)));
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

        Tensor exact = x.tanh();
        exact.compute(profile(exact, runtimeConfig(ApproxMode.OFF)));
        double[] yExact = exact.toDoubleArrayCopy();

        Tensor approx = x.tanh();
        approx.compute(profile(approx, runtimeConfig(ApproxMode.ALWAYS)));
        double[] yApprox = approx.toDoubleArrayCopy();

        double delta = 0.0;
        for (int i = 0; i < yExact.length; i++) {
            delta = Math.max(delta, Math.abs(yExact[i] - yApprox[i]));
        }
        assertTrue(delta > 1e-4, "ALWAYS mode should alter tanh outputs via fast approximation");
    }

    private static RuntimeConfig runtimeConfig(ApproxMode approxMode) {
        return new RuntimeConfig(
                config.backend.CpuKernelConfig.defaultsTraining(),
                new ApproximationConfig(approxMode, false),
                config.runtime.BlasConfig.disabled()
        );
    }

    private static ExecutionProfile profile(Tensor root, RuntimeConfig runtimeConfig) {
        return new ExecutionProfile(
                "test",
                "test",
                root.getDataType(),
                ExecutionMode.FORWARD,
                CompileConfig.noGraphOptimizationBaseline(),
                runtimeConfig
        );
    }
}
