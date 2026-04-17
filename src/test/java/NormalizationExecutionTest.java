import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class NormalizationExecutionTest {

    @Test
    void batchNormUsingBatchStatisticsMatchesExpectedValues() {
        Tensor input = new Tensor(new double[]{
                1, 2,
                3, 4
        }, new int[]{2, 2, 1, 1}, null, "input", DataType.FLOAT64);
        Tensor gamma = new Tensor(new double[]{1, 1}, new int[]{2}, null, "gamma", DataType.FLOAT64);
        Tensor beta = new Tensor(new double[]{0, 0}, new int[]{2}, null, "beta", DataType.FLOAT64);

        Tensor out = input.batchNorm(gamma, beta, 1, 1e-12);
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 2, 1, 1}, out.getShape());
        assertArrayEquals(new double[]{
                -1, -1,
                1, 1
        }, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void batchNormUsingExternalStatisticsMatchesExpectedValues() {
        Tensor input = new Tensor(new double[]{5, 8}, new int[]{1, 2, 1, 1}, null, "input", DataType.FLOAT64);
        Tensor gamma = new Tensor(new double[]{2, 3}, new int[]{2}, null, "gamma", DataType.FLOAT64);
        Tensor beta = new Tensor(new double[]{10, 20}, new int[]{2}, null, "beta", DataType.FLOAT64);
        Tensor mean = new Tensor(new double[]{1, 2}, new int[]{2}, null, "mean", DataType.FLOAT64);
        Tensor variance = new Tensor(new double[]{4, 9}, new int[]{2}, null, "variance", DataType.FLOAT64);

        Tensor out = input.batchNorm(gamma, beta, mean, variance, 1, 1e-12);
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{14, 26}, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void layerNormForwardAndParameterGradientsMatchExpectedValues() {
        Tensor forwardInput = new Tensor(new double[]{
                1, 3,
                2, 4
        }, new int[]{2, 2}, null, "input", DataType.FLOAT64);
        Tensor forwardGamma = new Tensor(new double[]{2, 3}, new int[]{2}, null, "gamma", DataType.FLOAT64);
        Tensor forwardBeta = new Tensor(new double[]{10, 20}, new int[]{2}, null, "beta", DataType.FLOAT64);

        Tensor out = forwardInput.layerNorm(forwardGamma, forwardBeta, 1e-12);
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{8, 23, 8, 23}, out.toDoubleArrayCopy(), 1e-6);

        Tensor backwardInput = new Tensor(new double[]{
                1, 3,
                2, 4
        }, new int[]{2, 2}, null, "input", DataType.FLOAT64);
        Tensor gamma = new Tensor(new double[]{2, 3}, new int[]{2}, null, "gamma", DataType.FLOAT64);
        Tensor beta = new Tensor(new double[]{10, 20}, new int[]{2}, null, "beta", DataType.FLOAT64);
        gamma.setRequiresGrad(true);
        beta.setRequiresGrad(true);

        Tensor loss = backwardInput.layerNorm(gamma, beta, 1e-12).sum();
        CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{-2, 2}, gamma.getGradient().toDoubleArrayCopy(), 1e-6);
        assertArrayEquals(new double[]{2, 2}, beta.getGradient().toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void rmsNormMatchesExpectedValues() {
        Tensor input = new Tensor(new double[]{3, 4}, new int[]{1, 2}, null, "input", DataType.FLOAT64);
        Tensor gamma = new Tensor(new double[]{1, 1}, new int[]{2}, null, "gamma", DataType.FLOAT64);

        Tensor out = input.rmsNorm(gamma, 1e-12);
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                3.0 / Math.sqrt(12.5),
                4.0 / Math.sqrt(12.5)
        }, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void layerNormBackwardSupportsMultipleLeadingAxes() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
                17, 18, 19, 20,
                21, 22, 23, 24
        }, new int[]{2, 3, 4}, null, "input", DataType.FLOAT64);
        Tensor gamma = new Tensor(new double[]{1, 1, 1, 1}, new int[]{4}, null, "gamma", DataType.FLOAT64);
        Tensor beta = new Tensor(new double[]{0, 0, 0, 0}, new int[]{4}, null, "beta", DataType.FLOAT64);
        gamma.setRequiresGrad(true);
        beta.setRequiresGrad(true);

        Tensor loss = input.layerNorm(gamma, beta, 1e-12).sum();
        CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertEquals(4, gamma.getGradient().getFlatDataSize());
        assertEquals(4, beta.getGradient().getFlatDataSize());
    }

    @Test
    void rmsNormBackwardSupportsMultipleLeadingAxes() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
                17, 18, 19, 20,
                21, 22, 23, 24
        }, new int[]{2, 3, 4}, null, "input", DataType.FLOAT64);
        Tensor gamma = new Tensor(new double[]{1, 1, 1, 1}, new int[]{4}, null, "gamma", DataType.FLOAT64);
        gamma.setRequiresGrad(true);

        Tensor loss = input.rmsNorm(gamma, 1e-12).sum();
        CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertEquals(4, gamma.getGradient().getFlatDataSize());
    }

    @Test
    void normalizationContractsRejectInvalidShapes() {
        Tensor input = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "input", DataType.FLOAT64);
        Tensor gamma2 = new Tensor(new double[]{1, 1}, new int[]{2}, null, "gamma2", DataType.FLOAT64);
        Tensor beta3 = new Tensor(new double[]{0, 0, 0}, new int[]{3}, null, "beta3", DataType.FLOAT64);
        Tensor meanBad = new Tensor(new double[]{0}, new int[]{1}, null, "meanBad", DataType.FLOAT64);
        Tensor varBad = new Tensor(new double[]{1}, new int[]{1}, null, "varBad", DataType.FLOAT64);

        assertThrows(IllegalArgumentException.class, () -> input.layerNorm(gamma2, beta3, 1e-5));
        assertThrows(IllegalArgumentException.class, () -> input.batchNorm(gamma2, beta3, 1, 1e-5));
        assertThrows(IllegalArgumentException.class, () -> input.batchNorm(gamma2, gamma2, meanBad, varBad, 1, 1e-5));
    }
}
