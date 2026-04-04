import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.AttentionOptions;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class AttentionExecutionTest {

    @Test
    void scaledDotProductAttentionWithoutMaskMatchesExpectedValues() {
        Tensor q = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "q", DataType.FLOAT64);
        Tensor k = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "k", DataType.FLOAT64);
        Tensor v = new Tensor(new double[]{
                10, 1,
                1, 10
        }, new int[]{1, 2, 2}, null, "v", DataType.FLOAT64);

        Tensor out = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(1.0));
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        double e = Math.exp(1.0);
        double denom = e + 1.0;
        assertArrayEquals(new int[]{1, 2, 2}, out.getShape());
        assertArrayEquals(new double[]{
                (10.0 * e + 1.0) / denom,
                (1.0 * e + 10.0) / denom,
                (10.0 + e) / denom,
                (1.0 + 10.0 * e) / denom
        }, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void scaledDotProductAttentionWithCausalMaskBlocksFutureTokens() {
        Tensor q = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "q", DataType.FLOAT64);
        Tensor k = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "k", DataType.FLOAT64);
        Tensor v = new Tensor(new double[]{
                10, 1,
                1, 10
        }, new int[]{1, 2, 2}, null, "v", DataType.FLOAT64);

        Tensor out = q.scaledDotProductAttention(k, v, AttentionOptions.causalDefaults().withScale(1.0));
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        double e = Math.exp(1.0);
        double denom = e + 1.0;
        assertArrayEquals(new double[]{
                10.0, 1.0,
                (10.0 + e) / denom,
                (1.0 + 10.0 * e) / denom
        }, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void scaledDotProductAttentionWithExternalMaskUsesBoolMask() {
        Tensor q = new Tensor(new double[]{1, 0}, new int[]{1, 1, 2}, null, "q", DataType.FLOAT64);
        Tensor k = new Tensor(new double[]{
                1, 0,
                0, 1
        }, new int[]{1, 2, 2}, null, "k", DataType.FLOAT64);
        Tensor v = new Tensor(new double[]{
                10, 1,
                1, 10
        }, new int[]{1, 2, 2}, null, "v", DataType.FLOAT64);
        Tensor mask = new Tensor(new byte[]{0, 1}, new int[]{1, 1, 2}, null, "mask", DataType.BOOL);

        Tensor out = q.scaledDotProductAttention(k, v, mask, AttentionOptions.defaults().withScale(1.0));
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{1.0, 10.0}, out.toDoubleArrayCopy(), 1e-6);
    }
}
