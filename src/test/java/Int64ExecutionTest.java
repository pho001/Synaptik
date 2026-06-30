import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class Int64ExecutionTest {
    private static final long LARGE = 9_007_199_254_740_993L;

    @Test
    void layoutKernelsPreserveLargeInt64ValuesExactly() {
        Tensor left = new Tensor(new long[]{LARGE, 2L}, new int[]{1, 2}, null, "left", DataType.INT64);
        Tensor right = new Tensor(new long[]{3L, LARGE + 2L}, new int[]{1, 2}, null, "right", DataType.INT64);
        Tensor out = Tensor.concat(0, left, right).tile(2, 1).pad(new int[]{1, 0}, new int[]{0, 1}, -5.0);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertEquals(DataType.INT64, out.getDataType());
        assertArrayEquals(new int[]{5, 3}, out.getShape());
        assertArrayEquals(new long[]{
                -5L, -5L, -5L,
                LARGE, 2L, -5L,
                3L, LARGE + 2L, -5L,
                LARGE, 2L, -5L,
                3L, LARGE + 2L, -5L
        }, out.toInt64ArrayCopy());
    }

    @Test
    void cumSumPreservesLargeInt64ValuesExactly() {
        Tensor input = new Tensor(new long[]{
                LARGE, 2L, 3L,
                10L, LARGE + 4L, 6L
        }, new int[]{2, 3}, null, "input", DataType.INT64);
        Tensor out = input.cumSum(1);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertEquals(DataType.INT64, out.getDataType());
        assertArrayEquals(new long[]{
                LARGE, LARGE + 2L, LARGE + 5L,
                10L, LARGE + 14L, LARGE + 20L
        }, out.toInt64ArrayCopy());
    }
}
