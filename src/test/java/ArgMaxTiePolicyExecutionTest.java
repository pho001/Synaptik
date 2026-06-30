import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.reduction.ArgMaxTiePolicy;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class ArgMaxTiePolicyExecutionTest {
    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT64", "FLOAT32", "BFLOAT16"})
    void tiePoliciesExecuteForKeepDimsTrueAndFalse(DataType dataType) {
        Tensor input = input(dataType);

        Tensor firstKeepDims = input.argMax(1, true, ArgMaxTiePolicy.FIRST_INDEX);
        Tensor firstNoKeepDims = input.argMax(1, false, ArgMaxTiePolicy.FIRST_INDEX);
        Tensor lastKeepDims = input.argMax(1, true, ArgMaxTiePolicy.LAST_INDEX);
        Tensor lastNoKeepDims = input.argMax(1, false, ArgMaxTiePolicy.LAST_INDEX);

        execute(firstKeepDims);
        execute(firstNoKeepDims);
        execute(lastKeepDims);
        execute(lastNoKeepDims);

        assertArrayEquals(new int[]{2, 1}, firstKeepDims.getShape());
        assertArrayEquals(new int[]{2}, firstNoKeepDims.getShape());
        assertArrayEquals(new long[]{1, 0}, firstKeepDims.toInt64ArrayCopy());
        assertArrayEquals(new long[]{1, 0}, firstNoKeepDims.toInt64ArrayCopy());

        assertArrayEquals(new int[]{2, 1}, lastKeepDims.getShape());
        assertArrayEquals(new int[]{2}, lastNoKeepDims.getShape());
        assertArrayEquals(new long[]{2, 3}, lastKeepDims.toInt64ArrayCopy());
        assertArrayEquals(new long[]{2, 3}, lastNoKeepDims.toInt64ArrayCopy());
    }

    private static Tensor input(DataType dataType) {
        return new Tensor(new double[]{
                1, 7, 7, 2,
                3, 3, 1, 3
        }, new int[]{2, 4}, null, "input", dataType);
    }

    private static void execute(Tensor out) {
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults())
                .execute(ExecutionMode.FORWARD);
    }
}
