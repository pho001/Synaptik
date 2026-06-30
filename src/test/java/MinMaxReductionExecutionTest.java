import runtime.contract.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class MinMaxReductionExecutionTest {
    private static final double EPS64 = 1e-9;
    private static final double EPS32 = 1e-6;

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void minAxisKeepDimsPreservesReducedAxisAsSingleton(DataType dataType) {
        Tensor a = new Tensor(new double[]{
                2, 1, 1,
                4, 5, 4
        }, new int[]{2, 3}, null, "matrix", dataType);

        Tensor out = a.min(1, true);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 1}, out.getShape());
        assertArrayEquals(new double[]{1.0, 4.0}, out.toDoubleArrayCopy(), eps(dataType));
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void maxAllMatchesReference(DataType dataType) {
        Tensor a = new Tensor(new double[]{1, 6, 3, 6}, new int[]{4}, null, "vector", dataType);

        Tensor out = a.max();
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{1}, out.getShape());
        assertArrayEquals(new double[]{6.0}, out.toDoubleArrayCopy(), eps(dataType));
    }

    @Test
    void minBackwardSplitsTiesAlongAxis() {
        Tensor a = new Tensor(new double[]{
                1, 1, 2,
                3, 2, 2
        }, new int[]{2, 3}, null, "matrix", DataType.FLOAT64);
        a.setRequiresGrad(true);

        Tensor out = a.min(1, true);
        CompiledGraph.compile(out, CompileConfig.training()).prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new int[]{2, 3}, a.getGradient().getShape());
        assertArrayEquals(new double[]{
                0.5, 0.5, 0.0,
                0.0, 0.5, 0.5
        }, a.getGradient().toDoubleArrayCopy(), EPS64);
    }

    @Test
    void maxBackwardAllSplitsGlobalTies() {
        Tensor a = new Tensor(new double[]{1, 5, 5, 2}, new int[]{4}, null, "vector", DataType.FLOAT64);
        a.setRequiresGrad(true);

        Tensor out = a.max();
        CompiledGraph.compile(out, CompileConfig.training()).prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new int[]{4}, a.getGradient().getShape());
        assertArrayEquals(new double[]{0.0, 0.5, 0.5, 0.0}, a.getGradient().toDoubleArrayCopy(), EPS64);
    }

    @Test
    void minNonContiguousStridedVsMaterializedEquivalence() {
        Tensor a = new Tensor(new double[]{2, 1, 5, 4, 3, 0}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig", DataType.FLOAT64);

        RuntimeConfig stridedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100_000));
        Tensor strided = a.min(1);
        CompiledGraph.compile(strided, CompileConfig.noGraphOptimizationBaseline()).prepare(stridedConfig).execute(ExecutionMode.FORWARD);

        RuntimeConfig materializedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0));
        Tensor materialized = a.min(1);
        CompiledGraph.compile(materialized, CompileConfig.noGraphOptimizationBaseline()).prepare(materializedConfig).execute(ExecutionMode.FORWARD);

        assertArrayEquals(materialized.toDoubleArrayCopy(), strided.toDoubleArrayCopy(), EPS64);
    }

    private static RuntimeConfig runtimeConfig(CpuKernelConfig cpuKernelConfig) {
        return new RuntimeConfig(cpuKernelConfig, config.runtime.ApproximationConfig.defaults(), config.runtime.BlasConfig.disabled());
    }

    private static double eps(DataType dataType) {
        return dataType == DataType.FLOAT64 ? EPS64 : EPS32;
    }
}
