import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import config.backend.CpuKernelConfig;
import graph.CompiledGraph;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class NonContiguousExecutionTest {
    private static final double EPS = 1e-9;
    private static final double EPS32 = 1e-6;

    @Test
    public void testAddNonContiguousStridedPath() {
        // size=6 < threshold(100): strided path should be selected
        RuntimeConfig runtimeConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100));

        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30, 40, 50, 60}, new int[]{2, 3}, null, "b_contig", DataType.FLOAT64);

        Tensor c = a.add(b);
        CompiledGraph.compile(c, CompileConfig.noGraphOptimizationBaseline()).prepare(runtimeConfig).execute(ExecutionMode.FORWARD);

        double[] expected = add(remapToContiguous(a), b.toDoubleArrayCopy());
        assertArrayEquals(expected, c.toDoubleArrayCopy(), EPS);
    }

    @Test
    public void testAddNonContiguousMaterializePath() {
        // threshold(0): materialize path should be selected
        RuntimeConfig runtimeConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0));

        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30, 40, 50, 60}, new int[]{2, 3}, null, "b_contig", DataType.FLOAT64);

        Tensor c = a.add(b);
        CompiledGraph.compile(c, CompileConfig.noGraphOptimizationBaseline()).prepare(runtimeConfig).execute(ExecutionMode.FORWARD);

        double[] expected = add(remapToContiguous(a), b.toDoubleArrayCopy());
        assertArrayEquals(expected, c.toDoubleArrayCopy(), EPS);
    }

    @Test
    public void testLogNonContiguousStridedVsMaterializeEquivalence() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig", DataType.FLOAT64);

        RuntimeConfig stridedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100));
        Tensor s = a.log();
        CompiledGraph.compile(s, CompileConfig.noGraphOptimizationBaseline()).prepare(stridedConfig).execute(ExecutionMode.FORWARD);
        double[] strided = s.toDoubleArrayCopy().clone();

        RuntimeConfig materializedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0));
        Tensor m = a.log();
        CompiledGraph.compile(m, CompileConfig.noGraphOptimizationBaseline()).prepare(materializedConfig).execute(ExecutionMode.FORWARD);
        double[] materialized = m.toDoubleArrayCopy().clone();

        assertArrayEquals(strided, materialized, EPS);
    }

    @Test
    public void testAddBroadcastWithNonContiguousInput() {
        RuntimeConfig runtimeConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100));

        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "b_broadcast", DataType.FLOAT64);

        Tensor c = a.add(b);
        CompiledGraph.compile(c, CompileConfig.noGraphOptimizationBaseline()).prepare(runtimeConfig).execute(ExecutionMode.FORWARD);

        Tensor ref = a.contiguous().add(b);
        CompiledGraph.compile(ref, CompileConfig.noGraphOptimizationBaseline()).prepare(runtimeConfig).execute(ExecutionMode.FORWARD);
        assertArrayEquals(ref.toDoubleArrayCopy(), c.toDoubleArrayCopy(), EPS);
    }

    @Test
    public void testCompareBroadcastWithNonContiguousInputStridedVsMaterializeEquivalence() {
        RuntimeConfig stridedConfig = runtimeConfig(new CpuKernelConfig(4, 1_000_000, 1_000_000, 1_000_000, 1_024, 100_000, 1_000_000));
        RuntimeConfig materializedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0));

        Tensor left = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "compare_left_noncontig", DataType.FLOAT64);
        Tensor right = new Tensor(new double[]{0, 3, 4}, new int[]{3}, null, "compare_right_broadcast", DataType.FLOAT64);

        Tensor strided = left.greaterThan(right);
        CompiledGraph.compile(strided, CompileConfig.noGraphOptimizationBaseline()).prepare(stridedConfig).execute(ExecutionMode.FORWARD);

        Tensor materialized = left.greaterThan(right);
        CompiledGraph.compile(materialized, CompileConfig.noGraphOptimizationBaseline()).prepare(materializedConfig).execute(ExecutionMode.FORWARD);

        assertArrayEquals(materialized.toBoolByteArrayCopy(), strided.toBoolByteArrayCopy());
    }

    @Test
    public void testLogicalBroadcastWithNonContiguousInputStridedVsMaterializeEquivalence() {
        RuntimeConfig stridedConfig = runtimeConfig(new CpuKernelConfig(4, 1_000_000, 1_000_000, 1_000_000, 1_024, 100_000, 1_000_000));
        RuntimeConfig materializedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0));

        Tensor left = new Tensor(new byte[]{1, 0, 1, 1, 0, 1}, new int[]{2, 3}, new int[]{1, 2}, null, "logical_left_noncontig", DataType.BOOL);
        Tensor right = new Tensor(new byte[]{1, 0, 1}, new int[]{3}, null, "logical_right_broadcast", DataType.BOOL);

        Tensor strided = left.logicalAnd(right);
        CompiledGraph.compile(strided, CompileConfig.noGraphOptimizationBaseline()).prepare(stridedConfig).execute(ExecutionMode.FORWARD);

        Tensor materialized = left.logicalAnd(right);
        CompiledGraph.compile(materialized, CompileConfig.noGraphOptimizationBaseline()).prepare(materializedConfig).execute(ExecutionMode.FORWARD);

        assertArrayEquals(materialized.toBoolByteArrayCopy(), strided.toBoolByteArrayCopy());
    }

    @Test
    public void testWhereBroadcastWithNonContiguousBranchStridedVsMaterializeEquivalence() {
        RuntimeConfig stridedConfig = runtimeConfig(new CpuKernelConfig(4, 1_000_000, 1_000_000, 1_000_000, 1_000_000, 100_000, 1_000_000));
        RuntimeConfig materializedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 0, 100_000, 0));

        Tensor condition = new Tensor(new byte[]{1, 0}, new int[]{2, 1}, null, "where_condition_broadcast", DataType.BOOL);
        Tensor ifTrue = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "where_true_noncontig", DataType.FLOAT64);
        Tensor ifFalse = new Tensor(new double[]{-1, -2, -3, -4, -5, -6}, new int[]{2, 3}, null, "where_false", DataType.FLOAT64);

        Tensor strided = Tensor.where(condition, ifTrue, ifFalse);
        CompiledGraph.compile(strided, CompileConfig.noGraphOptimizationBaseline()).prepare(stridedConfig).execute(ExecutionMode.FORWARD);

        Tensor materialized = Tensor.where(condition, ifTrue, ifFalse);
        CompiledGraph.compile(materialized, CompileConfig.noGraphOptimizationBaseline()).prepare(materializedConfig).execute(ExecutionMode.FORWARD);

        assertArrayEquals(materialized.toDoubleArrayCopy(), strided.toDoubleArrayCopy(), EPS);
    }

    @Test
    public void testAddBroadcastRowViewFloat32StridedVsMaterializeEquivalence() {
        RuntimeConfig stridedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100));
        RuntimeConfig materializedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0));

        Tensor left = new Tensor(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 4}, null, "left", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{10, 20}, new int[]{2, 1}, null, "bias", DataType.FLOAT32);

        Tensor strided = left.add(bias.expand(2, 4));
        CompiledGraph.compile(strided, CompileConfig.noGraphOptimizationBaseline()).prepare(stridedConfig).execute(ExecutionMode.FORWARD);

        Tensor materialized = left.add(bias.expand(2, 4));
        CompiledGraph.compile(materialized, CompileConfig.noGraphOptimizationBaseline()).prepare(materializedConfig).execute(ExecutionMode.FORWARD);

        assertArrayEquals(materialized.toDoubleArrayCopy(), strided.toDoubleArrayCopy(), EPS32);
    }

    @Test
    public void testMulNonContiguousFloat32StridedVsMaterializeEquivalence() {
        RuntimeConfig stridedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100));
        RuntimeConfig materializedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0));

        Tensor left = new Tensor(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 4}, new int[]{1, 2}, null, "left_noncontig", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{2, 3, 4, 5, 6, 7, 8, 9}, new int[]{2, 4}, null, "right_contig", DataType.FLOAT32);

        Tensor strided = left.mul(right);
        CompiledGraph.compile(strided, CompileConfig.noGraphOptimizationBaseline()).prepare(stridedConfig).execute(ExecutionMode.FORWARD);

        Tensor materialized = left.mul(right);
        CompiledGraph.compile(materialized, CompileConfig.noGraphOptimizationBaseline()).prepare(materializedConfig).execute(ExecutionMode.FORWARD);

        assertArrayEquals(materialized.toDoubleArrayCopy(), strided.toDoubleArrayCopy(), EPS32);
    }

    @Test
    public void testMulScalarExpandedRowViewFloat32StridedVsMaterializeEquivalence() {
        RuntimeConfig stridedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100));
        RuntimeConfig materializedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0));

        Tensor bias = new Tensor(new float[]{3, 7}, new int[]{2, 1}, null, "bias", DataType.FLOAT32);

        Tensor strided = bias.expand(2, 4).mul(0.5);
        CompiledGraph.compile(strided, CompileConfig.noGraphOptimizationBaseline()).prepare(stridedConfig).execute(ExecutionMode.FORWARD);

        Tensor materialized = bias.expand(2, 4).mul(0.5);
        CompiledGraph.compile(materialized, CompileConfig.noGraphOptimizationBaseline()).prepare(materializedConfig).execute(ExecutionMode.FORWARD);

        assertArrayEquals(materialized.toDoubleArrayCopy(), strided.toDoubleArrayCopy(), EPS32);
    }

    @Test
    public void testMulScalarExpandedRowViewBFloat16StridedVsMaterializeEquivalence() {
        RuntimeConfig stridedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100));
        RuntimeConfig materializedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0));

        Tensor bias = new Tensor(new float[]{3, 7}, new int[]{2, 1}, null, "bias_bf16", DataType.BFLOAT16);

        Tensor strided = bias.expand(2, 4).mul(0.5);
        CompiledGraph.compile(strided, CompileConfig.noGraphOptimizationBaseline()).prepare(stridedConfig).execute(ExecutionMode.FORWARD);

        Tensor materialized = bias.expand(2, 4).mul(0.5);
        CompiledGraph.compile(materialized, CompileConfig.noGraphOptimizationBaseline()).prepare(materializedConfig).execute(ExecutionMode.FORWARD);

        assertArrayEquals(materialized.toDoubleArrayCopy(), strided.toDoubleArrayCopy(), 2e-3);
    }

    @Test
    public void testMulSameShapeExpandedScalarBFloat16UsesBroadcastPath() {
        RuntimeConfig runtimeConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100));

        Tensor input = new Tensor(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 4}, null, "input_bf16", DataType.BFLOAT16);
        Tensor scale = Tensor.scalar(0.75d, DataType.BFLOAT16).expand(2, 4);

        Tensor out = input.mul(scale);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline()).prepare(runtimeConfig).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0}, out.toDoubleArrayCopy(), 2e-3);
    }

    private static RuntimeConfig runtimeConfig(CpuKernelConfig cpuKernelConfig) {
        return new RuntimeConfig(cpuKernelConfig, config.runtime.ApproximationConfig.defaults(), config.runtime.BlasConfig.disabled());
    }

    private static double[] add(double[] left, double[] right) {
        double[] out = new double[left.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = left[i] + right[i];
        }
        return out;
    }

    private static double[] remapToContiguous(Tensor src) {
        return src.toDoubleArrayCopy();
    }
}
