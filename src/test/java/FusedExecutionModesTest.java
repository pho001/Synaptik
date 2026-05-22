import backend.runtime.ExecutionMode;
import backend.cpu.fused.plan.FusedOperation;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.runtime.RuntimeConfig;
import config.backend.CpuKernelConfig;
import graph.CompiledGraph;
import backend.cpu.fused.runtime.FusedDTypeOps;
import backend.cpu.fused.asm.FusedAsmSpecializationMatcher;
import backend.cpu.fused.asm.emit.FusedOperationGenerator;
import graph.execution.PreparedExecutionStep;
import graph.execution.PreparedExecution;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class FusedExecutionModesTest {
    private static final double EPS = 1e-9;

    @Test
    void fusedGraphMatchesBaselineAcrossExecutionModes() {
        double[] aVals = buildInput(8192, 0.11);
        double[] bVals = buildInput(8192, -0.07);
        double[] cVals = buildInput(8192, 0.03);

        Tensor aBase = new Tensor(aVals.clone(), new int[]{aVals.length}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(bVals.clone(), new int[]{bVals.length}, null, "bBase", DataType.FLOAT64);
        Tensor cBase = new Tensor(cVals.clone(), new int[]{cVals.length}, null, "cBase", DataType.FLOAT64);
        Tensor baseline = aBase.add(bBase).mul(cBase).add(aBase.mul(0.25)).max(bBase).min(cBase).sigmoid();
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        // SCALAR
        assertModeMatches(expected, aVals, bVals, cVals,
                new CpuKernelConfig(4, 32, 32, 32, Integer.MAX_VALUE, Integer.MAX_VALUE));
        // VECTOR
        assertModeMatches(expected, aVals, bVals, cVals,
                new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE));
        // PARALLEL
        assertModeMatches(expected, aVals, bVals, cVals,
                new CpuKernelConfig(4, 32, 32, 32, Integer.MAX_VALUE, 1));
        // PARALLEL_VECTOR
        assertModeMatches(expected, aVals, bVals, cVals,
                new CpuKernelConfig(4, 32, 32, 32, 1, 1));
    }

    @Test
    void fusedGraphRespectsFloat32AndBFloat16Modes() {
        int size = 4096;
        double[] aVals = buildInput(size, 0.06);
        double[] bVals = buildInput(size, -0.02);
        double[] cVals = buildInput(size, 0.01);

        double[] outF32 = runTypedFused(aVals, bVals, cVals, DataType.FLOAT32);
        double[] outBF16 = runTypedFused(aVals, bVals, cVals, DataType.BFLOAT16);

        double[] expectedF32 = expectedTyped(aVals, bVals, cVals, FusedDTypeOps.MODE_F32);
        double[] expectedBF16 = expectedTyped(aVals, bVals, cVals, FusedDTypeOps.MODE_BF16);

        assertArrayEquals(expectedF32, outF32, 1e-6);
        assertArrayEquals(expectedBF16, outBF16, 6e-3);
    }

    @Test
    void fusedGraphMatchesBaselineWithBroadcastInputs() {
        Tensor aBase = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8
        }, new int[]{2, 1, 4}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(new double[]{
                10, 20, 30, 40,
                50, 60, 70, 80,
                90, 100, 110, 120
        }, new int[]{1, 3, 4}, null, "bBase", DataType.FLOAT64);
        Tensor cBase = new Tensor(new double[]{
                0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.1, 1.2,
                1.3, 1.4, 1.5, 1.6,
                1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4,
                2.5, 2.6, 2.7, 2.8
        }, new int[]{2, 3, 4}, null, "cBase", DataType.FLOAT64);

        Tensor baseline = aBase.add(bBase).mul(cBase).add(aBase).sigmoid();
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        assertBroadcastModeMatches(expected,
                new CpuKernelConfig(4, 32, 32, 32, Integer.MAX_VALUE, Integer.MAX_VALUE));
        assertBroadcastModeMatches(expected,
                new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE));
        assertBroadcastModeMatches(expected,
                new CpuKernelConfig(4, 32, 32, 32, Integer.MAX_VALUE, 1));
        assertBroadcastModeMatches(expected,
                new CpuKernelConfig(4, 32, 32, 32, 1, 1));
    }

    @Test
    void fusedBFloat16MatchesBaselineWithBroadcastInputs() {
        Tensor aBase = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8
        }, new int[]{2, 1, 4}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(new double[]{
                10, 20, 30, 40,
                50, 60, 70, 80,
                90, 100, 110, 120
        }, new int[]{1, 3, 4}, null, "bBase", DataType.FLOAT64);
        Tensor cBase = new Tensor(new double[]{
                0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.1, 1.2,
                1.3, 1.4, 1.5, 1.6,
                1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4,
                2.5, 2.6, 2.7, 2.8
        }, new int[]{2, 3, 4}, null, "cBase", DataType.FLOAT64);

        Tensor baseline = aBase.add(bBase).mul(cBase).add(aBase).sigmoid();
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aBase.toDoubleArrayCopy(), new int[]{2, 1, 4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bBase.toDoubleArrayCopy(), new int[]{1, 3, 4}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(cBase.toDoubleArrayCopy(), new int[]{2, 3, 4}, null, "c", DataType.FLOAT64);
        a.setDataType(DataType.BFLOAT16);
        b.setDataType(DataType.BFLOAT16);
        c.setDataType(DataType.BFLOAT16);

        Tensor out = a.add(b).mul(c).add(a).sigmoid();
        CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 2e-2);
    }

    @Test
    void fusedBFloat16MatchesBaselineWithNonContiguousInputs() {
        Tensor aBase = new Tensor(
                new double[]{1, 2, 3, 4, 5, 6},
                new int[]{2, 3},
                new int[]{1, 2},
                null,
                "aBase",
                DataType.FLOAT64
        );
        Tensor bBase = new Tensor(
                new double[]{6, 5, 4, 3, 2, 1},
                new int[]{2, 3},
                new int[]{1, 2},
                null,
                "bBase",
                DataType.FLOAT64
        );
        Tensor baseline = aBase.add(bBase).mul(aBase).sigmoid();
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(
                new double[]{1, 2, 3, 4, 5, 6},
                new int[]{2, 3},
                new int[]{1, 2},
                null,
                "a",
                DataType.FLOAT64
        );
        Tensor b = new Tensor(
                new double[]{6, 5, 4, 3, 2, 1},
                new int[]{2, 3},
                new int[]{1, 2},
                null,
                "b",
                DataType.FLOAT64
        );
        a.setDataType(DataType.BFLOAT16);
        b.setDataType(DataType.BFLOAT16);

        Tensor out = a.add(b).mul(a).sigmoid();
        CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 2e-2);
    }

    @Test
    void fusedGraphSupportsWhere() {
        Tensor cond = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{4}, null, "cond", DataType.BOOL);
        Tensor aBase = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(new double[]{10, 20, 30, 40}, new int[]{4}, null, "bBase", DataType.FLOAT64);

        Tensor baseline = Tensor.where(cond, aBase, bBase).relu().mul(aBase);
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor condFused = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{4}, null, "condFused", DataType.BOOL);
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30, 40}, new int[]{4}, null, "b", DataType.FLOAT64);

        Tensor out = Tensor.where(condFused, a, b).relu().mul(a);
        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(runtimeConfig(CpuKernelConfig.defaultsTraining()));
        prepared.execute(ExecutionMode.FORWARD);
        assertArrayEquals(expected, out.toDoubleArrayCopy(), EPS);
    }

    @Test
    void fusedGraphExecutesCanonicalizedConstantNodes() {
        assertConstantNodeFusion(DataType.FLOAT64, FusedDTypeOps.MODE_F64, 1e-9);
    }

    @Test
    void fusedGraphSpecializesMaskedScaleWhereAsmVectorPath() {
        int size = 4096;
        byte[] maskValues = new byte[size];
        float[] valueValues = new float[size];
        for (int i = 0; i < size; i++) {
            maskValues[i] = (byte) ((i & 3) == 0 ? 1 : 0);
            valueValues[i] = (float) (Math.sin(i * 0.03125) + (i % 11) * 0.125);
        }

        Tensor maskBase = new Tensor(maskValues.clone(), new int[]{size}, null, "maskBase", DataType.BOOL);
        Tensor fillBase = new Tensor(new float[]{-1000.0f}, new int[]{1}, null, "fillBase", DataType.FLOAT32);
        Tensor valuesBase = new Tensor(valueValues.clone(), new int[]{size}, null, "valuesBase", DataType.FLOAT32);
        Tensor baseline = Tensor.where(maskBase, valuesBase.mul(0.25), fillBase);
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, 1))).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor mask = new Tensor(maskValues.clone(), new int[]{size}, null, "mask", DataType.BOOL);
        Tensor fill = new Tensor(new float[]{-1000.0f}, new int[]{1}, null, "fill", DataType.FLOAT32);
        Tensor values = new Tensor(valueValues.clone(), new int[]{size}, null, "values", DataType.FLOAT32);
        Tensor out = Tensor.where(mask, values.mul(0.25), fill);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, 1)));
        prepared.execute(ExecutionMode.FORWARD);
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void fusedTensorPowExecutesThroughScalarPathWhenVectorRequested() {
        Tensor baseBaseline = new Tensor(new float[]{2f, 4f, 9f, 16f}, new int[]{4}, null, "baseBaseline", DataType.FLOAT32);
        Tensor exponentBaseline = new Tensor(new float[]{3f, 0.5f, 2f, -1f}, new int[]{4}, null, "exponentBaseline", DataType.FLOAT32);
        Tensor baseline = baseBaseline.pow(exponentBaseline).add(baseBaseline);
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, 1))).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor base = new Tensor(new float[]{2f, 4f, 9f, 16f}, new int[]{4}, null, "base", DataType.FLOAT32);
        Tensor exponent = new Tensor(new float[]{3f, 0.5f, 2f, -1f}, new int[]{4}, null, "exponent", DataType.FLOAT32);
        Tensor out = base.pow(exponent).add(base);
        PreparedExecution prepared = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, 1)));

        assertHasPreparedFusedStep(prepared);
        prepared.execute(ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void fusedGraphSupportsCompareAndLogicalBoolOutput() {
        Tensor aBase = new Tensor(new double[]{1, 5, 3, 8}, new int[]{4}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(new double[]{2, 4, 3, 1}, new int[]{4}, null, "bBase", DataType.FLOAT64);
        Tensor cBase = new Tensor(new double[]{0, 6, 2, 9}, new int[]{4}, null, "cBase", DataType.FLOAT64);
        Tensor dBase = new Tensor(new double[]{1, 7, 3, 2}, new int[]{4}, null, "dBase", DataType.FLOAT64);

        Tensor baseline = aBase.greaterThan(bBase).logicalOr(cBase.lessThan(dBase));
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        boolean[] expected = baseline.toBooleanArrayCopy().clone();

        Tensor a = new Tensor(new double[]{1, 5, 3, 8}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2, 4, 3, 1}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(new double[]{0, 6, 2, 9}, new int[]{4}, null, "c", DataType.FLOAT64);
        Tensor d = new Tensor(new double[]{1, 7, 3, 2}, new int[]{4}, null, "d", DataType.FLOAT64);

        Tensor out = a.greaterThan(b).logicalOr(c.lessThan(d));
        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(runtimeConfig(CpuKernelConfig.defaultsTraining()));
        prepared.execute(ExecutionMode.FORWARD);
        assertArrayEquals(expected, out.toBooleanArrayCopy());
    }

    @Test
    void fusedGraphSupportsInternalBoolConditionForWhere() {
        Tensor aBase = new Tensor(new double[]{1, 5, 3, 8}, new int[]{4}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(new double[]{2, 4, 3, 1}, new int[]{4}, null, "bBase", DataType.FLOAT64);
        Tensor xBase = new Tensor(new double[]{10, 20, 30, 40}, new int[]{4}, null, "xBase", DataType.FLOAT64);
        Tensor yBase = new Tensor(new double[]{100, 200, 300, 400}, new int[]{4}, null, "yBase", DataType.FLOAT64);

        Tensor baseline = Tensor.where(aBase.greaterThan(bBase), xBase, yBase).relu().mul(xBase);
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(new double[]{1, 5, 3, 8}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2, 4, 3, 1}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor x = new Tensor(new double[]{10, 20, 30, 40}, new int[]{4}, null, "x", DataType.FLOAT64);
        Tensor y = new Tensor(new double[]{100, 200, 300, 400}, new int[]{4}, null, "y", DataType.FLOAT64);

        Tensor out = Tensor.where(a.greaterThan(b), x, y).relu().mul(x);
        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(runtimeConfig(CpuKernelConfig.defaultsTraining()));
        prepared.execute(ExecutionMode.FORWARD);
        assertArrayEquals(expected, out.toDoubleArrayCopy(), EPS);
    }

    @Test
    void fusedGraphSupportsInternalBoolConditionForWhereInVectorMode() {
        int size = 4096;
        double[] aVals = buildInput(size, 0.07);
        double[] bVals = buildInput(size, -0.04);
        double[] xVals = buildInput(size, 0.03);
        double[] yVals = buildInput(size, -0.02);

        Tensor aBase = new Tensor(aVals.clone(), new int[]{size}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(bVals.clone(), new int[]{size}, null, "bBase", DataType.FLOAT64);
        Tensor xBase = new Tensor(xVals.clone(), new int[]{size}, null, "xBase", DataType.FLOAT64);
        Tensor yBase = new Tensor(yVals.clone(), new int[]{size}, null, "yBase", DataType.FLOAT64);
        Tensor baseline = Tensor.where(aBase.greaterThan(bBase), xBase, yBase).relu().mul(xBase);
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aVals.clone(), new int[]{size}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{size}, null, "b", DataType.FLOAT64);
        Tensor x = new Tensor(xVals.clone(), new int[]{size}, null, "x", DataType.FLOAT64);
        Tensor y = new Tensor(yVals.clone(), new int[]{size}, null, "y", DataType.FLOAT64);
        Tensor out = Tensor.where(a.greaterThan(b), x, y).relu().mul(x);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)));
        prepared.execute(ExecutionMode.FORWARD);
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void fusedGraphSupportsCompareAndLogicalIntermediatesInVectorMode() {
        int size = 4096;
        double[] aVals = buildInput(size, 0.09);
        double[] bVals = buildInput(size, -0.03);
        double[] cVals = buildInput(size, 0.02);
        double[] dVals = buildInput(size, -0.01);
        double[] xVals = buildInput(size, 0.05);
        double[] yVals = buildInput(size, -0.04);

        Tensor aBase = new Tensor(aVals.clone(), new int[]{size}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(bVals.clone(), new int[]{size}, null, "bBase", DataType.FLOAT64);
        Tensor cBase = new Tensor(cVals.clone(), new int[]{size}, null, "cBase", DataType.FLOAT64);
        Tensor dBase = new Tensor(dVals.clone(), new int[]{size}, null, "dBase", DataType.FLOAT64);
        Tensor xBase = new Tensor(xVals.clone(), new int[]{size}, null, "xBase", DataType.FLOAT64);
        Tensor yBase = new Tensor(yVals.clone(), new int[]{size}, null, "yBase", DataType.FLOAT64);

        Tensor baseline = Tensor.where(aBase.greaterThan(bBase).logicalOr(cBase.lessThan(dBase)), xBase, yBase).mul(xBase);
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aVals.clone(), new int[]{size}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{size}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(cVals.clone(), new int[]{size}, null, "c", DataType.FLOAT64);
        Tensor d = new Tensor(dVals.clone(), new int[]{size}, null, "d", DataType.FLOAT64);
        Tensor x = new Tensor(xVals.clone(), new int[]{size}, null, "x", DataType.FLOAT64);
        Tensor y = new Tensor(yVals.clone(), new int[]{size}, null, "y", DataType.FLOAT64);

        Tensor out = Tensor.where(a.greaterThan(b).logicalOr(c.lessThan(d)), x, y).mul(x);
        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)));
        prepared.execute(ExecutionMode.FORWARD);
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void fusedGraphSupportsExternalBoolInputInVectorMode() {
        int size = 4096;
        byte[] condVals = new byte[size];
        double[] xVals = buildInput(size, 0.05);
        double[] yVals = buildInput(size, -0.04);
        double[] zVals = buildInput(size, 0.02);
        for (int i = 0; i < size; i++) {
            condVals[i] = (i % 3) == 0 ? (byte) 1 : (byte) 0;
        }

        Tensor condBase = new Tensor(condVals.clone(), new int[]{size}, null, "condBase", DataType.BOOL);
        Tensor xBase = new Tensor(xVals.clone(), new int[]{size}, null, "xBase", DataType.FLOAT64);
        Tensor yBase = new Tensor(yVals.clone(), new int[]{size}, null, "yBase", DataType.FLOAT64);
        Tensor zBase = new Tensor(zVals.clone(), new int[]{size}, null, "zBase", DataType.FLOAT64);
        Tensor baseline = Tensor.where(condBase, xBase, yBase).relu().mul(zBase);
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor cond = new Tensor(condVals.clone(), new int[]{size}, null, "cond", DataType.BOOL);
        Tensor x = new Tensor(xVals.clone(), new int[]{size}, null, "x", DataType.FLOAT64);
        Tensor y = new Tensor(yVals.clone(), new int[]{size}, null, "y", DataType.FLOAT64);
        Tensor z = new Tensor(zVals.clone(), new int[]{size}, null, "z", DataType.FLOAT64);
        Tensor out = Tensor.where(cond, x, y).relu().mul(z);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)));
        prepared.execute(ExecutionMode.FORWARD);
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void fusedBFloat16SupportsCompareLogicalWhereGraph() {
        int size = 4096;
        double[] aVals = buildInput(size, 0.08);
        double[] bVals = buildInput(size, -0.03);
        double[] xVals = buildInput(size, 0.05);
        double[] yVals = buildInput(size, -0.02);

        Tensor aBase = new Tensor(aVals.clone(), new int[]{size}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(bVals.clone(), new int[]{size}, null, "bBase", DataType.FLOAT64);
        Tensor xBase = new Tensor(xVals.clone(), new int[]{size}, null, "xBase", DataType.FLOAT64);
        Tensor yBase = new Tensor(yVals.clone(), new int[]{size}, null, "yBase", DataType.FLOAT64);
        Tensor baseline = Tensor.where(aBase.greaterThan(bBase).logicalOr(aBase.lessThan(bBase)), xBase, yBase).sigmoid();
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aVals.clone(), new int[]{size}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{size}, null, "b", DataType.FLOAT64);
        Tensor x = new Tensor(xVals.clone(), new int[]{size}, null, "x", DataType.FLOAT64);
        Tensor y = new Tensor(yVals.clone(), new int[]{size}, null, "y", DataType.FLOAT64);
        a.setDataType(DataType.BFLOAT16);
        b.setDataType(DataType.BFLOAT16);
        x.setDataType(DataType.BFLOAT16);
        y.setDataType(DataType.BFLOAT16);

        Tensor out = Tensor.where(a.greaterThan(b).logicalOr(a.lessThan(b)), x, y).sigmoid();
        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)));
        prepared.execute(ExecutionMode.FORWARD);
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 2e-2);
    }

    @Test
    void fusedBFloat16SupportsAbsAndClampGraph() {
        int size = 4096;
        double[] aVals = buildInput(size, 0.06);
        double[] bVals = buildInput(size, -0.05);

        Tensor aBase = new Tensor(aVals.clone(), new int[]{size}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(bVals.clone(), new int[]{size}, null, "bBase", DataType.FLOAT64);
        Tensor baseline = aBase.sub(bBase).abs().clampMin(-0.25).clampMax(0.75).sigmoid();
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aVals.clone(), new int[]{size}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{size}, null, "b", DataType.FLOAT64);
        a.setDataType(DataType.BFLOAT16);
        b.setDataType(DataType.BFLOAT16);

        Tensor out = a.sub(b).abs().clampMin(-0.25).clampMax(0.75).sigmoid();
        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)));
        prepared.execute(ExecutionMode.FORWARD);
        assertHasPreparedFusedStep(prepared);
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 2e-2);
    }

    @Test
    void fusedBFloat16SupportsVectorizedNumericFastPath() {
        int size = 4096;
        double[] aVals = buildInput(size, 0.06);
        double[] bVals = buildInput(size, -0.05);

        Tensor aBase = new Tensor(aVals.clone(), new int[]{size}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(bVals.clone(), new int[]{size}, null, "bBase", DataType.FLOAT64);
        Tensor baseline = aBase.sub(bBase).abs().add(bBase.abs()).mul(0.5).clampMin(0.01).sqrt().clampMax(1.25);
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aVals.clone(), new int[]{size}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{size}, null, "b", DataType.FLOAT64);
        a.setDataType(DataType.BFLOAT16);
        b.setDataType(DataType.BFLOAT16);

        Tensor out = a.sub(b).abs().add(b.abs()).mul(0.5).clampMin(0.01).sqrt().clampMax(1.25);
        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)));
        prepared.execute(ExecutionMode.FORWARD);
        assertHasPreparedFusedStep(prepared);
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 2e-2);
    }

    @Test
    void fusedBFloat16GeneratedKernelContainsRealVectorOps() {
        int size = 4096;
        double[] aVals = buildInput(size, 0.06);
        double[] bVals = buildInput(size, -0.05);

        Tensor a = new Tensor(aVals.clone(), new int[]{size}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{size}, null, "b", DataType.FLOAT64);
        a.setDataType(DataType.BFLOAT16);
        b.setDataType(DataType.BFLOAT16);

        Tensor out = a.sub(b).abs().add(b.abs()).mul(0.5).clampMin(0.01).sqrt().clampMax(1.25);
        PreparedExecution prepared = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)));

        var fusedStep = findPreparedFusedStep(prepared);

        FusedOperation fused = (FusedOperation) fusedStep.executionOperation();
        byte[] bytecode = FusedOperationGenerator.generate(
                "debug/test/Bf16VectorKernel",
                fused.getPlan(),
                fused.getNumericContract(),
                fused.getApproximationContract(),
                testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).dispatchHints().vectorWidth(),
                FusedAsmSpecializationMatcher.match(fused.getPlan(), fused.getPrecisionMode())
        );
        String constantPool = new String(bytecode, StandardCharsets.ISO_8859_1);

        assertTrue(constantPool.contains("loadVectorBF16Array"), "Expected BF16 vector load helper in generated bytecode");
        assertTrue(constantPool.contains("storeVectorBF16Array"), "Expected BF16 vector store helper in generated bytecode");
    }

    @Test
    void fusedGraphSupportsBoolOutputInVectorMode() {
        int size = 4096;
        double[] aVals = buildInput(size, 0.08);
        double[] bVals = buildInput(size, -0.02);
        double[] cVals = buildInput(size, 0.01);
        double[] dVals = buildInput(size, -0.03);

        Tensor aBase = new Tensor(aVals.clone(), new int[]{size}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(bVals.clone(), new int[]{size}, null, "bBase", DataType.FLOAT64);
        Tensor cBase = new Tensor(cVals.clone(), new int[]{size}, null, "cBase", DataType.FLOAT64);
        Tensor dBase = new Tensor(dVals.clone(), new int[]{size}, null, "dBase", DataType.FLOAT64);
        Tensor baseline = aBase.greaterThan(bBase).logicalOr(cBase.lessThan(dBase));
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        boolean[] expected = baseline.toBooleanArrayCopy().clone();

        Tensor a = new Tensor(aVals.clone(), new int[]{size}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{size}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(cVals.clone(), new int[]{size}, null, "c", DataType.FLOAT64);
        Tensor d = new Tensor(dVals.clone(), new int[]{size}, null, "d", DataType.FLOAT64);
        Tensor out = a.greaterThan(b).logicalOr(c.lessThan(d));

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)));
        prepared.execute(ExecutionMode.FORWARD);
        assertHasPreparedFusedStep(prepared);
        assertArrayEquals(expected, out.toBooleanArrayCopy());
    }

    @Test
    void fusedGraphSupportsOffsetViewInputs() {
        Tensor base = new Tensor(
                new double[]{1, 2, 3, 4, 5, 6},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT64
        );
        Tensor view = base.select(0, 1);

        Tensor baseline = view.relu().exp().mul(view);
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor baseFused = new Tensor(
                new double[]{1, 2, 3, 4, 5, 6},
                new int[]{2, 3},
                null,
                "baseFused",
                DataType.FLOAT64
        );
        Tensor viewFused = baseFused.select(0, 1);
        Tensor out = viewFused.relu().exp().mul(viewFused);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(runtimeConfig(CpuKernelConfig.defaultsTraining()));
        prepared.execute(ExecutionMode.FORWARD);
        assertArrayEquals(expected, out.toDoubleArrayCopy(), EPS);
    }

    @Test
    void representativeFusedHotPathsPrepareGeneratedAsmExecutables() {
        assertGeneratedAsmExecutableFor(new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "cheapA", DataType.FLOAT32)
                .add(new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "cheapB", DataType.FLOAT32))
                .relu()
                .mul(0.5f));

        assertGeneratedAsmExecutableFor(new Tensor(buildInput(4096, 0.03), new int[]{4096}, null, "transA", DataType.FLOAT64)
                .exp()
                .tanh()
                .pow(2.0));

        Tensor broadcastA = new Tensor(new double[]{1, 2, 3, 4}, new int[]{1, 4}, null, "broadcastA", DataType.FLOAT64);
        Tensor broadcastB = new Tensor(new double[]{5, 6, 7, 8, 9, 10, 11, 12}, new int[]{2, 4}, null, "broadcastB", DataType.FLOAT64);
        assertGeneratedAsmExecutableFor(broadcastA.add(broadcastB).mul(broadcastB).sigmoid());
    }

    private static void assertModeMatches(
            double[] expected,
            double[] aVals,
            double[] bVals,
            double[] cVals,
            CpuKernelConfig config
    ) {
        Tensor a = new Tensor(aVals.clone(), new int[]{aVals.length}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{bVals.length}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(cVals.clone(), new int[]{cVals.length}, null, "c", DataType.FLOAT64);

        Tensor out = a.add(b).mul(c).add(a.mul(0.25)).max(b).min(c).sigmoid();
        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(runtimeConfig(config));
        prepared.execute(ExecutionMode.FORWARD);
        assertHasPreparedFusedStep(prepared);
        assertArrayEquals(expected, out.toDoubleArrayCopy(), EPS);
    }

    private static double[] buildInput(int size, double scale) {
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin(i * 0.1) + (i % 17) * scale;
        }
        return out;
    }

    private static void assertBroadcastModeMatches(
            double[] expected,
            CpuKernelConfig config
    ) {
        Tensor a = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8
        }, new int[]{2, 1, 4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{
                10, 20, 30, 40,
                50, 60, 70, 80,
                90, 100, 110, 120
        }, new int[]{1, 3, 4}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(new double[]{
                0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.1, 1.2,
                1.3, 1.4, 1.5, 1.6,
                1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4,
                2.5, 2.6, 2.7, 2.8
        }, new int[]{2, 3, 4}, null, "c", DataType.FLOAT64);

        Tensor out = a.add(b).mul(c).add(a).sigmoid();
        CompiledGraph.compile(out, fuseOnlyInferenceConfig()).prepare(runtimeConfig(config)).execute(ExecutionMode.FORWARD);
        assertArrayEquals(expected, out.toDoubleArrayCopy(), EPS);
    }

    private static double[] runTypedFused(
            double[] aVals,
            double[] bVals,
            double[] cVals,
            DataType dataType
    ) {
        Tensor a = new Tensor(aVals.clone(), new int[]{aVals.length}, null, "aTyped");
        Tensor b = new Tensor(bVals.clone(), new int[]{bVals.length}, null, "bTyped");
        Tensor c = new Tensor(cVals.clone(), new int[]{cVals.length}, null, "cTyped");
        a.setDataType(dataType);
        b.setDataType(dataType);
        c.setDataType(dataType);

        Tensor out = a.add(b).mul(c).add(a.mul(0.25)).max(b).min(c).sigmoid();
        CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(runtimeConfig(CpuKernelConfig.defaultsTraining())).execute(ExecutionMode.FORWARD);
        return out.toDoubleArrayCopy().clone();
    }

    private static double[] expectedTyped(double[] a, double[] b, double[] c, int mode) {
        double[] out = new double[a.length];
        for (int i = 0; i < out.length; i++) {
            double v1 = FusedDTypeOps.add(a[i], b[i], mode);
            double v2 = FusedDTypeOps.mul(v1, c[i], mode);
            double v3 = FusedDTypeOps.mulScalar(a[i], 0.25, mode);
            double v4 = FusedDTypeOps.add(v2, v3, mode);
            double v5 = FusedDTypeOps.max(v4, b[i], mode);
            double v6 = FusedDTypeOps.min(v5, c[i], mode);
            out[i] = FusedDTypeOps.sigmoid(v6, mode);
        }
        return out;
    }

    private static void assertConstantNodeFusion(DataType dataType, int mode, double tolerance) {
        double[] inputValues = buildInput(4096, 0.09);
        Tensor input = new Tensor(inputValues.clone(), new int[]{inputValues.length}, null, "x", dataType);

        operations.elementwise.unary.pow powZeroOp = dataType == DataType.FLOAT64 ? new operations.elementwise.unary.pow(0.0) : new operations.elementwise.unary.pow(0.0f);
        operations.elementwise.unary.mulScalar mulZeroOp = dataType == DataType.FLOAT64 ? new operations.elementwise.unary.mulScalar(0.0) : new operations.elementwise.unary.mulScalar(0.0f);
        Tensor powZero = new Tensor(new int[]{inputValues.length}, List.of(input), powZeroOp, "powZero", dataType);
        Tensor mulZero = new Tensor(new int[]{inputValues.length}, List.of(input), mulZeroOp, "mulZero", dataType);
        Tensor out = powZero.add(mulZero).add(input.pow(-1.0));

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        PreparedExecution prepared = compiledGraph.prepare(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)));
        prepared.execute(ExecutionMode.FORWARD);
        assertHasPreparedFusedStep(prepared);

        double[] expected = new double[inputValues.length];
        for (int i = 0; i < inputValues.length; i++) {
            double pow0 = FusedDTypeOps.pow(inputValues[i], 0.0d, mode);
            double mul0 = FusedDTypeOps.mulScalar(inputValues[i], 0.0d, mode);
            double inv = FusedDTypeOps.pow(inputValues[i], -1.0d, mode);
            expected[i] = FusedDTypeOps.add(FusedDTypeOps.add(pow0, mul0, mode), inv, mode);
        }
        assertArrayEquals(expected, out.toDoubleArrayCopy(), tolerance);
    }

    private static RuntimeConfig runtimeConfig(CpuKernelConfig cpuKernelConfig) {
        return new RuntimeConfig(cpuKernelConfig, config.runtime.ApproximationConfig.defaults(), config.runtime.BlasConfig.disabled());
    }

    private static void assertHasPreparedFusedStep(PreparedExecution prepared) {
        assertTrue(prepared.forwardSteps().stream().anyMatch(step -> testsupport.MetadataArtifacts.fusedExecutable(step.metadata()) != null),
                "Expected prepared fused execution metadata");
    }

    private static void assertGeneratedAsmExecutableFor(Tensor out) {
        PreparedExecution prepared = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)));
        var executable = testsupport.MetadataArtifacts.fusedExecutable(findPreparedFusedStep(prepared).metadata());
        assertTrue(!executable.getClass().getName().contains("InterpretedPreparedFusedExecutable"),
                () -> "Expected generated ASM fused executable, got " + executable.getClass().getName());
    }

    private static PreparedExecutionStep findPreparedFusedStep(PreparedExecution prepared) {
        return prepared.forwardSteps().stream()
                .filter(step -> testsupport.MetadataArtifacts.fusedExecutable(step.metadata()) != null)
                .findFirst()
                .orElseThrow(() -> new IllegalStateException("Expected prepared fused step"));
    }

    private static CompileConfig fuseOnlyInferenceConfig() {
        return CompileConfig.inference().withGraphOptimization(GraphOptimizationConfig.noGraphOptimization());
    }
}
