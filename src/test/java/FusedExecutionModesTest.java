import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import config.runtime.RuntimeConfig;
import config.backend.CpuKernelConfig;
import graph.CompiledGraph;
import graph.codegen.FusedDTypeOps;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

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
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);
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
    void fusedGraphRespectsFloat32AndFloat16Modes() {
        int size = 4096;
        double[] aVals = buildInput(size, 0.06);
        double[] bVals = buildInput(size, -0.02);
        double[] cVals = buildInput(size, 0.01);

        double[] outF32 = runTypedFused(aVals, bVals, cVals, DataType.FLOAT32);
        double[] outF16 = runTypedFused(aVals, bVals, cVals, DataType.FLOAT16);

        double[] expectedF32 = expectedTyped(aVals, bVals, cVals, FusedDTypeOps.MODE_F32);
        double[] expectedF16 = expectedTyped(aVals, bVals, cVals, FusedDTypeOps.MODE_F16);

        assertArrayEquals(expectedF32, outF32, 1e-6);
        assertArrayEquals(expectedF16, outF16, 2e-3);
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
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
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
    void fusedFloat16MatchesBaselineWithBroadcastInputs() {
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
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aBase.toDoubleArrayCopy(), new int[]{2, 1, 4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bBase.toDoubleArrayCopy(), new int[]{1, 3, 4}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(cBase.toDoubleArrayCopy(), new int[]{2, 3, 4}, null, "c", DataType.FLOAT64);
        a.setDataType(DataType.FLOAT16);
        b.setDataType(DataType.FLOAT16);
        c.setDataType(DataType.FLOAT16);

        Tensor out = a.add(b).mul(c).add(a).sigmoid();
        CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 2e-2);
    }

    @Test
    void fusedFloat16MatchesBaselineWithNonContiguousInputs() {
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
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);
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
        a.setDataType(DataType.FLOAT16);
        b.setDataType(DataType.FLOAT16);

        Tensor out = a.add(b).mul(a).sigmoid();
        CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 2e-2);
    }

    @Test
    void fusedGraphSupportsWhere() {
        Tensor cond = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{4}, null, "cond", DataType.BOOL);
        Tensor aBase = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(new double[]{10, 20, 30, 40}, new int[]{4}, null, "bBase", DataType.FLOAT64);

        Tensor baseline = Tensor.where(cond, aBase, bBase).relu().mul(aBase);
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor condFused = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{4}, null, "condFused", DataType.BOOL);
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30, 40}, new int[]{4}, null, "b", DataType.FLOAT64);

        Tensor out = Tensor.where(condFused, a, b).relu().mul(a);
        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        compiledGraph.execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);

        boolean hasFused = compiledGraph.getCompiledGraphAsList().stream()
                .anyMatch(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.FUSED);
        assertTrue(hasFused, "Expected fused node in graph containing where");
        assertArrayEquals(expected, out.toDoubleArrayCopy(), EPS);
    }

    @Test
    void fusedGraphSupportsCompareAndLogicalBoolOutput() {
        Tensor aBase = new Tensor(new double[]{1, 5, 3, 8}, new int[]{4}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(new double[]{2, 4, 3, 1}, new int[]{4}, null, "bBase", DataType.FLOAT64);
        Tensor cBase = new Tensor(new double[]{0, 6, 2, 9}, new int[]{4}, null, "cBase", DataType.FLOAT64);
        Tensor dBase = new Tensor(new double[]{1, 7, 3, 2}, new int[]{4}, null, "dBase", DataType.FLOAT64);

        Tensor baseline = aBase.greaterThan(bBase).logicalOr(cBase.lessThan(dBase));
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);
        boolean[] expected = baseline.toBooleanArrayCopy().clone();

        Tensor a = new Tensor(new double[]{1, 5, 3, 8}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2, 4, 3, 1}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(new double[]{0, 6, 2, 9}, new int[]{4}, null, "c", DataType.FLOAT64);
        Tensor d = new Tensor(new double[]{1, 7, 3, 2}, new int[]{4}, null, "d", DataType.FLOAT64);

        Tensor out = a.greaterThan(b).logicalOr(c.lessThan(d));
        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        compiledGraph.execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);

        boolean hasFused = compiledGraph.getCompiledGraphAsList().stream()
                .anyMatch(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.FUSED);
        assertTrue(hasFused, "Expected fused node in compare/logical graph");
        assertArrayEquals(expected, out.toBooleanArrayCopy());
    }

    @Test
    void fusedGraphSupportsInternalBoolConditionForWhere() {
        Tensor aBase = new Tensor(new double[]{1, 5, 3, 8}, new int[]{4}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(new double[]{2, 4, 3, 1}, new int[]{4}, null, "bBase", DataType.FLOAT64);
        Tensor xBase = new Tensor(new double[]{10, 20, 30, 40}, new int[]{4}, null, "xBase", DataType.FLOAT64);
        Tensor yBase = new Tensor(new double[]{100, 200, 300, 400}, new int[]{4}, null, "yBase", DataType.FLOAT64);

        Tensor baseline = Tensor.where(aBase.greaterThan(bBase), xBase, yBase).relu().mul(xBase);
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(new double[]{1, 5, 3, 8}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2, 4, 3, 1}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor x = new Tensor(new double[]{10, 20, 30, 40}, new int[]{4}, null, "x", DataType.FLOAT64);
        Tensor y = new Tensor(new double[]{100, 200, 300, 400}, new int[]{4}, null, "y", DataType.FLOAT64);

        Tensor out = Tensor.where(a.greaterThan(b), x, y).relu().mul(x);
        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        compiledGraph.execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);

        boolean hasFused = compiledGraph.getCompiledGraphAsList().stream()
                .anyMatch(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.FUSED);
        assertTrue(hasFused, "Expected fused node in compare-select graph");
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
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aVals.clone(), new int[]{size}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{size}, null, "b", DataType.FLOAT64);
        Tensor x = new Tensor(xVals.clone(), new int[]{size}, null, "x", DataType.FLOAT64);
        Tensor y = new Tensor(yVals.clone(), new int[]{size}, null, "y", DataType.FLOAT64);
        Tensor out = Tensor.where(a.greaterThan(b), x, y).relu().mul(x);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        compiledGraph.execute(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)), ExecutionMode.FORWARD);

        boolean hasFused = compiledGraph.getCompiledGraphAsList().stream()
                .anyMatch(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.FUSED);
        assertTrue(hasFused, "Expected fused node in compare-select vector graph");
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
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aVals.clone(), new int[]{size}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{size}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(cVals.clone(), new int[]{size}, null, "c", DataType.FLOAT64);
        Tensor d = new Tensor(dVals.clone(), new int[]{size}, null, "d", DataType.FLOAT64);
        Tensor x = new Tensor(xVals.clone(), new int[]{size}, null, "x", DataType.FLOAT64);
        Tensor y = new Tensor(yVals.clone(), new int[]{size}, null, "y", DataType.FLOAT64);

        Tensor out = Tensor.where(a.greaterThan(b).logicalOr(c.lessThan(d)), x, y).mul(x);
        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        compiledGraph.execute(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)), ExecutionMode.FORWARD);

        boolean hasFused = compiledGraph.getCompiledGraphAsList().stream()
                .anyMatch(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.FUSED);
        assertTrue(hasFused, "Expected fused node in compare/logical vector graph");
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
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor cond = new Tensor(condVals.clone(), new int[]{size}, null, "cond", DataType.BOOL);
        Tensor x = new Tensor(xVals.clone(), new int[]{size}, null, "x", DataType.FLOAT64);
        Tensor y = new Tensor(yVals.clone(), new int[]{size}, null, "y", DataType.FLOAT64);
        Tensor z = new Tensor(zVals.clone(), new int[]{size}, null, "z", DataType.FLOAT64);
        Tensor out = Tensor.where(cond, x, y).relu().mul(z);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        compiledGraph.execute(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)), ExecutionMode.FORWARD);

        boolean hasFused = compiledGraph.getCompiledGraphAsList().stream()
                .anyMatch(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.FUSED);
        assertTrue(hasFused, "Expected fused node in external-bool vector graph");
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void fusedFloat16SupportsCompareLogicalWhereGraph() {
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
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aVals.clone(), new int[]{size}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{size}, null, "b", DataType.FLOAT64);
        Tensor x = new Tensor(xVals.clone(), new int[]{size}, null, "x", DataType.FLOAT64);
        Tensor y = new Tensor(yVals.clone(), new int[]{size}, null, "y", DataType.FLOAT64);
        a.setDataType(DataType.FLOAT16);
        b.setDataType(DataType.FLOAT16);
        x.setDataType(DataType.FLOAT16);
        y.setDataType(DataType.FLOAT16);

        Tensor out = Tensor.where(a.greaterThan(b).logicalOr(a.lessThan(b)), x, y).sigmoid();
        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        compiledGraph.execute(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)), ExecutionMode.FORWARD);

        boolean hasFused = compiledGraph.getCompiledGraphAsList().stream()
                .anyMatch(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.FUSED);
        assertTrue(hasFused, "Expected fused node in FLOAT16 compare/logical/where graph");
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 2e-2);
    }

    @Test
    void fusedFloat16SupportsAbsAndClampGraph() {
        int size = 4096;
        double[] aVals = buildInput(size, 0.06);
        double[] bVals = buildInput(size, -0.05);

        Tensor aBase = new Tensor(aVals.clone(), new int[]{size}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(bVals.clone(), new int[]{size}, null, "bBase", DataType.FLOAT64);
        Tensor baseline = aBase.sub(bBase).abs().clampMin(-0.25).clampMax(0.75).sigmoid();
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aVals.clone(), new int[]{size}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{size}, null, "b", DataType.FLOAT64);
        a.setDataType(DataType.FLOAT16);
        b.setDataType(DataType.FLOAT16);

        Tensor out = a.sub(b).abs().clampMin(-0.25).clampMax(0.75).sigmoid();
        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        compiledGraph.execute(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)), ExecutionMode.FORWARD);

        boolean hasFused = compiledGraph.getCompiledGraphAsList().stream()
                .anyMatch(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.FUSED);
        assertTrue(hasFused, "Expected fused node in FLOAT16 abs/clamp graph");
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 2e-2);
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
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);
        boolean[] expected = baseline.toBooleanArrayCopy().clone();

        Tensor a = new Tensor(aVals.clone(), new int[]{size}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{size}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(cVals.clone(), new int[]{size}, null, "c", DataType.FLOAT64);
        Tensor d = new Tensor(dVals.clone(), new int[]{size}, null, "d", DataType.FLOAT64);
        Tensor out = a.greaterThan(b).logicalOr(c.lessThan(d));

        CompiledGraph compiledGraph = CompiledGraph.compile(out, fuseOnlyInferenceConfig());
        compiledGraph.execute(runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE)), ExecutionMode.FORWARD);

        boolean hasFused = compiledGraph.getCompiledGraphAsList().stream()
                .anyMatch(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.FUSED);
        assertTrue(hasFused, "Expected fused node in bool-output vector graph");
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
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);
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
        compiledGraph.execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);

        boolean hasFused = compiledGraph.getCompiledGraphAsList().stream()
                .anyMatch(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.FUSED);
        assertTrue(hasFused, "Expected fused node for offset-view input graph");
        assertArrayEquals(expected, out.toDoubleArrayCopy(), EPS);
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
        compiledGraph.execute(runtimeConfig(config), ExecutionMode.FORWARD);

        boolean hasFused = compiledGraph.getCompiledGraphAsList().stream()
                .anyMatch(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.FUSED);
        assertTrue(hasFused, "Expected fused node in compiled graph");
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
        CompiledGraph.compile(out, fuseOnlyInferenceConfig()).execute(runtimeConfig(config), ExecutionMode.FORWARD);
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
                .execute(runtimeConfig(CpuKernelConfig.defaultsTraining()), ExecutionMode.FORWARD);
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

    private static RuntimeConfig runtimeConfig(CpuKernelConfig cpuKernelConfig) {
        return new RuntimeConfig(cpuKernelConfig, config.runtime.ApproximationConfig.defaults(), config.runtime.BlasConfig.disabled());
    }

    private static OptimizerConfig fuseOnlyInferenceConfig() {
        return OptimizerConfig.inferenceDefaults().withStageOrder(java.util.List.of(OptimizerStage.FUSE));
    }
}
