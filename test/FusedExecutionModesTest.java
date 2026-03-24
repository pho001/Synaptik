import Backend.ComputeEngine;
import Config.backend.CpuKernelConfig;
import Graph.codegen.FusedDTypeOps;
import Graph.optimizer.GraphOptimizer;
import Graph.optimizer.rules.FuseElementWiseRule;
import Tensor.DataType;
import Tensor.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class FusedExecutionModesTest {
    private static final double EPS = 1e-9;

    @AfterEach
    void resetConfig() {
        ComputeEngine.setCpuKernelConfig(CpuKernelConfig.defaultsTraining());
    }

    @Test
    void fusedGraphMatchesBaselineAcrossExecutionModes() {
        double[] aVals = buildInput(8192, 0.11);
        double[] bVals = buildInput(8192, -0.07);
        double[] cVals = buildInput(8192, 0.03);

        Tensor aBase = new Tensor(aVals.clone(), new int[]{aVals.length}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(bVals.clone(), new int[]{bVals.length}, null, "bBase", DataType.FLOAT64);
        Tensor cBase = new Tensor(cVals.clone(), new int[]{cVals.length}, null, "cBase", DataType.FLOAT64);
        Tensor baseline = aBase.add(bBase).mul(cBase).add(aBase.mul(0.25)).max(bBase).min(cBase).sigmoid();
        baseline.compute(new GraphOptimizer());
        double[] expected = baseline.toDoubleArrayCopy().clone();

        GraphOptimizer fuseOnly = new GraphOptimizer();
        fuseOnly.addRule(new FuseElementWiseRule());

        // SCALAR
        assertModeMatches(expected, aVals, bVals, cVals, fuseOnly,
                new CpuKernelConfig(4, 32, 32, 32, Integer.MAX_VALUE, Integer.MAX_VALUE));
        // VECTOR
        assertModeMatches(expected, aVals, bVals, cVals, fuseOnly,
                new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE));
        // PARALLEL
        assertModeMatches(expected, aVals, bVals, cVals, fuseOnly,
                new CpuKernelConfig(4, 32, 32, 32, Integer.MAX_VALUE, 1));
        // PARALLEL_VECTOR
        assertModeMatches(expected, aVals, bVals, cVals, fuseOnly,
                new CpuKernelConfig(4, 32, 32, 32, 1, 1));
    }

    @Test
    void fusedGraphRespectsFloat32AndFloat16Modes() {
        int size = 4096;
        double[] aVals = buildInput(size, 0.06);
        double[] bVals = buildInput(size, -0.02);
        double[] cVals = buildInput(size, 0.01);

        GraphOptimizer fuseOnly = new GraphOptimizer();
        fuseOnly.addRule(new FuseElementWiseRule());

        double[] outF32 = runTypedFused(aVals, bVals, cVals, DataType.FLOAT32, fuseOnly);
        double[] outF16 = runTypedFused(aVals, bVals, cVals, DataType.FLOAT16, fuseOnly);

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
        baseline.compute(new GraphOptimizer());
        double[] expected = baseline.toDoubleArrayCopy().clone();

        GraphOptimizer fuseOnly = new GraphOptimizer();
        fuseOnly.addRule(new FuseElementWiseRule());

        assertBroadcastModeMatches(expected, fuseOnly,
                new CpuKernelConfig(4, 32, 32, 32, Integer.MAX_VALUE, Integer.MAX_VALUE));
        assertBroadcastModeMatches(expected, fuseOnly,
                new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE));
        assertBroadcastModeMatches(expected, fuseOnly,
                new CpuKernelConfig(4, 32, 32, 32, Integer.MAX_VALUE, 1));
        assertBroadcastModeMatches(expected, fuseOnly,
                new CpuKernelConfig(4, 32, 32, 32, 1, 1));
    }

    private static void assertModeMatches(
            double[] expected,
            double[] aVals,
            double[] bVals,
            double[] cVals,
            GraphOptimizer fuseOnly,
            CpuKernelConfig config
    ) {
        ComputeEngine.setCpuKernelConfig(config);

        Tensor a = new Tensor(aVals.clone(), new int[]{aVals.length}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{bVals.length}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(cVals.clone(), new int[]{cVals.length}, null, "c", DataType.FLOAT64);

        Tensor out = a.add(b).mul(c).add(a.mul(0.25)).max(b).min(c).sigmoid();
        out.compute(fuseOnly);

        boolean hasFused = out.getCompiledGraph().getCompiledGraphAsList().stream()
                .anyMatch(t -> t.getOperation() != null && t.getOperation().opType() == Operations.Operation.OpType.FUSED);
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
            GraphOptimizer fuseOnly,
            CpuKernelConfig config
    ) {
        ComputeEngine.setCpuKernelConfig(config);

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
        out.compute(fuseOnly);
        assertArrayEquals(expected, out.toDoubleArrayCopy(), EPS);
    }

    private static double[] runTypedFused(
            double[] aVals,
            double[] bVals,
            double[] cVals,
            DataType dataType,
            GraphOptimizer fuseOnly
    ) {
        Tensor a = new Tensor(aVals.clone(), new int[]{aVals.length}, null, "aTyped");
        Tensor b = new Tensor(bVals.clone(), new int[]{bVals.length}, null, "bTyped");
        Tensor c = new Tensor(cVals.clone(), new int[]{cVals.length}, null, "cTyped");
        a.setDataType(dataType);
        b.setDataType(dataType);
        c.setDataType(dataType);

        Tensor out = a.add(b).mul(c).add(a.mul(0.25)).max(b).min(c).sigmoid();
        out.compute(fuseOnly);
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
}
