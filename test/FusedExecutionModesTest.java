import Backend.ComputeEngine;
import Config.backend.CpuKernelConfig;
import Graph.optimizer.GraphOptimizer;
import Graph.optimizer.rules.FuseElementWiseRule;
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

        Tensor aBase = new Tensor(aVals.clone(), new int[]{aVals.length}, null, "aBase");
        Tensor bBase = new Tensor(bVals.clone(), new int[]{bVals.length}, null, "bBase");
        Tensor cBase = new Tensor(cVals.clone(), new int[]{cVals.length}, null, "cBase");
        Tensor baseline = aBase.add(bBase).mul(cBase).add(aBase.mul(0.25)).sigmoid();
        baseline.compute(new GraphOptimizer());
        double[] expected = baseline.getData().clone();

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

    private static void assertModeMatches(
            double[] expected,
            double[] aVals,
            double[] bVals,
            double[] cVals,
            GraphOptimizer fuseOnly,
            CpuKernelConfig config
    ) {
        ComputeEngine.setCpuKernelConfig(config);

        Tensor a = new Tensor(aVals.clone(), new int[]{aVals.length}, null, "a");
        Tensor b = new Tensor(bVals.clone(), new int[]{bVals.length}, null, "b");
        Tensor c = new Tensor(cVals.clone(), new int[]{cVals.length}, null, "c");

        Tensor out = a.add(b).mul(c).add(a.mul(0.25)).sigmoid();
        out.compute(fuseOnly);

        boolean hasFused = out.getCompiledGraph().getCompiledGraphAsList().stream()
                .anyMatch(t -> t.getOperation() != null && t.getOperation().opType() == Operations.Operation.OpType.FUSED);
        assertTrue(hasFused, "Expected fused node in compiled graph");
        assertArrayEquals(expected, out.getData(), EPS);
    }

    private static double[] buildInput(int size, double scale) {
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin(i * 0.1) + (i % 17) * scale;
        }
        return out;
    }
}
