import benchmark.scenario.BenchmarkGraphRecipes;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class BenchmarkScenarioRecipeTest {
    @Test
    void optimizerBenchmarkGraphRecipeMatchesInlineFormula() {
        double[] baseA = buildInput(4096, 0.07);
        double[] baseB = buildInput(4096, -0.03);
        double[] baseC = buildInput(4096, 0.02);

        RunResult expected = runInlineBenchmarkGraph(baseA, baseB, baseC, 3);
        RunResult actual = runRecipeBenchmarkGraph(baseA, baseB, baseC, 3);

        assertArrayEquals(expected.out, actual.out, 1e-9);
        assertArrayEquals(expected.gradA, actual.gradA, 1e-9);
        assertArrayEquals(expected.gradB, actual.gradB, 1e-9);
        assertArrayEquals(expected.gradC, actual.gradC, 1e-9);
    }

    @Test
    void broadcastRecipeMatchesInlineFormula() {
        Tensor aInline = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8
        }, new int[]{2, 1, 4}, null, "aInline", DataType.FLOAT64);
        Tensor bInline = new Tensor(new double[]{
                10, 20, 30, 40,
                50, 60, 70, 80,
                90, 100, 110, 120
        }, new int[]{1, 3, 4}, null, "bInline", DataType.FLOAT64);
        Tensor cInline = new Tensor(new double[]{
                0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.1, 1.2,
                1.3, 1.4, 1.5, 1.6,
                1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4,
                2.5, 2.6, 2.7, 2.8
        }, new int[]{2, 3, 4}, null, "cInline", DataType.FLOAT64);

        Tensor expected = aInline.add(bInline).mul(cInline).add(aInline).sigmoid();
        expected.compute();

        Tensor aRecipe = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8
        }, new int[]{2, 1, 4}, null, "aRecipe", DataType.FLOAT64);
        Tensor bRecipe = new Tensor(new double[]{
                10, 20, 30, 40,
                50, 60, 70, 80,
                90, 100, 110, 120
        }, new int[]{1, 3, 4}, null, "bRecipe", DataType.FLOAT64);
        Tensor cRecipe = new Tensor(new double[]{
                0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.1, 1.2,
                1.3, 1.4, 1.5, 1.6,
                1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4,
                2.5, 2.6, 2.7, 2.8
        }, new int[]{2, 3, 4}, null, "cRecipe", DataType.FLOAT64);

        Tensor actual = BenchmarkGraphRecipes.buildBroadcastGraph(aRecipe, bRecipe, cRecipe);
        actual.compute();

        assertArrayEquals(expected.toDoubleArrayCopy(), actual.toDoubleArrayCopy(), 1e-9);
    }

    private static RunResult runInlineBenchmarkGraph(double[] baseA, double[] baseB, double[] baseC, int graphBlocks) {
        Tensor A = new Tensor(baseA.clone(), new int[]{baseA.length}, null, "A", DataType.FLOAT64);
        Tensor B = new Tensor(baseB.clone(), new int[]{baseB.length}, null, "B", DataType.FLOAT64);
        Tensor C = new Tensor(baseC.clone(), new int[]{baseC.length}, null, "C", DataType.FLOAT64);
        A.setRequiresGrad(true);
        B.setRequiresGrad(true);
        C.setRequiresGrad(true);

        Tensor linearIn = prefixTensor("LIN_IN", baseA, true, 64, 64);
        Tensor w1 = prefixTensor("LIN_W1", baseB, false, 64, 64);
        Tensor b1 = prefixTensor("LIN_B1", baseC, false, 64, 64);
        Tensor w2 = prefixTensor("LIN_W2", baseC, false, 64, 64);
        Tensor b2 = prefixTensor("LIN_B2", baseA, false, 64, 64);
        Tensor w3 = prefixTensor("LIN_W3", baseA, false, 64, 64);
        Tensor b3 = prefixTensor("LIN_B3", baseB, false, 64, 64);

        int blocks = Math.max(1, graphBlocks);
        Tensor x = A.mul(0.50).add(B.mul(0.30)).sub(C.mul(0.20));
        for (int i = 0; i < blocks; i++) {
            x = x.mul(0.70).add(B.mul(0.20));
            x = x.sub(C.mul(0.10));
            x = x.add(A.mul(0.05));
            x = x.mul(0.95).add(B.mul(0.03)).sub(C.mul(0.02));
        }
        Tensor linear1 = linearIn.matmul(w1).add(b1);
        Tensor linear2 = linear1.matmul(w2).add(b2);
        Tensor linear3 = linear2.matmul(w3).add(b3);
        Tensor linearScalar = linear3.sum();
        Tensor out = x.mul(x).add(B.mul(0.01)).add(linearScalar);

        out.compute();
        out.getCompiledGraph().setTrainingModeOn();
        out.compute();
        return new RunResult(
                out.toDoubleArrayCopy().clone(),
                A.getGradient().toDoubleArrayCopy().clone(),
                B.getGradient().toDoubleArrayCopy().clone(),
                C.getGradient().toDoubleArrayCopy().clone()
        );
    }

    private static RunResult runRecipeBenchmarkGraph(double[] baseA, double[] baseB, double[] baseC, int graphBlocks) {
        Tensor A = new Tensor(baseA.clone(), new int[]{baseA.length}, null, "A", DataType.FLOAT64);
        Tensor B = new Tensor(baseB.clone(), new int[]{baseB.length}, null, "B", DataType.FLOAT64);
        Tensor C = new Tensor(baseC.clone(), new int[]{baseC.length}, null, "C", DataType.FLOAT64);
        A.setRequiresGrad(true);
        B.setRequiresGrad(true);
        C.setRequiresGrad(true);

        Tensor linearIn = prefixTensor("LIN_IN", baseA, true, 64, 64);
        Tensor w1 = prefixTensor("LIN_W1", baseB, false, 64, 64);
        Tensor b1 = prefixTensor("LIN_B1", baseC, false, 64, 64);
        Tensor w2 = prefixTensor("LIN_W2", baseC, false, 64, 64);
        Tensor b2 = prefixTensor("LIN_B2", baseA, false, 64, 64);
        Tensor w3 = prefixTensor("LIN_W3", baseA, false, 64, 64);
        Tensor b3 = prefixTensor("LIN_B3", baseB, false, 64, 64);

        Tensor out = BenchmarkGraphRecipes.buildOptimizerBenchmarkGraph(
                A, B, C, linearIn, w1, b1, w2, b2, w3, b3, graphBlocks
        );
        out.compute();
        out.getCompiledGraph().setTrainingModeOn();
        out.compute();
        return new RunResult(
                out.toDoubleArrayCopy().clone(),
                A.getGradient().toDoubleArrayCopy().clone(),
                B.getGradient().toDoubleArrayCopy().clone(),
                C.getGradient().toDoubleArrayCopy().clone()
        );
    }

    private static Tensor prefixTensor(String label, double[] data, boolean requiresGrad, int... shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        double[] sliced = new double[size];
        System.arraycopy(data, 0, sliced, 0, size);
        Tensor t = new Tensor(shape, null, label, DataType.FLOAT64);
        t.setData(sliced);
        t.setRequiresGrad(requiresGrad);
        return t;
    }

    private static double[] buildInput(int size, double scale) {
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.cos(i * 0.05) + (i % 11) * scale;
        }
        return out;
    }

    private record RunResult(double[] out, double[] gradA, double[] gradB, double[] gradC) {}
}
