import graph.optimizer.GraphOptimizer;
import graph.optimizer.rules.MemoryOptimizerRule;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

public class MemoryOptimizerRuleDataTypeTest {

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void memoryRuleMatchesBaselineForTypedExecution(DataType dataType) {
        double[] aData = new double[]{0.8, 0.83, 0.86, 0.89, 0.92, 0.95, 0.98, 1.01};
        double[] bData = new double[]{1.2, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32, 1.34};
        double[] cData = new double[]{0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27};

        RunResult baseline = runSequence(aData, bData, cData, dataType, null);

        GraphOptimizer withMem = new GraphOptimizer();
        withMem.addRule(new MemoryOptimizerRule());
        RunResult mem = runSequence(aData, bData, cData, dataType, withMem);

        double eps = dataType == DataType.FLOAT32 ? 1e-5 : 1e-9;
        assertArrayEquals(baseline.out, mem.out, eps);
        assertArrayEquals(baseline.gradA, mem.gradA, eps);
        assertArrayEquals(baseline.gradB, mem.gradB, eps);
        assertArrayEquals(baseline.gradC, mem.gradC, eps);
    }

    @Test
    void memoryRuleBypassesMixedDTypeGraph() {
        Tensor a = new Tensor(new double[]{1.0}, new int[]{1}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new double[]{2.0}, new int[]{1}, null, "b", DataType.FLOAT64);
        List<Tensor> graph = List.of(a, b);

        List<Tensor> out = new MemoryOptimizerRule().apply(graph);
        assertSame(graph, out);
    }

    @Test
    void memoryRuleBypassesFloat16Graph() {
        Tensor a = new Tensor(new double[]{1.0}, new int[]{1}, null, "a16", DataType.FLOAT16);
        Tensor b = new Tensor(new double[]{2.0}, new int[]{1}, null, "b16", DataType.FLOAT16);
        List<Tensor> graph = List.of(a, b);

        List<Tensor> out = new MemoryOptimizerRule().apply(graph);
        assertSame(graph, out);
    }

    private static RunResult runSequence(double[] aData, double[] bData, double[] cData, DataType dataType, GraphOptimizer optimizer) {
        Tensor A = new Tensor(aData.clone(), new int[]{aData.length}, null, "A", dataType);
        Tensor B = new Tensor(bData.clone(), new int[]{bData.length}, null, "B", dataType);
        Tensor C = new Tensor(cData.clone(), new int[]{cData.length}, null, "C", dataType);
        A.setRequiresGrad(true);
        B.setRequiresGrad(true);
        C.setRequiresGrad(true);

        Tensor out = buildSequence(A, B, C);
        if (optimizer == null) {
            out.compute();
            out.getCompiledGraph().setTrainingModeOn();
            out.compute();
        } else {
            out.compute(optimizer);
            out.getCompiledGraph().setTrainingModeOn();
            out.compute(optimizer);
        }

        return new RunResult(
                out.toDoubleArrayCopy().clone(),
                A.getGradient().toDoubleArrayCopy().clone(),
                B.getGradient().toDoubleArrayCopy().clone(),
                C.getGradient().toDoubleArrayCopy().clone()
        );
    }

    private static Tensor buildSequence(Tensor A, Tensor B, Tensor C) {
        Tensor t1 = A.div(B);
        Tensor t2 = A.sub(C);
        Tensor t3 = B.add(C);
        Tensor t4 = t1.div(t2);
        Tensor t5 = t3.mul(t4);
        Tensor t6 = t4.add(t5);
        return t6.pow(2);
    }

    private record RunResult(double[] out, double[] gradA, double[] gradB, double[] gradC) {}
}

