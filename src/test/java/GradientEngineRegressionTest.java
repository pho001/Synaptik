import runtime.contract.ExecutionMode;
import config.optimizer.CseConfig;
import config.optimizer.FuseConfig;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class GradientEngineRegressionTest {
    private static final int VECTOR_LEN = 16;
    private static final double SCALAR_EPS = 1e-12;
    private static final double VECTOR_EPS = 1e-12;

    @Test
    public void testScalarSequenceForwardAndGradientsNoOptVsOpt() {
        RunResult noOpt = runSequence(CompileConfig.training());

        RunResult opt = runSequence(optimizedTrainingConfig());

        assertEquals(64.0, noOpt.te7, SCALAR_EPS);
        assertEquals(-12.8, noOpt.gradA, SCALAR_EPS);
        assertEquals(-48.0, noOpt.gradB, SCALAR_EPS);
        assertEquals(41.6, noOpt.gradC, SCALAR_EPS);

        assertEquals(noOpt.te7, opt.te7, SCALAR_EPS);
        assertEquals(noOpt.gradA, opt.gradA, SCALAR_EPS);
        assertEquals(noOpt.gradB, opt.gradB, SCALAR_EPS);
        assertEquals(noOpt.gradC, opt.gradC, SCALAR_EPS);
    }

    @Test
    public void testVectorSequenceForwardAndGradientsNoOptVsOpt() {
        double[] aData = new double[VECTOR_LEN];
        double[] bData = new double[VECTOR_LEN];
        double[] cData = new double[VECTOR_LEN];
        for (int i = 0; i < VECTOR_LEN; i++) {
            aData[i] = 0.8 + i * 0.03;
            bData[i] = 1.2 + i * 0.02;
            cData[i] = 0.2 + i * 0.01;
        }

        RunResultVec noOpt = runSequenceVec(aData, bData, cData, CompileConfig.training());

        RunResultVec opt = runSequenceVec(aData, bData, cData, optimizedTrainingConfig());

        assertArrayEquals(noOpt.te7, opt.te7, VECTOR_EPS);
        assertArrayEquals(noOpt.gradA, opt.gradA, VECTOR_EPS);
        assertArrayEquals(noOpt.gradB, opt.gradB, VECTOR_EPS);
        assertArrayEquals(noOpt.gradC, opt.gradC, VECTOR_EPS);
    }

    private static RunResult runSequence(CompileConfig optimizerConfig) {
        Tensor A = Tensor.scalar(10.0);
        Tensor B = Tensor.scalar(2.0);
        Tensor C = Tensor.scalar(5.0);
        A.setDataType(DataType.FLOAT64);
        B.setDataType(DataType.FLOAT64);
        C.setDataType(DataType.FLOAT64);
        A.setRequiresGrad(true);
        B.setRequiresGrad(true);
        C.setRequiresGrad(true);

        Tensor Te7 = buildSequence(A, B, C);

        CompiledGraph.compile(Te7, optimizerConfig).prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        return new RunResult(
                Te7.toDoubleArrayCopy()[0],
                A.getGradient().toDoubleArrayCopy()[0],
                B.getGradient().toDoubleArrayCopy()[0],
                C.getGradient().toDoubleArrayCopy()[0]
        );
    }

    private static RunResultVec runSequenceVec(double[] aData, double[] bData, double[] cData, CompileConfig optimizerConfig) {
        Tensor A = new Tensor(aData.clone(), new int[]{aData.length}, null, "A", DataType.FLOAT64);
        Tensor B = new Tensor(bData.clone(), new int[]{bData.length}, null, "B", DataType.FLOAT64);
        Tensor C = new Tensor(cData.clone(), new int[]{cData.length}, null, "C", DataType.FLOAT64);
        A.setRequiresGrad(true);
        B.setRequiresGrad(true);
        C.setRequiresGrad(true);

        Tensor Te7 = buildSequence(A, B, C);

        CompiledGraph.compile(Te7, optimizerConfig).prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        return new RunResultVec(
                Te7.toDoubleArrayCopy().clone(),
                A.getGradient().toDoubleArrayCopy().clone(),
                B.getGradient().toDoubleArrayCopy().clone(),
                C.getGradient().toDoubleArrayCopy().clone()
        );
    }

    private static Tensor buildSequence(Tensor A, Tensor B, Tensor C) {
        Tensor Te1 = A.div(B);
        Tensor Te2 = A.sub(C);
        Tensor Te3 = B.add(C);
        Tensor Te4 = Te1.div(Te2);
        Tensor Te5 = Te3.mul(Te4);
        Tensor Te6 = Te4.add(Te5);
        return Te6.pow(2);
    }

    private record RunResult(double te7, double gradA, double gradB, double gradC) {}

    private record RunResultVec(double[] te7, double[] gradA, double[] gradB, double[] gradC) {}

    private static CompileConfig optimizedTrainingConfig() {
        return CompileConfig.training();
    }
}
