import backend.ComputeEngine;
import benchmark.BlasPolicyConfigurer;
import benchmark.OptimizerBuilder;
import benchmark.OptimizerCandidate;
import benchmark.TuningKnobs;
import benchmark.scenario.BenchmarkGraphRecipes;
import benchmark.scenario.BenchmarkScenarioFactory;
import benchmark.scenario.LinearGraphShape;
import benchmark.scenario.PreparedBenchmarkScenario;
import benchmark.scenario.PreparedBroadcastScenario;
import benchmark.scenario.ScenarioTensorFactory;
import graph.optimizer.GraphOptimizer;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class BenchmarkScenarioFactoryTest {

    @AfterEach
    void resetBackendConfig() {
        ComputeEngine.setCpuKernelConfig(config.backend.CpuKernelConfig.defaultsTraining());
    }

    @Test
    void preparedBenchmarkScenarioMatchesManualAssembly() {
        OptimizerCandidate candidate = new OptimizerCandidate("NO_OPT_TEST", java.util.List.of(), TuningKnobs.trainingDefaults());
        double[] baseA = buildInput(4096, 0.05);
        double[] baseB = buildInput(4096, -0.02);
        double[] baseC = buildInput(4096, 0.03);
        LinearGraphShape shape = LinearGraphShape.square64();

        RunResult expected = runManualBenchmarkScenario(candidate, baseA, baseB, baseC, shape, true, 2);

        PreparedBenchmarkScenario actual = BenchmarkScenarioFactory.createOptimizerBenchmarkScenario(
                baseA, baseB, baseC, candidate, DataType.FLOAT64, true, 2, shape
        );
        actual.output().compute(actual.optimizer());
        actual.output().getCompiledGraph().setTrainingModeOn();
        actual.output().compute(actual.optimizer());

        assertArrayEquals(expected.out, actual.output().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(expected.gradA, actual.a().getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(expected.gradB, actual.b().getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(expected.gradC, actual.c().getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void preparedBroadcastScenarioMatchesManualAssembly() {
        OptimizerCandidate candidate = new OptimizerCandidate("NO_OPT_BCAST", java.util.List.of(), TuningKnobs.trainingDefaults());
        double[] baseA = buildInput(2 * 1 * 4, 0.1);
        double[] baseB = buildInput(1 * 3 * 4, -0.05);
        double[] baseC = buildInput(2 * 3 * 4, 0.02);

        double[] expected = runManualBroadcastScenario(candidate, baseA, baseB, baseC, 2, 3, 4);

        PreparedBroadcastScenario actual = BenchmarkScenarioFactory.createBroadcastScenario(
                baseA, baseB, baseC, candidate, DataType.FLOAT64, 2, 3, 4
        );
        actual.output().compute(actual.optimizer());

        assertArrayEquals(expected, actual.output().toDoubleArrayCopy(), 1e-9);
    }

    private static RunResult runManualBenchmarkScenario(
            OptimizerCandidate candidate,
            double[] baseA,
            double[] baseB,
            double[] baseC,
            LinearGraphShape shape,
            boolean requiresGrad,
            int graphBlocks
    ) {
        BlasPolicyConfigurer.apply(candidate.knobs());
        ComputeEngine.setCpuKernelConfig(candidate.knobs().kernelConfig().cpu());

        Tensor A = ScenarioTensorFactory.flatTensor("A", baseA, requiresGrad, DataType.FLOAT64);
        Tensor B = ScenarioTensorFactory.flatTensor("B", baseB, requiresGrad, DataType.FLOAT64);
        Tensor C = ScenarioTensorFactory.flatTensor("C", baseC, requiresGrad, DataType.FLOAT64);
        Tensor linearIn = ScenarioTensorFactory.prefixTensorStrict("LIN_IN", baseA, requiresGrad, DataType.FLOAT64, shape.batch(), shape.in());
        Tensor w1 = ScenarioTensorFactory.prefixTensorStrict("LIN_W1", baseB, false, DataType.FLOAT64, shape.in(), shape.h1());
        Tensor b1 = ScenarioTensorFactory.prefixTensorStrict("LIN_B1", baseC, false, DataType.FLOAT64, shape.batch(), shape.h1());
        Tensor w2 = ScenarioTensorFactory.prefixTensorStrict("LIN_W2", baseC, false, DataType.FLOAT64, shape.h1(), shape.h2());
        Tensor b2 = ScenarioTensorFactory.prefixTensorStrict("LIN_B2", baseA, false, DataType.FLOAT64, shape.batch(), shape.h2());
        Tensor w3 = ScenarioTensorFactory.prefixTensorStrict("LIN_W3", baseA, false, DataType.FLOAT64, shape.h2(), shape.out());
        Tensor b3 = ScenarioTensorFactory.prefixTensorStrict("LIN_B3", baseB, false, DataType.FLOAT64, shape.batch(), shape.out());

        Tensor out = BenchmarkGraphRecipes.buildOptimizerBenchmarkGraph(A, B, C, linearIn, w1, b1, w2, b2, w3, b3, graphBlocks);
        GraphOptimizer optimizer = OptimizerBuilder.build(candidate);
        out.prepareCompiledGraph(optimizer);
        out.compute(optimizer);
        out.getCompiledGraph().setTrainingModeOn();
        out.compute(optimizer);

        return new RunResult(
                out.toDoubleArrayCopy().clone(),
                A.getGradient().toDoubleArrayCopy().clone(),
                B.getGradient().toDoubleArrayCopy().clone(),
                C.getGradient().toDoubleArrayCopy().clone()
        );
    }

    private static double[] runManualBroadcastScenario(
            OptimizerCandidate candidate,
            double[] baseA,
            double[] baseB,
            double[] baseC,
            int b0,
            int b1,
            int f
    ) {
        BlasPolicyConfigurer.apply(candidate.knobs());
        ComputeEngine.setCpuKernelConfig(candidate.knobs().kernelConfig().cpu());

        Tensor A = ScenarioTensorFactory.shapedTensor("BA", baseA, false, DataType.FLOAT64, new int[]{b0, 1, f});
        Tensor B = ScenarioTensorFactory.shapedTensor("BB", baseB, false, DataType.FLOAT64, new int[]{1, b1, f});
        Tensor C = ScenarioTensorFactory.shapedTensor("BC", baseC, false, DataType.FLOAT64, new int[]{b0, b1, f});
        Tensor out = BenchmarkGraphRecipes.buildBroadcastGraph(A, B, C);
        GraphOptimizer optimizer = OptimizerBuilder.build(candidate);
        out.prepareCompiledGraph(optimizer);
        out.compute(optimizer);
        return out.toDoubleArrayCopy().clone();
    }

    private static double[] buildInput(int size, double scale) {
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin(i * 0.09) + (i % 13) * scale;
        }
        return out;
    }

    private record RunResult(double[] out, double[] gradA, double[] gradB, double[] gradC) {}
}
