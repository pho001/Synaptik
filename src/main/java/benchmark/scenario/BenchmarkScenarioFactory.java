package benchmark.scenario;

import backend.ComputeEngine;
import benchmark.BlasPolicyConfigurer;
import benchmark.OptimizerBuilder;
import benchmark.OptimizerCandidate;
import graph.optimizer.GraphOptimizer;
import tensor.DataType;
import tensor.Tensor;

public final class BenchmarkScenarioFactory {
    private BenchmarkScenarioFactory() {}

    public static PreparedBenchmarkScenario createOptimizerBenchmarkScenario(
            double[] baseA,
            double[] baseB,
            double[] baseC,
            OptimizerCandidate candidate,
            DataType dataType,
            boolean requiresGrad,
            int graphBlocks,
            LinearGraphShape linearShape
    ) {
        if (candidate == null) {
            throw new IllegalArgumentException("candidate cannot be null");
        }
        LinearGraphShape shape = linearShape == null ? LinearGraphShape.square64() : linearShape;

        BlasPolicyConfigurer.apply(candidate.knobs());
        ComputeEngine.setCpuKernelConfig(candidate.knobs().kernelConfig().cpu());

        Tensor A = ScenarioTensorFactory.flatTensor("A", baseA, requiresGrad, dataType);
        Tensor B = ScenarioTensorFactory.flatTensor("B", baseB, requiresGrad, dataType);
        Tensor C = ScenarioTensorFactory.flatTensor("C", baseC, requiresGrad, dataType);

        Tensor linearIn = ScenarioTensorFactory.prefixTensorStrict("LIN_IN", baseA, requiresGrad, dataType, shape.batch(), shape.in());
        Tensor w1 = ScenarioTensorFactory.prefixTensorStrict("LIN_W1", baseB, false, dataType, shape.in(), shape.h1());
        Tensor b1 = ScenarioTensorFactory.prefixTensorStrict("LIN_B1", baseC, false, dataType, shape.batch(), shape.h1());
        Tensor w2 = ScenarioTensorFactory.prefixTensorStrict("LIN_W2", baseC, false, dataType, shape.h1(), shape.h2());
        Tensor b2 = ScenarioTensorFactory.prefixTensorStrict("LIN_B2", baseA, false, dataType, shape.batch(), shape.h2());
        Tensor w3 = ScenarioTensorFactory.prefixTensorStrict("LIN_W3", baseA, false, dataType, shape.h2(), shape.out());
        Tensor b3 = ScenarioTensorFactory.prefixTensorStrict("LIN_B3", baseB, false, dataType, shape.batch(), shape.out());

        Tensor out = BenchmarkGraphRecipes.buildOptimizerBenchmarkGraph(
                A, B, C, linearIn, w1, b1, w2, b2, w3, b3, graphBlocks
        );
        GraphOptimizer optimizer = OptimizerBuilder.build(candidate);
        out.prepareCompiledGraph(optimizer);
        return new PreparedBenchmarkScenario(A, B, C, out, optimizer);
    }

    public static PreparedBroadcastScenario createBroadcastScenario(
            double[] baseA,
            double[] baseB,
            double[] baseC,
            OptimizerCandidate candidate,
            DataType dataType,
            int b0,
            int b1,
            int f
    ) {
        if (candidate == null) {
            throw new IllegalArgumentException("candidate cannot be null");
        }

        BlasPolicyConfigurer.apply(candidate.knobs());
        ComputeEngine.setCpuKernelConfig(candidate.knobs().kernelConfig().cpu());

        Tensor A = ScenarioTensorFactory.shapedTensor("BA", baseA, false, dataType, new int[]{b0, 1, f});
        Tensor B = ScenarioTensorFactory.shapedTensor("BB", baseB, false, dataType, new int[]{1, b1, f});
        Tensor C = ScenarioTensorFactory.shapedTensor("BC", baseC, false, dataType, new int[]{b0, b1, f});

        Tensor out = BenchmarkGraphRecipes.buildBroadcastGraph(A, B, C);
        GraphOptimizer optimizer = OptimizerBuilder.build(candidate);
        out.prepareCompiledGraph(optimizer);
        return new PreparedBroadcastScenario(out, optimizer);
    }
}
