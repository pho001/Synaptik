package benchmark.measure;

import benchmark.OptimizerCandidate;

@FunctionalInterface
public interface BenchmarkScenarioSource {
    MeasuredBenchmarkScenario create(
            double[] baseA,
            double[] baseB,
            double[] baseC,
            OptimizerCandidate candidate,
            boolean requiresGrad,
            int graphBlocks
    );
}
