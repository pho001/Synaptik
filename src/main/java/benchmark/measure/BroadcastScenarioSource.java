package benchmark.measure;

import benchmark.OptimizerCandidate;

@FunctionalInterface
public interface BroadcastScenarioSource {
    MeasuredBroadcastScenario create(
            double[] baseA,
            double[] baseB,
            double[] baseC,
            OptimizerCandidate candidate
    );
}
