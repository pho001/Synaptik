package benchmark.measure;

import benchmark.OptimizerCandidate;

public interface CandidateMeasurementCachePort {
    CandidateMeasurementResult get(OptimizerCandidate candidate, String tier, int warmupIters, int measureIters);

    void put(OptimizerCandidate candidate, String tier, int warmupIters, int measureIters, CandidateMeasurementResult result);
}
