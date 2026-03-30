package benchmark.autotune;

import benchmark.OptimizerCandidate;

@FunctionalInterface
public interface CandidatePerfSource {
    CandidatePerf measure(
            OptimizerCandidate candidate,
            int warmupIters,
            int measureIters,
            String tier,
            CandidateEvalCache cache
    );
}
