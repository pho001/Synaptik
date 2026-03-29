package Benchmark.autotune;

import Benchmark.OptimizerCandidate;

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
