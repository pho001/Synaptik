package Benchmark.autotune;

import Benchmark.OptimizerCandidate;

public record Phase1Step(
        OptimizerCandidate candidate,
        Phase1CandidateResult result,
        Phase1Counters counters,
        double rowMs
) {}
