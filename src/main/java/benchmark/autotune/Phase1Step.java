package benchmark.autotune;

import benchmark.OptimizerCandidate;

public record Phase1Step(
        OptimizerCandidate candidate,
        Phase1CandidateResult result,
        Phase1Counters counters,
        double rowMs
) {}
