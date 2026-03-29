package Benchmark.autotune;

import Benchmark.OptimizerCandidate;

public record RefineProgressUpdate(
        OptimizerCandidate candidate,
        int refinedIndex,
        int finalists,
        AutoTuneResult bestTraining,
        AutoTuneResult bestInference,
        double rowMs,
        double fwdMs,
        double trainMs,
        double broadcastMs
) {}
