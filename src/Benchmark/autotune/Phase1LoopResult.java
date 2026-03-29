package Benchmark.autotune;

import java.util.List;

public record Phase1LoopResult(
        Phase1Counters counters,
        List<AutoTuneResult> validPhase1
) {}
