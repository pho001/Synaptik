package Benchmark.autotune;

public record AutoTuneSessionConfig(
        boolean safetySweepOnly,
        boolean safetyStateless,
        AutoTuneFinalizationConfig finalizationConfig
) {}
