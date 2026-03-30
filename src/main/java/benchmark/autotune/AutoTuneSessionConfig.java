package benchmark.autotune;

public record AutoTuneSessionConfig(
        boolean safetySweepOnly,
        boolean safetyStateless,
        AutoTuneFinalizationConfig finalizationConfig
) {}
