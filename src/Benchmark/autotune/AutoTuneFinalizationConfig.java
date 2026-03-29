package Benchmark.autotune;

public record AutoTuneFinalizationConfig(
        int refineTopK,
        boolean numericsPostcheckEnabled,
        RefineConfig refineConfig
) {
    public AutoTuneFinalizationConfig {
        if (refineTopK <= 0 || refineConfig == null) {
            throw new IllegalArgumentException("Invalid AutoTuneFinalizationConfig");
        }
    }
}
