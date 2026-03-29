package Benchmark.autotune;

public record RefineConfig(int repeats, int warmupIters, int measureIters) {
    public RefineConfig {
        if (repeats <= 0 || warmupIters < 0 || measureIters <= 0) {
            throw new IllegalArgumentException("Invalid RefineConfig");
        }
    }
}
