package Benchmark.measure;

public record MeasurementPolicy(int extraPrewarmIters, int warmupIters, int measureIters) {
    public MeasurementPolicy {
        if (extraPrewarmIters < 0) {
            throw new IllegalArgumentException("extraPrewarmIters must be >= 0");
        }
        if (warmupIters < 0) {
            throw new IllegalArgumentException("warmupIters must be >= 0");
        }
        if (measureIters <= 0) {
            throw new IllegalArgumentException("measureIters must be > 0");
        }
    }
}
