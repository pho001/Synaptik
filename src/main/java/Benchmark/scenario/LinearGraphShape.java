package Benchmark.scenario;

public record LinearGraphShape(int batch, int in, int h1, int h2, int out) {
    public LinearGraphShape {
        if (batch <= 0 || in <= 0 || h1 <= 0 || h2 <= 0 || out <= 0) {
            throw new IllegalArgumentException("All linear graph dimensions must be > 0");
        }
    }

    public static LinearGraphShape square64() {
        return new LinearGraphShape(64, 64, 64, 64, 64);
    }
}
