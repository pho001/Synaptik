import Benchmark.OptimizerBenchmarkFramework;

/**
 * Backward-compatible facade.
 * Main benchmark implementation now lives in package Benchmark.
 */
public final class OptimizerBenchmark {
    private OptimizerBenchmark() {}

    public static void main(String[] args) {
        run();
    }

    public static void run() {
        OptimizerBenchmarkFramework.run();
    }
}
