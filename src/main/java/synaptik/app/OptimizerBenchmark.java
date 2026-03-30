package synaptik.app;

import benchmark.OptimizerBenchmarkFramework;

/**
 * Backward-compatible facade for the benchmark entrypoint.
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
