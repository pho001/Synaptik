package Benchmark.measure;

public final class MeasurementExecutor {
    private MeasurementExecutor() {}

    public static double measureAverageMs(MeasurementPolicy policy, Runnable iteration, NanoClock clock) {
        if (policy == null) {
            throw new IllegalArgumentException("policy cannot be null");
        }
        if (iteration == null) {
            throw new IllegalArgumentException("iteration cannot be null");
        }
        NanoClock effectiveClock = clock == null ? NanoClock.SYSTEM : clock;

        for (int i = 0; i < policy.extraPrewarmIters(); i++) {
            iteration.run();
        }
        for (int i = 0; i < policy.warmupIters(); i++) {
            iteration.run();
        }

        long t0 = effectiveClock.nanoTime();
        for (int i = 0; i < policy.measureIters(); i++) {
            iteration.run();
        }
        long t1 = effectiveClock.nanoTime();
        return (t1 - t0) / 1_000_000.0 / policy.measureIters();
    }
}
