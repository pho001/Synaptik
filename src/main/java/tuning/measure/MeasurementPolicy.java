package tuning.measure;

public record MeasurementPolicy(
        int warmupIters,
        int measureIters,
        int repeats,
        boolean measureCompile,
        boolean measurePrepare,
        boolean measureColdRun,
        boolean measureSteadyState,
        boolean captureStepTrace
) {
    public MeasurementPolicy {
        if (warmupIters < 0) {
            throw new IllegalArgumentException("warmupIters must be >= 0");
        }
        if (measureIters < 1) {
            throw new IllegalArgumentException("measureIters must be >= 1");
        }
        if (repeats < 1) {
            throw new IllegalArgumentException("repeats must be >= 1");
        }
    }

    public static MeasurementPolicy defaults() {
        return new MeasurementPolicy(
                3,
                10,
                3,
                true,
                true,
                true,
                true,
                false
        );
    }
}
