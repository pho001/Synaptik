package graph.execution.trace;

public record PrepareTrace(
        boolean measured,
        long durationNs,
        int forwardStepCount,
        int backwardStepCount
) {
    public static PrepareTrace skipped() {
        return new PrepareTrace(false, 0L, 0, 0);
    }
}
