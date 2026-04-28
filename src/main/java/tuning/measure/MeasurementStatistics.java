package tuning.measure;

/**
 * Aggregate latency statistics in milliseconds.
 *
 * @param meanMs arithmetic mean of samples
 * @param medianMs median sample value, used as the primary ranking metric
 * @param p90Ms 90th percentile sample value
 */
public record MeasurementStatistics(
        double meanMs,
        double medianMs,
        double p90Ms
) {
    /**
     * @return zero-valued statistics used when steady-state measurement is skipped
     */
    public static MeasurementStatistics zero() {
        return new MeasurementStatistics(0.0d, 0.0d, 0.0d);
    }
}
