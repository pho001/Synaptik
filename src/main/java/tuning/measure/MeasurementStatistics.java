package tuning.measure;

public record MeasurementStatistics(
        double meanMs,
        double medianMs,
        double p90Ms
) {
    public static MeasurementStatistics zero() {
        return new MeasurementStatistics(0.0d, 0.0d, 0.0d);
    }
}
