package Benchmark.measure;

public final class MeasurementScoring {
    private MeasurementScoring() {}

    public static double score(
            double forwardMs,
            double trainMs,
            double broadcastMs,
            int graphInfSize,
            int graphTrnSize,
            MeasurementObjective objective
    ) {
        if (objective == MeasurementObjective.INFERENCE) {
            return forwardMs + (0.0005 * graphInfSize);
        }
        return (0.35 * forwardMs)
                + (0.50 * trainMs)
                + (0.0005 * graphTrnSize);
    }
}
