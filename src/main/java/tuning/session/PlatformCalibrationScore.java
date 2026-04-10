package tuning.session;

public record PlatformCalibrationScore(
        boolean valid,
        double score,
        double geometricMeanMs,
        double worstBucketMedianMs,
        double variancePenalty,
        String explanation
) {
    public PlatformCalibrationScore {
        explanation = explanation == null ? "" : explanation;
    }

    public static PlatformCalibrationScore invalid(String explanation) {
        return new PlatformCalibrationScore(
                false,
                Double.POSITIVE_INFINITY,
                Double.NaN,
                Double.NaN,
                Double.NaN,
                explanation
        );
    }
}
