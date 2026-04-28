package tuning.calibration;

/**
 * Numeric score assigned to a platform calibration candidate.
 *
 * <p>Lower {@link #score()} values are better. Invalid scores sort poorly by
 * convention because {@link #score()} is positive infinity.</p>
 *
 * @param valid whether the candidate is eligible for selection
 * @param score comparable scalar score
 * @param geometricMeanMs geometric mean latency component, when available
 * @param worstBucketMedianMs worst workload median component, when available
 * @param variancePenalty variance or stability penalty component, when available
 * @param explanation human-readable metric or invalid reason
 */
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

    /**
     * Creates an invalid score that cannot win selection.
     *
     * @param explanation invalid reason
     * @return invalid score with positive-infinity comparable score
     */
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
