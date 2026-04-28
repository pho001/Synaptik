package tuning.validate;

/**
 * Controls candidate validation before measurement.
 *
 * <p>Validation can be disabled for exploratory timing, but production tuning
 * flows should enable it so invalid candidates are reported and skipped before
 * measurement. Dtype-aware profiles derive effective tolerances from the
 * candidate dtype while retaining fixed values as fallbacks.</p>
 *
 * @param absTolerance fixed absolute tolerance or dtype-aware fallback
 * @param relTolerance fixed relative tolerance or dtype-aware fallback
 * @param requireGradientMatch whether gradient targets must also match
 * @param enabled whether validation should run
 * @param toleranceProfile tolerance-selection strategy
 */
public record ValidationPolicy(
        double absTolerance,
        double relTolerance,
        boolean requireGradientMatch,
        boolean enabled,
        ValidationToleranceProfile toleranceProfile
) {
    public ValidationPolicy {
        if (absTolerance < 0.0d) {
            throw new IllegalArgumentException("absTolerance must be >= 0");
        }
        if (relTolerance < 0.0d) {
            throw new IllegalArgumentException("relTolerance must be >= 0");
        }
        toleranceProfile = toleranceProfile == null ? ValidationToleranceProfile.FIXED : toleranceProfile;
    }

    public ValidationPolicy(
            double absTolerance,
            double relTolerance,
            boolean requireGradientMatch,
            boolean enabled
    ) {
        this(absTolerance, relTolerance, requireGradientMatch, enabled, ValidationToleranceProfile.FIXED);
    }

    /**
     * @return policy that skips validation and treats candidates as valid
     */
    public static ValidationPolicy disabled() {
        return new ValidationPolicy(0.0d, 0.0d, false, false);
    }

    /**
     * @return strict fixed-tolerance validation policy
     */
    public static ValidationPolicy defaults() {
        return new ValidationPolicy(1e-9, 1e-9, false, true);
    }

    /**
     * Creates a quick dtype-aware validation policy.
     *
     * @param requireGradientMatch whether gradients must be compared
     * @return quick dtype-aware policy
     */
    public static ValidationPolicy quickDTypeAware(boolean requireGradientMatch) {
        return new ValidationPolicy(1e-8, 1e-8, requireGradientMatch, true, ValidationToleranceProfile.QUICK_DTYPE_AWARE);
    }

    /**
     * Creates a balanced dtype-aware validation policy.
     *
     * @param requireGradientMatch whether gradients must be compared
     * @return balanced dtype-aware policy
     */
    public static ValidationPolicy balancedDTypeAware(boolean requireGradientMatch) {
        return new ValidationPolicy(1e-9, 1e-9, requireGradientMatch, true, ValidationToleranceProfile.BALANCED_DTYPE_AWARE);
    }

    /**
     * Creates a thorough dtype-aware validation policy.
     *
     * @param requireGradientMatch whether gradients must be compared
     * @return thorough dtype-aware policy
     */
    public static ValidationPolicy thoroughDTypeAware(boolean requireGradientMatch) {
        return new ValidationPolicy(1e-9, 1e-9, requireGradientMatch, true, ValidationToleranceProfile.THOROUGH_DTYPE_AWARE);
    }

    /**
     * Resolves the effective absolute tolerance for a dtype.
     *
     * @param dataType dtype being validated
     * @return effective absolute tolerance
     */
    public double absTolerance(tensor.DataType dataType) {
        return toleranceProfile.absTolerance(dataType, absTolerance);
    }

    /**
     * Resolves the effective relative tolerance for a dtype.
     *
     * @param dataType dtype being validated
     * @return effective relative tolerance
     */
    public double relTolerance(tensor.DataType dataType) {
        return toleranceProfile.relTolerance(dataType, relTolerance);
    }
}
