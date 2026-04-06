package tuning.validate;

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

    public static ValidationPolicy disabled() {
        return new ValidationPolicy(0.0d, 0.0d, false, false);
    }

    public static ValidationPolicy defaults() {
        return new ValidationPolicy(1e-9, 1e-9, false, true);
    }

    public static ValidationPolicy quickDTypeAware(boolean requireGradientMatch) {
        return new ValidationPolicy(1e-8, 1e-8, requireGradientMatch, true, ValidationToleranceProfile.QUICK_DTYPE_AWARE);
    }

    public static ValidationPolicy balancedDTypeAware(boolean requireGradientMatch) {
        return new ValidationPolicy(1e-9, 1e-9, requireGradientMatch, true, ValidationToleranceProfile.BALANCED_DTYPE_AWARE);
    }

    public static ValidationPolicy thoroughDTypeAware(boolean requireGradientMatch) {
        return new ValidationPolicy(1e-9, 1e-9, requireGradientMatch, true, ValidationToleranceProfile.THOROUGH_DTYPE_AWARE);
    }

    public double absTolerance(tensor.DataType dataType) {
        return toleranceProfile.absTolerance(dataType, absTolerance);
    }

    public double relTolerance(tensor.DataType dataType) {
        return toleranceProfile.relTolerance(dataType, relTolerance);
    }
}
