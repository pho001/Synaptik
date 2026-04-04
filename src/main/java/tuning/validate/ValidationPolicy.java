package tuning.validate;

public record ValidationPolicy(
        double absTolerance,
        double relTolerance,
        boolean requireGradientMatch,
        boolean enabled
) {
    public ValidationPolicy {
        if (absTolerance < 0.0d) {
            throw new IllegalArgumentException("absTolerance must be >= 0");
        }
        if (relTolerance < 0.0d) {
            throw new IllegalArgumentException("relTolerance must be >= 0");
        }
    }

    public static ValidationPolicy disabled() {
        return new ValidationPolicy(0.0d, 0.0d, false, false);
    }

    public static ValidationPolicy defaults() {
        return new ValidationPolicy(1e-9, 1e-9, false, true);
    }
}
