package tuning.validate;

import java.util.Map;

/**
 * Outcome of validating one candidate workload.
 *
 * @param valid whether measurement may proceed
 * @param status stable status label such as {@code "skipped"} or {@code "invalid"}
 * @param reason human-readable reason, especially for failures
 * @param metrics optional numeric comparison metrics
 */
public record ValidationResult(
        boolean valid,
        String status,
        String reason,
        Map<String, Double> metrics
) {
    public ValidationResult {
        status = status == null ? "unknown" : status;
        reason = reason == null ? "" : reason;
        metrics = metrics == null ? Map.of() : Map.copyOf(metrics);
    }

    /**
     * @return successful result representing disabled or unavailable validation
     */
    public static ValidationResult skipped() {
        return new ValidationResult(true, "skipped", "", Map.of());
    }

    /**
     * Creates a failed validation result.
     *
     * @param reason failure reason
     * @return invalid result
     */
    public static ValidationResult failure(String reason) {
        return new ValidationResult(false, "invalid", reason, Map.of());
    }
}
