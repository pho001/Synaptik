package tuning.validate;

import java.util.Map;

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

    public static ValidationResult skipped() {
        return new ValidationResult(true, "skipped", "", Map.of());
    }

    public static ValidationResult failure(String reason) {
        return new ValidationResult(false, "invalid", reason, Map.of());
    }
}
