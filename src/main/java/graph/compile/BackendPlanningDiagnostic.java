package graph.compile;

/**
 * Structured diagnostic emitted by backend ownership planning.
 */
public record BackendPlanningDiagnostic(
        Severity severity,
        String code,
        String message
) {
    public BackendPlanningDiagnostic {
        severity = severity == null ? Severity.INFO : severity;
        code = code == null ? "" : code;
        message = message == null ? "" : message;
    }

    public enum Severity {
        INFO,
        WARNING,
        ERROR
    }
}
