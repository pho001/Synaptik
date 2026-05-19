package graph.execution.trace;

import java.util.Map;

/**
 * Backend contributor diagnostics captured during preparation.
 *
 * @param contributor contributor name
 * @param attributes diagnostic attributes
 */
public record BackendPrepareDiagnosticTrace(
        String contributor,
        Map<String, Object> attributes
) {
    public BackendPrepareDiagnosticTrace {
        contributor = contributor == null ? "" : contributor;
        attributes = Map.copyOf(attributes == null ? Map.of() : attributes);
    }
}
