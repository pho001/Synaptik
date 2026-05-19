package graph.compile;

/**
 * Pure renderer for an existing backend planning artifact.
 */
public final class BackendPlanningReportRenderer {
    private BackendPlanningReportRenderer() {
    }

    public static String renderText(BackendPlanningResult result) {
        BackendPlanningResult resolved = result == null ? BackendPlanningResult.empty() : result;
        StringBuilder out = new StringBuilder();
        out.append("backendPlanning.jobs=").append(resolved.jobs().size()).append('\n');
        out.append("backendPlanning.plannedPartitions=").append(resolved.plannedPartitions().size()).append('\n');
        for (BackendPlanningDiagnostic diagnostic : resolved.diagnostics()) {
            out.append("diagnostic.")
                    .append(diagnostic.severity().name())
                    .append('.')
                    .append(diagnostic.code())
                    .append('=')
                    .append(diagnostic.message())
                    .append('\n');
        }
        return out.toString();
    }
}
