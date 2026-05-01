package tuning.benchmark.report;

public final class JsonGpuCoverageTriageReportRenderer {
    private JsonGpuCoverageTriageReportRenderer() {
    }

    public static String render(GpuCoverageTriageReport report) {
        if (report == null) {
            throw new IllegalArgumentException("report cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        appendHotPathTargets(sb, report);
        sb.append(",\n");
        appendTopGaps(sb, report);
        sb.append(",\n");
        appendCategoryCounts(sb, report);
        sb.append(",\n");
        appendFamilyCounts(sb, report);
        sb.append(",\n");
        appendDownstreamTargets(sb, report);
        sb.append("\n}\n");
        return sb.toString();
    }

    private static void appendHotPathTargets(StringBuilder sb, GpuCoverageTriageReport report) {
        sb.append("  \"hotPathTargets\": [\n");
        for (int i = 0; i < report.hotPathTargets().size(); i++) {
            if (i > 0) {
                sb.append(",\n");
            }
            GpuHotPathCoverageTarget target = report.hotPathTargets().get(i);
            sb.append("    {");
            sb.append("\"workloadName\": \"").append(escape(target.workloadName())).append("\", ");
            sb.append("\"targetKind\": \"").append(escape(target.targetKind())).append("\", ");
            sb.append("\"ownerPhase\": ").append(target.ownerPhase()).append(", ");
            sb.append("\"requirementFamilies\": [");
            appendStringArray(sb, target.requirementFamilies());
            sb.append("], ");
            sb.append("\"rationale\": \"").append(escape(target.rationale())).append("\"");
            sb.append("}");
        }
        sb.append("\n  ]");
    }

    private static void appendTopGaps(StringBuilder sb, GpuCoverageTriageReport report) {
        sb.append("  \"topGaps\": [\n");
        for (int i = 0; i < report.topGaps().size(); i++) {
            if (i > 0) {
                sb.append(",\n");
            }
            GpuCoverageGap gap = report.topGaps().get(i);
            sb.append("    {");
            sb.append("\"workloadName\": \"").append(escape(gap.workloadName())).append("\", ");
            sb.append("\"candidateName\": \"").append(escape(gap.candidateName())).append("\", ");
            sb.append("\"backend\": \"").append(escape(gap.backend())).append("\", ");
            sb.append("\"category\": \"").append(gap.category().name()).append("\", ");
            sb.append("\"reason\": \"").append(escape(gap.reason())).append("\", ");
            sb.append("\"count\": ").append(gap.count()).append(", ");
            sb.append("\"severityScore\": ").append(gap.severityScore()).append(", ");
            sb.append("\"requirementFamily\": \"").append(escape(gap.requirementFamily())).append("\"");
            sb.append("}");
        }
        sb.append("\n  ]");
    }

    private static void appendCategoryCounts(StringBuilder sb, GpuCoverageTriageReport report) {
        sb.append("  \"gapCountsByCategory\": {");
        int index = 0;
        for (var entry : report.gapCountsByCategory().entrySet()) {
            if (index++ > 0) {
                sb.append(", ");
            }
            sb.append("\"").append(entry.getKey().name()).append("\": ").append(entry.getValue());
        }
        sb.append("}");
    }

    private static void appendFamilyCounts(StringBuilder sb, GpuCoverageTriageReport report) {
        sb.append("  \"gapCountsByRequirementFamily\": {");
        int index = 0;
        for (var entry : report.gapCountsByRequirementFamily().entrySet()) {
            if (index++ > 0) {
                sb.append(", ");
            }
            sb.append("\"").append(escape(entry.getKey())).append("\": ").append(entry.getValue());
        }
        sb.append("}");
    }

    private static void appendDownstreamTargets(StringBuilder sb, GpuCoverageTriageReport report) {
        sb.append("  \"downstreamPhaseTargets\": [\n");
        int index = 0;
        for (var entry : report.downstreamPhaseTargets().entrySet()) {
            if (index++ > 0) {
                sb.append(",\n");
            }
            sb.append("    {");
            sb.append("\"phase\": \"Phase ").append(entry.getKey()).append("\", ");
            sb.append("\"targets\": [");
            appendStringArray(sb, entry.getValue());
            sb.append("]}");
        }
        sb.append("\n  ]");
    }

    private static void appendStringArray(StringBuilder sb, java.util.List<String> values) {
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append("\"").append(escape(values.get(i))).append("\"");
        }
    }

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
