package tuning.report;

import java.util.Locale;

public final class JsonBenchmarkSuiteReportRenderer {
    private JsonBenchmarkSuiteReportRenderer() {
    }

    public static String render(BenchmarkSuiteReport report) {
        if (report == null) {
            throw new IllegalArgumentException("report cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"createdAt\": \"").append(report.createdAt()).append("\",\n");
        sb.append("  \"totalCandidates\": ").append(report.totalCandidateCount()).append(",\n");
        sb.append("  \"totalSuccesses\": ").append(report.totalSuccessCount()).append(",\n");
        sb.append("  \"totalFailures\": ").append(report.totalFailureCount()).append(",\n");
        report.overallBestCandidate().ifPresent(best -> {
            sb.append("  \"overallBestCandidate\": \"").append(escape(best.entry().name())).append("\",\n");
            sb.append("  \"overallBestMedianMs\": ")
                    .append(format(best.measurement().steadyStateStats().medianMs()))
                    .append(",\n");
        });
        sb.append("  \"candidateSummaries\": [\n");
        for (int i = 0; i < report.candidateSummaries().size(); i++) {
            if (i > 0) {
                sb.append(",\n");
            }
            BenchmarkSuiteCandidateSummary summary = report.candidateSummaries().get(i);
            sb.append("    {");
            sb.append("\"name\": \"").append(escape(summary.candidateName())).append("\", ");
            sb.append("\"role\": \"").append(summary.role().name()).append("\", ");
            sb.append("\"workloads\": ").append(summary.workloadCount()).append(", ");
            sb.append("\"successes\": ").append(summary.successCount()).append(", ");
            sb.append("\"averageMedianMs\": ").append(format(summary.averageMedianMs())).append(", ");
            sb.append("\"averageSpeedupVsBaseline\": ").append(format(summary.averageSpeedupVsBaseline()));
            sb.append("}");
        }
        sb.append("\n  ],\n");
        sb.append("  \"hotspots\": [\n");
        java.util.List<BenchmarkSuiteHotspot> hotspots = report.hotspots(10);
        for (int i = 0; i < hotspots.size(); i++) {
            if (i > 0) {
                sb.append(",\n");
            }
            BenchmarkSuiteHotspot hotspot = hotspots.get(i);
            sb.append("    {");
            sb.append("\"workload\": \"").append(escape(hotspot.workloadName())).append("\", ");
            sb.append("\"candidate\": \"").append(escape(hotspot.candidateName())).append("\", ");
            sb.append("\"opType\": \"").append(escape(hotspot.opType())).append("\", ");
            sb.append("\"label\": \"").append(escape(hotspot.label())).append("\", ");
            sb.append("\"durationMs\": ").append(format(hotspot.durationNs() / 1_000_000.0d));
            sb.append("}");
        }
        sb.append("\n  ],\n");
        sb.append("  \"workloads\": [\n");
        for (int i = 0; i < report.workloadReports().size(); i++) {
            if (i > 0) {
                sb.append(",\n");
            }
            String nested = JsonBenchmarkReportRenderer.render(report.workloadReports().get(i)).indent(4).stripTrailing();
            sb.append(nested);
        }
        sb.append("\n  ]\n");
        sb.append("}\n");
        return sb.toString();
    }

    private static String format(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.6f", value) : "null";
    }

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
