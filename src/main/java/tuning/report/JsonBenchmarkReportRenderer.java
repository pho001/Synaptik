package tuning.report;

import java.util.Locale;

public final class JsonBenchmarkReportRenderer {
    private JsonBenchmarkReportRenderer() {
    }

    public static String render(BenchmarkReport report) {
        if (report == null) {
            throw new IllegalArgumentException("report cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"workloadName\": \"").append(escape(report.workloadName())).append("\",\n");
        sb.append("  \"createdAt\": \"").append(report.createdAt()).append("\",\n");
        sb.append("  \"bestCandidateName\": \"").append(escape(report.bestCandidateName())).append("\",\n");
        sb.append("  \"successCount\": ").append(report.successCount()).append(",\n");
        sb.append("  \"failureCount\": ").append(report.failureCount()).append(",\n");
        report.baselineNoOpt()
                .filter(base -> base.measurement() != null)
                .ifPresent(base -> sb.append("  \"baselineNoOptMedianMs\": ")
                        .append(format(base.measurement().steadyStateStats().medianMs()))
                        .append(",\n"));
        report.baselineNoOptConservativeRuntime()
                .filter(base -> base.measurement() != null)
                .ifPresent(base -> sb.append("  \"baselineNoOptConservativeMedianMs\": ")
                        .append(format(base.measurement().steadyStateStats().medianMs()))
                        .append(",\n"));
        sb.append("  \"candidates\": [\n");
        for (int i = 0; i < report.candidates().size(); i++) {
            BenchmarkCandidateReport candidate = report.candidates().get(i);
            if (i > 0) {
                sb.append(",\n");
            }
            sb.append("    {\n");
            sb.append("      \"name\": \"").append(escape(candidate.candidate().name())).append("\",\n");
            sb.append("      \"baselineKind\": \"").append(candidate.baselineKind().name()).append("\",\n");
            sb.append("      \"success\": ").append(candidate.success()).append(",\n");
            sb.append("      \"validationStatus\": \"").append(escape(candidate.validation().status())).append("\",\n");
            sb.append("      \"failureReason\": \"").append(escape(candidate.failureReason())).append("\"");
            if (candidate.measurement() != null) {
                var trace = candidate.measurement().trace();
                var stats = candidate.measurement().steadyStateStats();
                double speedupNoOpt = report.speedupVsNoOpt(candidate);
                double speedupNoOptCr = report.speedupVsNoOptConservativeRuntime(candidate);
                sb.append(",\n");
                sb.append("      \"timing\": {\n");
                sb.append("        \"compileMs\": ").append(format(nanosToMs(trace.compile().durationNs()))).append(",\n");
                sb.append("        \"prepareMs\": ").append(format(nanosToMs(trace.prepare().durationNs()))).append(",\n");
                sb.append("        \"tracedRunMs\": ").append(format(nanosToMs(trace.run().durationNs()))).append(",\n");
                sb.append("        \"meanMs\": ").append(format(stats.meanMs())).append(",\n");
                sb.append("        \"medianMs\": ").append(format(stats.medianMs())).append(",\n");
                sb.append("        \"p90Ms\": ").append(format(stats.p90Ms())).append("\n");
                sb.append("      },\n");
                sb.append("      \"speedup\": {\n");
                sb.append("        \"vsNoOpt\": ").append(format(speedupNoOpt)).append(",\n");
                sb.append("        \"vsNoOptConservativeRuntime\": ").append(format(speedupNoOptCr)).append("\n");
                sb.append("      },\n");
                sb.append("      \"trace\": {\n");
                sb.append("        \"mode\": \"").append(trace.run().mode().name()).append("\",\n");
                sb.append("        \"stepCount\": ").append(trace.run().steps().size()).append(",\n");
                sb.append("        \"hotSteps\": [\n");
                java.util.List<graph.execution.trace.ExecutionStepTrace> hotSteps = trace.run().steps().stream()
                        .sorted(java.util.Comparator.comparingLong(graph.execution.trace.ExecutionStepTrace::durationNs).reversed())
                        .limit(5)
                        .toList();
                for (int j = 0; j < hotSteps.size(); j++) {
                    var step = hotSteps.get(j);
                    if (j > 0) {
                        sb.append(",\n");
                    }
                    sb.append("          {");
                    sb.append("\"index\": ").append(step.index()).append(", ");
                    sb.append("\"label\": \"").append(escape(step.label())).append("\", ");
                    sb.append("\"opType\": \"").append(escape(step.opType())).append("\", ");
                    sb.append("\"durationMs\": ").append(format(nanosToMs(step.durationNs())));
                    sb.append("}");
                }
                sb.append("\n        ]\n");
                sb.append("      }\n");
                sb.append("    }");
            } else {
                sb.append("\n");
                sb.append("    }");
            }
        }
        sb.append("\n  ]\n");
        sb.append("}\n");
        return sb.toString();
    }

    private static String escape(String value) {
        if (value == null) {
            return "";
        }
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static String format(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.6f", value) : "null";
    }

    private static double nanosToMs(long durationNs) {
        return durationNs / 1_000_000.0d;
    }
}
