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
        report.baseline()
                .filter(base -> base.measurement() != null)
                .ifPresent(base -> sb.append("  \"baselineMedianMs\": ")
                        .append(format(base.measurement().steadyStateStats().medianMs()))
                        .append(",\n"));
        sb.append("  \"candidates\": [\n");
        for (int i = 0; i < report.candidates().size(); i++) {
            BenchmarkCandidateReport candidate = report.candidates().get(i);
            if (i > 0) {
                sb.append(",\n");
            }
            sb.append("    {\n");
            sb.append("      \"name\": \"").append(escape(candidate.entry().name())).append("\",\n");
            sb.append("      \"role\": \"").append(candidate.entry().role().name()).append("\",\n");
            sb.append("      \"success\": ").append(candidate.success()).append(",\n");
            sb.append("      \"validationStatus\": \"").append(escape(candidate.validation().status())).append("\",\n");
            sb.append("      \"stages\": ").append(stageOrderJson(candidate)).append(",\n");
            sb.append("      \"failureReason\": \"").append(escape(candidate.failureReason())).append("\"");
            if (candidate.measurement() != null) {
                var trace = candidate.measurement().trace();
                var stats = candidate.measurement().steadyStateStats();
                double speedup = report.speedupVsBaseline(candidate);
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
                sb.append("        \"vsBaseline\": ").append(format(speedup)).append("\n");
                sb.append("      },\n");
                sb.append("      \"trace\": {\n");
                sb.append("        \"mode\": \"").append(trace.run().mode().name()).append("\",\n");
                sb.append("        \"stepCount\": ").append(trace.run().steps().size()).append(",\n");
                sb.append("        \"parallelUsed\": ").append(usesParallel(trace.run().steps())).append(",\n");
                sb.append("        \"vectorUsed\": ").append(usesVector(trace.run().steps())).append(",\n");
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

    private static String stageOrderJson(BenchmarkCandidateReport candidate) {
        if (candidate == null || candidate.entry() == null || candidate.entry().profile() == null) {
            return "[]";
        }
        var stages = candidate.entry().profile().optimizer().stageOrder();
        if (stages == null || stages.isEmpty()) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < stages.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append('"').append(escape(stages.get(i).name())).append('"');
        }
        sb.append(']');
        return sb.toString();
    }

    private static boolean usesParallel(java.util.List<graph.execution.trace.ExecutionStepTrace> steps) {
        if (steps == null || steps.isEmpty()) {
            return false;
        }
        for (var step : steps) {
            var metadata = step.metadata();
            if (metadata == null) {
                continue;
            }
            var dispatch = metadata.dispatch();
            if (dispatch != null && isParallelMode(dispatch.mode())) {
                return true;
            }
            var reduction = metadata.reduction();
            if (reduction != null && isParallelMode(reduction.mode())) {
                return true;
            }
            var matMul = metadata.matMul();
            if (matMul != null && matMul.parallel()) {
                return true;
            }
        }
        return false;
    }

    private static boolean usesVector(java.util.List<graph.execution.trace.ExecutionStepTrace> steps) {
        if (steps == null || steps.isEmpty()) {
            return false;
        }
        for (var step : steps) {
            var metadata = step.metadata();
            if (metadata == null) {
                continue;
            }
            var dispatch = metadata.dispatch();
            if (dispatch != null && isVectorMode(dispatch.mode())) {
                return true;
            }
            var reduction = metadata.reduction();
            if (reduction != null && isVectorMode(reduction.mode())) {
                return true;
            }
        }
        return false;
    }

    private static boolean isParallelMode(String mode) {
        return "PARALLEL".equals(mode) || "PARALLEL_VECTOR".equals(mode);
    }

    private static boolean isVectorMode(String mode) {
        return "VECTOR".equals(mode) || "PARALLEL_VECTOR".equals(mode);
    }
}
