package tuning.report;

import java.util.Comparator;
import java.util.Locale;

public final class TextBenchmarkReportRenderer {
    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_RESET = "\u001B[0m";

    private TextBenchmarkReportRenderer() {
    }

    public static String render(BenchmarkReport report) {
        if (report == null) {
            throw new IllegalArgumentException("report cannot be null");
        }

        StringBuilder sb = new StringBuilder();
        sb.append("Benchmark Report\n");
        sb.append("workload=").append(report.workloadName()).append('\n');
        sb.append("createdAt=").append(report.createdAt()).append('\n');
        sb.append("bestCandidate=").append(report.bestCandidateName().isBlank() ? "n/a" : report.bestCandidateName()).append("\n\n");

        sb.append("Summary\n");
        sb.append("successes=").append(report.successCount()).append('\n');
        sb.append("failures=").append(report.failureCount()).append('\n');
        report.baseline()
                .filter(base -> base.measurement() != null)
                .ifPresent(base -> sb.append("baselineMedianMs=")
                        .append(String.format(Locale.US, "%.6f", base.measurement().steadyStateStats().medianMs()))
                        .append('\n'));
        report.bestCandidate().ifPresent(best -> {
            sb.append("bestMedianMs=").append(String.format(Locale.US, "%.6f", best.measurement().steadyStateStats().medianMs())).append('\n');
            sb.append("bestMeanMs=").append(String.format(Locale.US, "%.6f", best.measurement().steadyStateStats().meanMs())).append('\n');
        });
        sb.append('\n');

        sb.append("Candidates\n");
        sb.append(String.format(
                Locale.US,
                "%-34s %-8s %-12s %-12s %-12s %-12s %-12s %-12s%n",
                "name", "status", "compileMs", "prepareMs", "traceMs", "medianMs", "p90Ms", "vsBaseline"
        ));
        report.candidates().stream()
                .sorted(Comparator.comparing(r -> r.entry().name()))
                .forEach(candidate -> {
                    boolean highlight = shouldHighlight(report, candidate);
                    if (candidate.measurement() == null) {
                        String row = String.format(
                                Locale.US,
                                "%-34s %-8s %-12s %-12s %-12s %-12s %-12s %-12s%n",
                                candidate.entry().name(),
                                "FAIL",
                                "n/a",
                                "n/a",
                                "n/a",
                                "n/a",
                                "n/a",
                                "n/a"
                        );
                        sb.append(colorizeIfNeeded(row, highlight));
                        return;
                    }
                    var trace = candidate.measurement().trace();
                    var stats = candidate.measurement().steadyStateStats();
                    double speedup = report.speedupVsBaseline(candidate);
                    String row = String.format(
                            Locale.US,
                            "%-34s %-8s %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f %-12s%n",
                            label(candidate),
                            candidate.success() ? "OK" : "FAIL",
                            nanosToMs(trace.compile().durationNs()),
                            nanosToMs(trace.prepare().durationNs()),
                            nanosToMs(trace.run().durationNs()),
                            stats.medianMs(),
                            stats.p90Ms(),
                            formatRatio(speedup)
                    );
                    sb.append(colorizeIfNeeded(row, highlight));
                });
        sb.append('\n');

        report.candidates().stream()
                .sorted(Comparator.comparing(r -> r.entry().name()))
                .forEach(candidate -> {
                    sb.append("- ").append(label(candidate)).append('\n');
                    sb.append("  winner=").append(shouldHighlight(report, candidate)).append('\n');
                    sb.append("  success=").append(candidate.success()).append('\n');
                    sb.append("  validation=").append(candidate.validation().status()).append('\n');
                    sb.append("  stages=").append(formatStageOrder(candidate)).append('\n');
                    if (!candidate.failureReason().isBlank()) {
                        sb.append("  failure=").append(candidate.failureReason()).append('\n');
                    }
                    if (candidate.measurement() != null) {
                        var trace = candidate.measurement().trace();
                        var stats = candidate.measurement().steadyStateStats();
                        sb.append("  compileMs=").append(formatMs(trace.compile().durationNs())).append('\n');
                        sb.append("  prepareMs=").append(formatMs(trace.prepare().durationNs())).append('\n');
                        sb.append("  tracedRunMs=").append(formatMs(trace.run().durationNs())).append('\n');
                        sb.append("  stepCount=").append(trace.run().steps().size()).append('\n');
                        sb.append("  parallelUsed=").append(usesParallel(trace.run().steps())).append('\n');
                        sb.append("  vectorUsed=").append(usesVector(trace.run().steps())).append('\n');
                        sb.append("  steadyStateMeanMs=").append(String.format(Locale.US, "%.6f", stats.meanMs())).append('\n');
                        sb.append("  steadyStateMedianMs=").append(String.format(Locale.US, "%.6f", stats.medianMs())).append('\n');
                        sb.append("  steadyStateP90Ms=").append(String.format(Locale.US, "%.6f", stats.p90Ms())).append('\n');
                        sb.append("  speedupVsBaseline=").append(formatRatio(report.speedupVsBaseline(candidate))).append('\n');
                        appendHotSteps(sb, trace.run().steps(), 5);
                    }
                });

        return sb.toString();
    }

    private static String formatMs(long durationNs) {
        return String.format(Locale.US, "%.6f", durationNs / 1_000_000.0d);
    }

    private static double nanosToMs(long durationNs) {
        return durationNs / 1_000_000.0d;
    }

    private static void appendHotSteps(StringBuilder sb, java.util.List<graph.execution.trace.ExecutionStepTrace> steps, int limit) {
        if (steps == null || steps.isEmpty() || limit <= 0) {
            return;
        }
        sb.append("  hotSteps:\n");
        steps.stream()
                .sorted(java.util.Comparator.comparingLong(graph.execution.trace.ExecutionStepTrace::durationNs).reversed())
                .limit(limit)
                .forEach(step -> sb.append("    ")
                        .append(step.index())
                        .append(": ")
                        .append(step.opType())
                        .append(" [")
                        .append(step.label())
                        .append("] ")
                        .append(String.format(Locale.US, "%.6fms", nanosToMs(step.durationNs())))
                        .append('\n'));
    }

    private static String label(BenchmarkCandidateReport candidate) {
        return candidate.baseline() ? candidate.entry().name() + " [baseline]" : candidate.entry().name();
    }

    private static String formatRatio(double ratio) {
        return Double.isFinite(ratio) ? String.format(Locale.US, "%.3fx", ratio) : "n/a";
    }

    private static String formatStageOrder(BenchmarkCandidateReport candidate) {
        if (candidate == null || candidate.entry() == null || candidate.entry().profile() == null) {
            return "[]";
        }
        return candidate.entry().profile().optimizer().stageOrder().toString();
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

    private static boolean shouldHighlight(BenchmarkReport report, BenchmarkCandidateReport candidate) {
        return report != null
                && candidate != null
                && report.candidates().size() > 1
                && !report.bestCandidateName().isBlank()
                && report.bestCandidateName().equals(candidate.entry().name());
    }

    private static String colorizeIfNeeded(String row, boolean highlight) {
        if (!highlight || row == null || row.isEmpty()) {
            return row;
        }
        return ANSI_GREEN + row + ANSI_RESET;
    }
}
