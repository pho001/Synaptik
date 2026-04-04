package tuning.report;

import java.util.Comparator;
import java.util.Locale;

public final class TextBenchmarkReportRenderer {
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
        report.baselineNoOpt()
                .filter(base -> base.measurement() != null)
                .ifPresent(base -> sb.append("baselineNoOptMedianMs=")
                        .append(String.format(Locale.US, "%.6f", base.measurement().steadyStateStats().medianMs()))
                        .append('\n'));
        report.baselineNoOptConservativeRuntime()
                .filter(base -> base.measurement() != null)
                .ifPresent(base -> sb.append("baselineNoOptConservativeMedianMs=")
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
                "%-34s %-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s%n",
                "name", "status", "compileMs", "prepareMs", "traceMs", "medianMs", "p90Ms", "vsNoOpt", "vsNoOptCR"
        ));
        report.candidates().stream()
                .sorted(Comparator.comparing(r -> r.candidate().name()))
                .forEach(candidate -> {
                    if (candidate.measurement() == null) {
                        sb.append(String.format(
                                Locale.US,
                                "%-34s %-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s%n",
                                candidate.candidate().name(),
                                "FAIL",
                                "n/a",
                                "n/a",
                                "n/a",
                                "n/a",
                                "n/a",
                                "n/a",
                                "n/a"
                        ));
                        return;
                    }
                    var trace = candidate.measurement().trace();
                    var stats = candidate.measurement().steadyStateStats();
                    double speedupNoOpt = report.speedupVsNoOpt(candidate);
                    double speedupNoOptCr = report.speedupVsNoOptConservativeRuntime(candidate);
                    sb.append(String.format(
                            Locale.US,
                            "%-34s %-8s %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f %-12s %-12s%n",
                            label(candidate),
                            candidate.success() ? "OK" : "FAIL",
                            nanosToMs(trace.compile().durationNs()),
                            nanosToMs(trace.prepare().durationNs()),
                            nanosToMs(trace.run().durationNs()),
                            stats.medianMs(),
                            stats.p90Ms(),
                            formatRatio(speedupNoOpt),
                            formatRatio(speedupNoOptCr)
                    ));
                });
        sb.append('\n');

        report.candidates().stream()
                .sorted(Comparator.comparing(r -> r.candidate().name()))
                .forEach(candidate -> {
                    sb.append("- ").append(label(candidate)).append('\n');
                    sb.append("  success=").append(candidate.success()).append('\n');
                    sb.append("  validation=").append(candidate.validation().status()).append('\n');
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
                        sb.append("  steadyStateMeanMs=").append(String.format(Locale.US, "%.6f", stats.meanMs())).append('\n');
                        sb.append("  steadyStateMedianMs=").append(String.format(Locale.US, "%.6f", stats.medianMs())).append('\n');
                        sb.append("  steadyStateP90Ms=").append(String.format(Locale.US, "%.6f", stats.p90Ms())).append('\n');
                        sb.append("  speedupVsNoOpt=").append(formatRatio(report.speedupVsNoOpt(candidate))).append('\n');
                        sb.append("  speedupVsNoOptConservativeRuntime=").append(formatRatio(report.speedupVsNoOptConservativeRuntime(candidate))).append('\n');
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
        return switch (candidate.baselineKind()) {
            case NO_OPT -> candidate.candidate().name() + " [baseline]";
            case NO_OPT_CONSERVATIVE_RUNTIME -> candidate.candidate().name() + " [baseline+conservative-runtime]";
            case NONE -> candidate.candidate().name();
        };
    }

    private static String formatRatio(double ratio) {
        return Double.isFinite(ratio) ? String.format(Locale.US, "%.3fx", ratio) : "n/a";
    }
}
