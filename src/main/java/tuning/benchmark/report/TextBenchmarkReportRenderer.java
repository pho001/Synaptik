package tuning.benchmark.report;

import java.util.Comparator;
import java.util.Locale;
import java.util.Map;

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
                        sb.append("  cpuMaterializationCount=").append(trace.run().cpuMaterializations().size()).append('\n');
                        sb.append("  parallelUsed=").append(usesParallel(trace.run().steps())).append('\n');
                        sb.append("  vectorUsed=").append(usesVector(trace.run().steps())).append('\n');
                        sb.append("  steadyStateMeanMs=").append(String.format(Locale.US, "%.6f", stats.meanMs())).append('\n');
                        sb.append("  steadyStateMedianMs=").append(String.format(Locale.US, "%.6f", stats.medianMs())).append('\n');
                        sb.append("  steadyStateP90Ms=").append(String.format(Locale.US, "%.6f", stats.p90Ms())).append('\n');
                        sb.append("  speedupVsBaseline=").append(formatRatio(report.speedupVsBaseline(candidate))).append('\n');
                        appendCpuMaterializations(sb, trace.run().cpuMaterializations());
                        appendHotSteps(sb, trace.run().steps(), 5);
                        appendAllSteps(sb, trace.run().steps());
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

    private static void appendCpuMaterializations(
            StringBuilder sb,
            java.util.List<graph.execution.trace.CpuMaterializationTrace> materializations
    ) {
        if (materializations == null || materializations.isEmpty()) {
            return;
        }
        sb.append("  cpuMaterializations:\n");
        for (var materialization : materializations) {
            sb.append("    - nodeId=").append(materialization.nodeId())
                    .append(" reason=").append(materialization.reason())
                    .append(" from=").append(materialization.materializedFrom().isBlank()
                            ? "CPU"
                            : materialization.materializedFrom())
                    .append(" residency=").append(materialization.sourceResidency())
                    .append(" bytes=").append(materialization.bytes())
                    .append(" durationMs=").append(formatMs(materialization.durationNs()))
                    .append(" completed=").append(materialization.completed());
            if (!materialization.detail().isBlank()) {
                sb.append(" detail=").append(materialization.detail());
            }
            sb.append('\n');
        }
    }

    private static void appendHotSteps(StringBuilder sb, java.util.List<graph.execution.trace.ExecutionStepTrace> steps, int limit) {
        if (steps == null || steps.isEmpty() || limit <= 0) {
            return;
        }
        sb.append("  hotSteps:\n");
        steps.stream()
                .sorted(java.util.Comparator.comparingLong(graph.execution.trace.ExecutionStepTrace::durationNs).reversed())
                .limit(limit)
                .forEach(step -> {
                    sb.append("    ")
                            .append(step.index())
                            .append(": ")
                            .append(step.opType())
                            .append(" [")
                            .append(step.label())
                            .append("] ")
                            .append(String.format(Locale.US, "%.6fms", nanosToMs(step.durationNs())));
                    if (step.metadata() != null && step.metadata().fused() != null) {
                        String backend = step.metadata().fused().executionBackend();
                        if (backend != null && !backend.isBlank()) {
                            sb.append(" backend=").append(backend);
                        }
                    }
                    appendMetalHotStepSummary(sb, step);
                    sb.append('\n');
                });
    }

    private static void appendAllSteps(StringBuilder sb, java.util.List<graph.execution.trace.ExecutionStepTrace> steps) {
        if (steps == null || steps.isEmpty()) {
            return;
        }
        sb.append("  steps:\n");
        for (var step : steps) {
            sb.append("    - index=").append(step.index())
                    .append(" label=").append(step.label())
                    .append(" opType=").append(step.opType())
                    .append(" shape=").append(step.shape())
                    .append(" dtype=").append(step.dataType())
                    .append(" backend=").append(step.backend())
                    .append(" kernel=").append(step.kernel())
                    .append(" durationMs=").append(String.format(Locale.US, "%.6f", nanosToMs(step.durationNs())))
                    .append('\n');

            var metadata = step.metadata();
            if (metadata == null) {
                continue;
            }

            sb.append("      kind=").append(metadata.kind()).append('\n');
            if (metadata.attributes() != null && !metadata.attributes().isEmpty()) {
                sb.append("      attributes=").append(formatMap(metadata.attributes())).append('\n');
            }
            if (metadata.compute() != null) {
                sb.append("      compute: ")
                        .append("mode=").append(metadata.compute().mode())
                        .append(" storageType=").append(metadata.compute().storageType())
                        .append(" computeType=").append(metadata.compute().computeType())
                        .append(" backend=").append(metadata.compute().backend())
                        .append(" accumulateType=").append(metadata.compute().accumulateType())
                        .append('\n');
            }
            if (metadata.layout() != null) {
                sb.append("      layout: ")
                        .append("storageOffset=").append(metadata.layout().storageOffset())
                        .append(" contiguous=").append(metadata.layout().contiguous())
                        .append(" stridedPath=").append(metadata.layout().stridedPath())
                        .append(" targetType=").append(metadata.layout().targetType())
                        .append('\n');
            }
            if (metadata.dispatch() != null) {
                sb.append("      dispatch: ")
                        .append("mode=").append(metadata.dispatch().mode())
                        .append(" vectorWidth=").append(metadata.dispatch().vectorWidth())
                        .append(" plannedWorkers=").append(metadata.dispatch().plannedWorkers())
                        .append(" scalarChunkSize=").append(metadata.dispatch().scalarChunkSize())
                        .append(" vectorChunkSize=").append(metadata.dispatch().vectorChunkSize())
                        .append('\n');
            }
            if (metadata.reduction() != null) {
                sb.append("      reduction: ")
                        .append("mode=").append(metadata.reduction().mode())
                        .append(" plannedWorkers=").append(metadata.reduction().plannedWorkers())
                        .append(" chunkSize=").append(metadata.reduction().chunkSize())
                        .append(" vectorWidth=").append(metadata.reduction().vectorWidth())
                        .append(" accuracyMode=").append(metadata.reduction().accuracyMode())
                        .append('\n');
            }
            if (metadata.matMul() != null) {
                sb.append("      matMul: ")
                        .append("useBlas=").append(metadata.matMul().useBlas())
                        .append(" useBatchedBlas=").append(metadata.matMul().useBatchedBlas())
                        .append(" parallel=").append(metadata.matMul().parallel())
                        .append(" tileM=").append(metadata.matMul().tileM())
                        .append(" tileN=").append(metadata.matMul().tileN())
                        .append(" tileK=").append(metadata.matMul().tileK())
                        .append(" plannedWorkers=").append(metadata.matMul().plannedWorkers())
                        .append(" work=").append(metadata.matMul().work())
                        .append(" microKernel=").append(metadata.matMul().microKernel())
                        .append('\n');
            }
            if (metadata.conv() != null) {
                sb.append("      conv: ")
                        .append("executionKind=").append(metadata.conv().executionKind())
                        .append(" lowered=").append(metadata.conv().lowered())
                        .append(" blasUsed=").append(metadata.conv().blasUsed())
                        .append(" blasProvider=").append(metadata.conv().blasProvider())
                        .append(" m=").append(metadata.conv().m())
                        .append(" n=").append(metadata.conv().n())
                        .append(" k=").append(metadata.conv().k())
                        .append(" blasCalls=").append(metadata.conv().blasCalls())
                        .append(" javaCalls=").append(metadata.conv().javaCalls())
                        .append('\n');
            }
            if (metadata.fused() != null) {
                sb.append("      fused: ")
                        .append("precisionMode=").append(metadata.fused().precisionMode())
                        .append(" lowCostHint=").append(metadata.fused().lowCostHint())
                        .append(" dispatchFamily=").append(metadata.fused().dispatchFamily())
                        .append(" schedulerSignature=").append(metadata.fused().schedulerSignature())
                        .append(" executionBackend=").append(metadata.fused().executionBackend())
                        .append(" dispatchScale=").append(metadata.fused().dispatchScale())
                        .append(" fusedNodeCount=").append(metadata.fused().fusedNodeCount())
                        .append(" fusedInputCount=").append(metadata.fused().fusedInputCount())
                        .append('\n');
            }
        }
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

    private static String formatMap(Map<String, Object> map) {
        if (map == null || map.isEmpty()) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder("{");
        int index = 0;
        for (var entry : map.entrySet()) {
            if (index++ > 0) {
                sb.append(", ");
            }
            sb.append(entry.getKey()).append('=').append(entry.getValue());
        }
        sb.append('}');
        return sb.toString();
    }

    private static void appendMetalHotStepSummary(StringBuilder sb, graph.execution.trace.ExecutionStepTrace step) {
        if (step == null || step.metadata() == null || step.metadata().attributes() == null) {
            return;
        }
        Map<String, Object> attrs = step.metadata().attributes();
        if (!attrs.containsKey("metalBridgeAvailable")) {
            return;
        }
        Object path = attrs.get("metalExecutionPath");
        if (path != null && !String.valueOf(path).isBlank()) {
            sb.append(" metalPath=").append(path);
        }
        sb.append(" metalFallback=").append(attrs.getOrDefault("metalUsedCpuFallback", false));
        Object inputBytes = attrs.get("metalInputBytes");
        Object outputBytes = attrs.get("metalOutputBytes");
        if (inputBytes != null || outputBytes != null) {
            sb.append(" metalBytes=").append(inputBytes == null ? 0 : inputBytes)
                    .append("->").append(outputBytes == null ? 0 : outputBytes);
        }
        sb.append(" metalCopyInMs=").append(formatNsAttr(attrs.get("metalJavaToNativeCopyNs")))
                .append(" metalNativeMs=").append(formatNsAttr(attrs.get("metalNativeExecuteNs")))
                .append(" metalCopyOutMs=").append(formatNsAttr(attrs.get("metalNativeToJavaCopyNs")));
        Object reason = attrs.get("metalFallbackReason");
        if (Boolean.TRUE.equals(attrs.get("metalUsedCpuFallback")) && reason != null && !String.valueOf(reason).isBlank()) {
            sb.append(" reason=").append(reason);
        }
    }

    private static String formatNsAttr(Object value) {
        if (!(value instanceof Number number)) {
            return "0.000000";
        }
        return String.format(Locale.US, "%.6f", number.longValue() / 1_000_000.0d);
    }
}
