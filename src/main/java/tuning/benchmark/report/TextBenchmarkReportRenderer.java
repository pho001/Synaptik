package tuning.benchmark.report;

import backend.accelerator.lowering.GpuLoweredRegionManifestRenderer;

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
                    sb.append("  compile=").append(formatCompilePolicy(candidate)).append('\n');
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
                        sb.append("  hostDeviceTransferCount=").append(trace.run().hostDeviceTransfers().size()).append('\n');
                        appendNativeCpuSummary(sb, NativeCpuTraceSummary.fromSteps(trace.run().steps()));
                        appendNativeCpuRegionSummary(sb, NativeCpuRegionTraceSummary.fromSteps(trace.run().steps()));
                        appendRuntimeCopySummary(sb, RuntimeCopyTraceSummary.fromRun(trace.run()));
                        appendBf16PerformanceSummary(sb, Bf16PerformanceSummary.fromTrace(trace));
                        appendHostDeviceTransferSummary(sb, HostDeviceTransferSummary.fromRun(trace.run()));
                        appendNativeCpuMemorySummary(sb, trace.run().nativeCpuMemory());
                        appendNativeOptimizerSummary(sb, trace.run().nativeOptimizers());
                        appendAcceleratorSummary(sb, AcceleratorTraceSummary.fromSteps(trace.run().steps()));
                        appendGpuCoverageSummary(sb, GpuCoverageSummary.fromTrace(trace));
                        appendCrossBackendRouterEvidence(sb, CrossBackendRouterEvidence.fromTrace(trace));
                        appendBackendSelectionCost(sb, trace.prepare().backendSelection());
                        appendOptimizerCost(sb, trace.compile().optimizerTrace());
                        sb.append("  parallelUsed=").append(usesParallel(trace.run().steps())).append('\n');
                        sb.append("  vectorUsed=").append(usesVector(trace.run().steps())).append('\n');
                        sb.append("  steadyStateMeanMs=").append(String.format(Locale.US, "%.6f", stats.meanMs())).append('\n');
                        sb.append("  steadyStateMedianMs=").append(String.format(Locale.US, "%.6f", stats.medianMs())).append('\n');
                        sb.append("  steadyStateP90Ms=").append(String.format(Locale.US, "%.6f", stats.p90Ms())).append('\n');
                        sb.append("  speedupVsBaseline=").append(formatRatio(report.speedupVsBaseline(candidate))).append('\n');
                        appendCpuMaterializations(sb, trace.run().cpuMaterializations());
                        appendHostDeviceTransfers(sb, trace.run().hostDeviceTransfers());
                        appendHotSteps(sb, trace.run().steps(), 5);
                        appendAllSteps(sb, trace.run().steps());
                    }
                });

        return sb.toString();
    }

    private static void appendNativeCpuSummary(StringBuilder sb, NativeCpuTraceSummary summary) {
        if (summary == null || !summary.present()) {
            return;
        }
        sb.append("  nativeCpuSummary=")
                .append("nativeKernelCount=").append(summary.nativeKernelCount())
                .append(" arrayKernelCount=").append(summary.arrayKernelCount())
                .append(" fallbackCount=").append(summary.fallbackCount())
                .append('\n');
    }

    private static void appendNativeCpuRegionSummary(StringBuilder sb, NativeCpuRegionTraceSummary summary) {
        if (summary == null || !summary.present()) {
            return;
        }
        sb.append("  nativeCpuRegionSummary=")
                .append("selectedRegionCount=").append(summary.selectedRegionCount())
                .append(" rejectedRegionCount=").append(summary.rejectedRegionCount())
                .append(" nativeRouteCount=").append(summary.nativeRouteCount())
                .append(" fallbackCount=").append(summary.fallbackCount())
                .append(" measuredWinClaimCount=").append(summary.measuredWinClaimCount())
                .append(" measuredWinProofCount=").append(summary.measuredWinProofCount())
                .append(" providerNodeCount=").append(summary.providerNodeCount())
                .append(" localKernelNodeCount=").append(summary.localKernelNodeCount())
                .append(" segmentScalarNodeCount=").append(summary.segmentScalarNodeCount())
                .append(" stridedNodeCount=").append(summary.stridedNodeCount())
                .append(" stridedMaterializationCount=").append(summary.stridedMaterializationCount())
                .append(" benchmarkRowCounts=").append(summary.benchmarkRowCounts())
                .append(" layoutClassCounts=").append(summary.layoutClassCounts())
                .append(" regionResultResidencyCounts=").append(summary.regionResultResidencyCounts())
                .append(" regionAutoEligibleNodeCount=").append(summary.regionAutoEligibleNodeCount())
                .append(" boundaryOutputCount=").append(summary.boundaryOutputCount())
                .append(" fallbackReasons=").append(summary.fallbackReasons())
                .append(" stridedFallbackReasons=").append(summary.stridedFallbackReasons())
                .append(" rejectionReasons=").append(summary.rejectionReasons())
                .append('\n');
    }

    private static void appendRuntimeCopySummary(StringBuilder sb, RuntimeCopyTraceSummary summary) {
        if (summary == null || !summary.present()) {
            return;
        }
        sb.append("  runtimeCopySummary=")
                .append("cpuMaterializationBytes=").append(summary.cpuMaterializationBytes())
                .append(" cpuMaterializationDurationNs=").append(summary.cpuMaterializationDurationNs())
                .append(" matMulCopyInBytes=").append(summary.matMulCopyInBytes())
                .append(" matMulCopyOutBytes=").append(summary.matMulCopyOutBytes())
                .append(" matMulNativeTempBytes=").append(summary.matMulNativeTempBytes())
                .append('\n');
    }

    private static void appendBf16PerformanceSummary(StringBuilder sb, Bf16PerformanceSummary summary) {
        if (summary == null || !summary.present()) {
            return;
        }
        sb.append("  bf16PerformanceSummary=")
                .append("matMulStepCount=").append(summary.matMulStepCount())
                .append(" bgemmOutputCount=").append(summary.bgemmOutputCount())
                .append(" sbgemmContinuationCount=").append(summary.sbgemmContinuationCount())
                .append(" promotedF32Count=").append(summary.promotedF32Count())
                .append(" promotedNonBlasStepCount=").append(summary.promotedNonBlasStepCount())
                .append(" promotedNonBlasRegionNodeCount=").append(summary.promotedNonBlasRegionNodeCount())
                .append(" promotedNonBlasSegmentScalarCount=").append(summary.promotedNonBlasSegmentScalarCount())
                .append(" promotedNonBlasArrayFallbackCount=").append(summary.promotedNonBlasArrayFallbackCount())
                .append(" javaRouteCount=").append(summary.javaRouteCount())
                .append(" unavailableRouteCount=").append(summary.unavailableRouteCount())
                .append(" openblasSbgemmAvailable=").append(summary.openblasSbgemmAvailable())
                .append(" openblasBgemmAvailable=").append(summary.openblasBgemmAvailable())
                .append(" copyInBytes=").append(summary.copyInBytes())
                .append(" copyOutBytes=").append(summary.copyOutBytes())
                .append(" nativeTempBytes=").append(summary.nativeTempBytes())
                .append(" optimizerTraceCount=").append(summary.optimizerTraceCount())
                .append(" optimizerArrayFallbackCount=").append(summary.optimizerArrayFallbackCount())
                .append(" optimizerNativeCount=").append(summary.optimizerNativeCount())
                .append(" activationsOnlyPolicyCount=").append(summary.activationsOnlyPolicyCount())
                .append(" f32MasterPolicyCount=").append(summary.f32MasterPolicyCount())
                .append(" experimentalPolicyCount=").append(summary.experimentalPolicyCount())
                .append(" fallbackReasons=").append(summary.fallbackReasons())
                .append('\n');
    }

    private static void appendHostDeviceTransferSummary(StringBuilder sb, HostDeviceTransferSummary summary) {
        if (summary == null || !summary.present()) {
            return;
        }
        sb.append("  hostDeviceTransferSummary=")
                .append("transferCount=").append(summary.transferCount())
                .append(" bytes=").append(summary.bytes())
                .append(" javaArrayBytes=").append(summary.javaArrayBytes())
                .append(" nativeBytes=").append(summary.nativeBytes())
                .append(" deviceBytes=").append(summary.deviceBytes())
                .append(" fallbackCount=").append(summary.fallbackCount())
                .append('\n');
    }

    private static void appendNativeCpuMemorySummary(
            StringBuilder sb,
            graph.execution.trace.NativeCpuMemoryTrace trace
    ) {
        if (trace == null || !trace.present()) {
            return;
        }
        sb.append("  nativeCpuMemory=")
                .append("allocationCount=").append(trace.allocationCount())
                .append(" releaseCount=").append(trace.releaseCount())
                .append(" retainCount=").append(trace.retainCount())
                .append(" allocationFailureCount=").append(trace.allocationFailureCount())
                .append(" requestedPoolPolicy=").append(trace.requestedPoolPolicy())
                .append(" effectivePoolPolicy=").append(trace.effectivePoolPolicy())
                .append(" requestedBytes=").append(trace.requestedBytes())
                .append(" allocatedBytes=").append(trace.allocatedBytes())
                .append(" currentLiveBytes=").append(trace.currentLiveBytes())
                .append(" peakLiveBytes=").append(trace.peakLiveBytes())
                .append(" retainedBytes=").append(trace.retainedBytes())
                .append(" poolHitCount=").append(trace.poolHitCount())
                .append(" poolMissCount=").append(trace.poolMissCount())
                .append(" pooledBytes=").append(trace.pooledBytes())
                .append(" reusedBytes=").append(trace.reusedBytes())
                .append(" discardedBytes=").append(trace.discardedBytes())
                .append(" wastedBytes=").append(trace.wastedBytes())
                .append('\n');
    }

    private static void appendNativeOptimizerSummary(
            StringBuilder sb,
            java.util.List<graph.execution.trace.NativeOptimizerTrace> traces
    ) {
        if (traces == null || traces.isEmpty()) {
            return;
        }
        long nativeCount = traces.stream().filter(trace -> "CPU_NATIVE".equals(trace.route())).count();
        long arrayCount = traces.stream().filter(trace -> "CPU_ARRAY".equals(trace.route())).count();
        long metalCount = traces.stream().filter(trace -> "GPU_METAL".equals(trace.route())).count();
        long elements = traces.stream().mapToLong(graph.execution.trace.NativeOptimizerTrace::elementCount).sum();
        sb.append("  nativeOptimizerSummary=")
                .append("updateCount=").append(traces.size())
                .append(" nativeCount=").append(nativeCount)
                .append(" arrayCount=").append(arrayCount)
                .append(" metalCount=").append(metalCount)
                .append(" elementCount=").append(elements)
                .append('\n');
        for (graph.execution.trace.NativeOptimizerTrace trace : traces) {
            sb.append("  optimizerUpdate=")
                    .append("optimizer=").append(trace.optimizer())
                    .append(" route=").append(trace.route())
                    .append(" dtype=").append(trace.dataType())
                    .append(" parameterNodeId=").append(trace.parameterNodeId())
                    .append(" gradientNodeId=").append(trace.gradientNodeId())
                    .append(" elementCount=").append(trace.elementCount())
                    .append(" publicationPolicy=").append(trace.publicationPolicy())
                    .append(" gradientPublication=").append(trace.gradientPublication())
                    .append(" optimizerStateStorage=").append(trace.optimizerStateStorage())
                    .append(" bf16TrainingPolicy=").append(trace.bf16TrainingPolicy())
                    .append(" nativeCpuFailurePolicy=").append(trace.nativeCpuFailurePolicy())
                    .append(" parameterResidencyBefore=").append(trace.parameterResidencyBefore())
                    .append(" parameterResidencyAfter=").append(trace.parameterResidencyAfter())
                    .append(" gradientResidencyBefore=").append(trace.gradientResidencyBefore())
                    .append(" gradientResidencyAfter=").append(trace.gradientResidencyAfter())
                    .append(" publicationSkippedReason=").append(trace.publicationSkippedReason())
                    .append(" fallbackReason=").append(trace.fallbackReason())
                    .append('\n');
        }
    }

    private static String formatMs(long durationNs) {
        return String.format(Locale.US, "%.6f", durationNs / 1_000_000.0d);
    }

    private static double nanosToMs(long durationNs) {
        return durationNs / 1_000_000.0d;
    }

    private record NativeCpuTraceSummary(
            int nativeKernelCount,
            int arrayKernelCount,
            int fallbackCount,
            boolean present
    ) {
        static NativeCpuTraceSummary fromSteps(java.util.List<graph.execution.trace.ExecutionStepTrace> steps) {
            if (steps == null || steps.isEmpty()) {
                return new NativeCpuTraceSummary(0, 0, 0, false);
            }
            int nativeKernels = 0;
            int arrayKernels = 0;
            int fallbacks = 0;
            boolean sawNativeAttrs = false;
            for (var step : steps) {
                if (step == null || step.metadata() == null || step.metadata().attributes() == null) {
                    continue;
                }
                Map<String, Object> attrs = step.metadata().attributes();
                if (!attrs.containsKey("nativeCpuKernelStatus")
                        && !attrs.containsKey("requestedCpuStorage")
                        && !attrs.containsKey("actualCpuStorage")
                        && !attrs.containsKey("nativeCpuFallbackReason")) {
                    continue;
                }
                sawNativeAttrs = true;
                String fallbackReason = String.valueOf(attrs.getOrDefault("nativeCpuFallbackReason", ""));
                if (!fallbackReason.isBlank()) {
                    fallbacks++;
                }
                if (attrs.containsKey("nativeCpuKernelStatus") && fallbackReason.isBlank()) {
                    nativeKernels++;
                }
                if ("CPU_ARRAY".equals(String.valueOf(attrs.getOrDefault("actualCpuStorage", "")))) {
                    arrayKernels++;
                }
            }
            return new NativeCpuTraceSummary(nativeKernels, arrayKernels, fallbacks, sawNativeAttrs);
        }
    }

    private record NativeCpuRegionTraceSummary(
            int selectedRegionCount,
            int rejectedRegionCount,
            int nativeRouteCount,
            int fallbackCount,
            int measuredWinClaimCount,
            int measuredWinProofCount,
            int providerNodeCount,
            int localKernelNodeCount,
            int segmentScalarNodeCount,
            int stridedNodeCount,
            int stridedMaterializationCount,
            java.util.Map<String, Integer> benchmarkRowCounts,
            java.util.Map<String, Integer> layoutClassCounts,
            java.util.Map<String, Integer> regionResultResidencyCounts,
            int regionAutoEligibleNodeCount,
            int boundaryOutputCount,
            java.util.List<String> fallbackReasons,
            java.util.List<String> stridedFallbackReasons,
            java.util.List<String> rejectionReasons,
            boolean present
    ) {
        static NativeCpuRegionTraceSummary fromSteps(java.util.List<graph.execution.trace.ExecutionStepTrace> steps) {
            if (steps == null || steps.isEmpty()) {
                return empty(false);
            }
            int selected = 0;
            int rejected = 0;
            int nativeRoutes = 0;
            int fallbacks = 0;
            int measuredWinClaims = 0;
            int measuredWinProofs = 0;
            int providers = 0;
            int localKernels = 0;
            int segmentScalarNodes = 0;
            int stridedNodes = 0;
            int stridedMaterializations = 0;
            int boundaries = 0;
            java.util.LinkedHashMap<String, Integer> benchmarkRowCounts = new java.util.LinkedHashMap<>();
            java.util.LinkedHashMap<String, Integer> layoutClassCounts = new java.util.LinkedHashMap<>();
            java.util.LinkedHashMap<String, Integer> regionResultResidencyCounts = new java.util.LinkedHashMap<>();
            int regionAutoEligibleNodes = 0;
            java.util.LinkedHashSet<String> fallbackReasons = new java.util.LinkedHashSet<>();
            java.util.LinkedHashSet<String> stridedFallbackReasons = new java.util.LinkedHashSet<>();
            java.util.LinkedHashSet<String> rejectionReasons = new java.util.LinkedHashSet<>();
            boolean present = false;
            for (var step : steps) {
                if (step == null || step.metadata() == null || step.metadata().attributes() == null) {
                    continue;
                }
                Map<String, Object> attrs = step.metadata().attributes();
                if (!attrs.containsKey("nativeCpuRegionDecision")) {
                    continue;
                }
                present = true;
                String decision = String.valueOf(attrs.getOrDefault("nativeCpuRegionDecision", ""));
                String route = String.valueOf(attrs.getOrDefault("nativeCpuRegionRoute", ""));
                String reason = String.valueOf(attrs.getOrDefault("nativeCpuRegionReason", ""));
                String fallbackReason = String.valueOf(attrs.getOrDefault("nativeCpuRegionFallbackReason", ""));
                if ("SELECTED".equals(decision)) {
                    selected++;
                } else if ("REJECTED".equals(decision)) {
                    rejected++;
                    if (!reason.isBlank()) {
                        rejectionReasons.add(reason);
                    }
                }
                if ("NATIVE".equals(route)) {
                    nativeRoutes++;
                }
                if (!fallbackReason.isBlank()) {
                    fallbacks++;
                    fallbackReasons.add(fallbackReason);
                }
                if (NativeCpuRegionMeasuredWinEvidence.claimed(attrs)) {
                    measuredWinClaims++;
                    if (NativeCpuRegionMeasuredWinEvidence.proven(attrs)) {
                        measuredWinProofs++;
                    }
                }
                providers += listSize(attrs.get("nativeCpuRegionProviderNodes"));
                localKernels += listSize(attrs.get("nativeCpuRegionLocalKernelNodes"));
                segmentScalarNodes += nativeCpuRegionSegmentScalarNodeCount(attrs);
                stridedNodes += intAttr(attrs.get("nativeCpuStridedNodeCount"));
                stridedMaterializations += intAttr(attrs.get("nativeCpuStridedMaterializationCount"));
                mergeStringCounts(layoutClassCounts, attrs.get("nativeCpuLayoutClassCounts"));
                mergeNestedStringCounts(regionResultResidencyCounts, attrs.get("nativeCpuRegionResultResidencies"));
                regionAutoEligibleNodes += trueCount(attrs.get("nativeCpuRegionAutoEligible"));
                addStrings(stridedFallbackReasons, attrs.get("nativeCpuStridedFallbackReasons"));
                boundaries += listSize(attrs.get("nativeCpuRegionOutputs"));
                increment(benchmarkRowCounts, nativeCpuRegionBenchmarkRow(attrs));
            }
            return new NativeCpuRegionTraceSummary(
                    selected,
                    rejected,
                    nativeRoutes,
                    fallbacks,
                    measuredWinClaims,
                    measuredWinProofs,
                    providers,
                    localKernels,
                    segmentScalarNodes,
                    stridedNodes,
                    stridedMaterializations,
                    orderedMap(benchmarkRowCounts),
                    orderedMap(layoutClassCounts),
                    orderedMap(regionResultResidencyCounts),
                    regionAutoEligibleNodes,
                    boundaries,
                    java.util.List.copyOf(fallbackReasons),
                    java.util.List.copyOf(stridedFallbackReasons),
                    java.util.List.copyOf(rejectionReasons),
                    present
            );
        }

        private static NativeCpuRegionTraceSummary empty(boolean present) {
            return new NativeCpuRegionTraceSummary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    java.util.Map.of(), java.util.Map.of(), java.util.Map.of(), 0,
                    0, java.util.List.of(), java.util.List.of(), java.util.List.of(), present);
        }
    }

    private record RuntimeCopyTraceSummary(
            long cpuMaterializationBytes,
            long cpuMaterializationDurationNs,
            long matMulCopyInBytes,
            long matMulCopyOutBytes,
            long matMulNativeTempBytes,
            boolean present
    ) {
        static RuntimeCopyTraceSummary fromRun(graph.execution.trace.RunTrace run) {
            if (run == null) {
                return new RuntimeCopyTraceSummary(0L, 0L, 0L, 0L, 0L, false);
            }
            long materializationBytes = 0L;
            long materializationDurationNs = 0L;
            boolean sawEvidence = false;
            for (var materialization : run.cpuMaterializations()) {
                if (materialization == null) {
                    continue;
                }
                sawEvidence = true;
                materializationBytes += materialization.bytes();
                materializationDurationNs += materialization.durationNs();
            }

            long copyInBytes = 0L;
            long copyOutBytes = 0L;
            long nativeTempBytes = 0L;
            for (var step : run.steps()) {
                if (step == null || step.metadata() == null || step.metadata().matMul() == null) {
                    continue;
                }
                var matMul = step.metadata().matMul();
                if (matMul.copyInBytes() >= 0L) {
                    sawEvidence = true;
                    copyInBytes += matMul.copyInBytes();
                }
                if (matMul.copyOutBytes() >= 0L) {
                    sawEvidence = true;
                    copyOutBytes += matMul.copyOutBytes();
                }
                if (matMul.nativeTempBytes() >= 0L) {
                    sawEvidence = true;
                    nativeTempBytes += matMul.nativeTempBytes();
                }
            }
            return new RuntimeCopyTraceSummary(
                    materializationBytes,
                    materializationDurationNs,
                    copyInBytes,
                    copyOutBytes,
                    nativeTempBytes,
                    sawEvidence
            );
        }
    }

    private record HostDeviceTransferSummary(
            int transferCount,
            long bytes,
            long javaArrayBytes,
            long nativeBytes,
            long deviceBytes,
            int fallbackCount,
            boolean present
    ) {
        static HostDeviceTransferSummary fromRun(graph.execution.trace.RunTrace run) {
            if (run == null || run.hostDeviceTransfers().isEmpty()) {
                return new HostDeviceTransferSummary(0, 0L, 0L, 0L, 0L, 0, false);
            }
            int fallbackCount = 0;
            long bytes = 0L;
            long javaArrayBytes = 0L;
            long nativeBytes = 0L;
            long deviceBytes = 0L;
            for (var transfer : run.hostDeviceTransfers()) {
                if (transfer == null) {
                    continue;
                }
                bytes += transfer.bytes();
                javaArrayBytes += transfer.javaArrayBytes();
                nativeBytes += transfer.nativeBytes();
                deviceBytes += transfer.deviceBytes();
                if (!transfer.fallbackReason().isBlank()) {
                    fallbackCount++;
                }
            }
            return new HostDeviceTransferSummary(
                    run.hostDeviceTransfers().size(),
                    bytes,
                    javaArrayBytes,
                    nativeBytes,
                    deviceBytes,
                    fallbackCount,
                    true
            );
        }
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

    private static void appendHostDeviceTransfers(
            StringBuilder sb,
            java.util.List<graph.execution.trace.HostDeviceTransferTrace> transfers
    ) {
        if (transfers == null || transfers.isEmpty()) {
            return;
        }
        sb.append("  hostDeviceTransfers:\n");
        for (var transfer : transfers) {
            sb.append("    - nodeId=").append(transfer.nodeId())
                    .append(" backend=").append(transfer.backend())
                    .append(" kind=").append(transfer.transferKind())
                    .append(" source=").append(transfer.sourceResidency())
                    .append(" target=").append(transfer.targetResidency())
                    .append(" bytes=").append(transfer.bytes())
                    .append(" javaArrayBytes=").append(transfer.javaArrayBytes())
                    .append(" nativeBytes=").append(transfer.nativeBytes())
                    .append(" deviceBytes=").append(transfer.deviceBytes())
                    .append(" direct=").append(transfer.directTransferSupported())
                    .append(" success=").append(transfer.success());
            if (!transfer.fallbackReason().isBlank()) {
                sb.append(" fallbackReason=").append(transfer.fallbackReason());
            }
            if (!transfer.detail().isBlank()) {
                sb.append(" detail=").append(transfer.detail());
            }
            sb.append('\n');
        }
    }

    private static void appendAcceleratorSummary(StringBuilder sb, AcceleratorTraceSummary summary) {
        if (summary == null || !summary.present()) {
            return;
        }
        sb.append("  accelerator:\n");
        for (var entry : summary.backends().entrySet()) {
            var backend = entry.getValue();
            sb.append("    - backend=").append(entry.getKey())
                    .append(" steps=").append(backend.steps())
                    .append(" bufferBindingSteps=").append(backend.bufferBindingSteps())
                    .append(" tensorArraySteps=").append(backend.tensorArraySteps())
                    .append(" cpuFallbackSteps=").append(backend.cpuFallbackSteps())
                    .append(" unavailableSteps=").append(backend.unavailableSteps())
                    .append(" preparedInputSteps=").append(backend.preparedInputSteps())
                    .append(" reasonCodes=").append(backend.reasonCodes())
                    .append(" fallbackReasons=").append(backend.fallbackReasons())
                    .append(" executionRouteCounts=").append(backend.executionRouteCounts())
                    .append(" rejectedRouteReasonCounts=").append(backend.rejectedRouteReasonCounts())
                    .append(" bytes=").append(backend.inputBytes()).append("->").append(backend.outputBytes())
                    .append(" nativeCopyStrategies=").append(backend.nativeCopyStrategies())
                    .append(" outputBufferWriteStatuses=").append(backend.outputBufferWriteStatuses())
                    .append(" javaToNativeMs=").append(formatNsAttr(backend.javaToNativeCopyNs()))
                    .append(" nativeToJavaMs=").append(formatNsAttr(backend.nativeToJavaCopyNs()))
                    .append(" nativeDeviceCopyMs=").append(formatNsAttr(backend.nativeDeviceCopyNs()))
                    .append('\n');
        }
    }

    private static void appendGpuCoverageSummary(StringBuilder sb, GpuCoverageSummary summary) {
        if (summary == null || !summary.present()) {
            return;
        }
        sb.append("  coverage:\n");
        for (var entry : summary.backends().entrySet()) {
            var backend = entry.getValue();
            sb.append("    - backend=").append(entry.getKey())
                    .append(" totalStepCount=").append(backend.totalStepCount())
                    .append(" acceleratorStepCount=").append(backend.acceleratorStepCount())
                    .append(" gpuCoverageRatio=").append(String.format(Locale.US, "%.6f", backend.gpuCoverageRatio()))
                    .append(" selectedRegionCount=").append(backend.selectedRegionCount())
                    .append(" multiOpGpuRegionCount=").append(backend.multiOpGpuRegionCount())
                    .append(" maxSelectedRegionLength=").append(backend.maxSelectedRegionLength())
                    .append(" averageSelectedRegionLength=")
                    .append(String.format(Locale.US, "%.6f", backend.averageSelectedRegionLength()))
                    .append(" loweredPrimitiveCount=").append(backend.loweredPrimitiveCount())
                    .append(" rejectedCandidateCount=").append(backend.rejectedCandidateCount())
                    .append(" rejectedCandidateReasonCounts=").append(backend.rejectedCandidateReasonCounts())
                    .append(" bufferBindingStepCount=").append(backend.bufferBindingStepCount())
                    .append(" nativeBufferStepCount=").append(backend.nativeBufferStepCount())
                    .append(" tensorArrayStepCount=").append(backend.tensorArrayStepCount())
                    .append(" cpuFallbackStepCount=").append(backend.cpuFallbackStepCount())
                    .append(" fallbackCount=").append(backend.fallbackCount())
                    .append(" cpuMaterializationCount=").append(backend.cpuMaterializationCount())
                    .append(" internalCpuMaterializationCount=").append(backend.internalCpuMaterializationCount())
                    .append(" gradientPublicationMaterializationCount=")
                    .append(backend.gradientPublicationMaterializationCount())
                    .append(" cpuMaterializationReasonCounts=").append(backend.cpuMaterializationReasonCounts())
                    .append(" cpuMaterializationBytes=").append(backend.cpuMaterializationBytes())
                    .append(" cpuMaterializationDurationNs=").append(backend.cpuMaterializationDurationNs())
                    .append(" copyDurationNs=").append(backend.copyDurationNs())
                    .append(" nativeCopyStrategyCounts=").append(backend.nativeCopyStrategyCounts())
                    .append(" executionRouteCounts=").append(backend.executionRouteCounts())
                    .append(" rejectedRouteReasonCounts=").append(backend.rejectedRouteReasonCounts())
                    .append(" deviceHandoffCount=").append(backend.deviceHandoffCount())
                    .append(" gpuLayoutMaterializationCount=").append(backend.gpuLayoutMaterializationCount())
                    .append(" gpuLayoutMaterializationBytes=").append(backend.gpuLayoutMaterializationBytes())
                    .append(" gpuLayoutTransformKindCounts=").append(backend.gpuLayoutTransformKindCounts())
                    .append(" gpuLayoutTargetLayoutClassCounts=").append(backend.gpuLayoutTargetLayoutClassCounts())
                    .append(" storageResidencyCounts=").append(backend.storageResidencyCounts())
                    .append(" dtypeResidencyReasons=").append(backend.dtypeResidencyReasons())
                    .append(" gpuFusedSubpatternCount=").append(backend.gpuFusedSubpatternCount())
                    .append(" gpuFusedSubpatternTypes=").append(backend.gpuFusedSubpatternTypes())
                    .append(" gpuFusedSubpatternOriginalNodeIds=").append(backend.gpuFusedSubpatternOriginalNodeIds())
                    .append(" gpuFusedSubpatternLoweredPrimitiveCount=").append(backend.gpuFusedSubpatternLoweredPrimitiveCount())
                    .append(" gpuFusedSubpatternReasons=").append(backend.gpuFusedSubpatternReasons())
                    .append(" reasonCodes=").append(backend.reasonCodes())
                    .append(" fallbackReasons=").append(backend.fallbackReasons())
                    .append('\n');
            if (!backend.dtypeResidencyReasons().isEmpty()) {
                sb.append("      DType Residency Evidence\n");
                backend.dtypeResidencyReasons().forEach((reason, count) ->
                        sb.append("        - count=").append(count).append(" ").append(reason).append('\n'));
            }
            GpuCoverageGateResult gate = GpuCoverageRegressionGate.evaluate(
                    summary,
                    GpuCoverageGatePolicy.reportNativeBufferTarget(entry.getKey(), backend)
            );
            sb.append("      coverageGate backend=").append(entry.getKey())
                    .append(" gatePassed=").append(gate.passed())
                    .append(" gateFailures=").append(gate.failures())
                    .append('\n');
            GpuCoverageNativeEvidence nativeEvidence = nativeEvidence(entry.getKey(), backend);
            sb.append("      nativeEvidence backend=").append(nativeEvidence.backend())
                    .append(" nativeStatus=").append(nativeEvidence.nativeStatus())
                    .append(" detail=").append(nativeEvidence.detail())
                    .append(" capabilitySkipped=").append("capabilitySkipped".equals(nativeEvidence.nativeStatus()))
                    .append('\n');
            appendTargetCoverageTruth(sb, entry.getKey());
        }
    }

    private static void appendCrossBackendRouterEvidence(StringBuilder sb, CrossBackendRouterEvidence evidence) {
        if (evidence == null || !evidence.present()) {
            return;
        }
        sb.append("  routerEvidence:\n");
        for (var entry : evidence.backends().entrySet()) {
            var backend = entry.getValue();
            sb.append("    - backend=").append(entry.getKey())
                    .append(" acceleratorPathCounts=").append(backend.acceleratorPathCounts())
                    .append(" backendRouteCounts=").append(backend.backendRouteCounts())
                    .append(" rejectedRouteReasonCounts=").append(backend.rejectedRouteReasonCounts())
                    .append(" reasonCodeCounts=").append(backend.reasonCodeCounts())
                    .append(" fallbackReasonCounts=").append(backend.fallbackReasonCounts())
                    .append(" nativeCopyStrategyCounts=").append(backend.nativeCopyStrategyCounts())
                    .append(" outputBufferWriteStatusCounts=").append(backend.outputBufferWriteStatusCounts())
                    .append(" selectedRegionCount=").append(backend.selectedRegionCount())
                    .append(" maxSelectedRegionLength=").append(backend.maxSelectedRegionLength())
                    .append(" loweredPrimitiveCount=").append(backend.loweredPrimitiveCount())
                    .append(" gpuFusedSubpatternCount=").append(backend.gpuFusedSubpatternCount())
                    .append(" tensorArrayStepCount=").append(backend.tensorArrayStepCount())
                    .append(" cpuFallbackStepCount=").append(backend.cpuFallbackStepCount())
                    .append(" cpuMaterializationCount=").append(backend.cpuMaterializationCount())
                    .append(" internalCpuMaterializationCount=").append(backend.internalCpuMaterializationCount())
                    .append(" deviceHandoffCount=").append(backend.deviceHandoffCount())
                    .append('\n');
        }
    }

    private static void appendTargetCoverageTruth(StringBuilder sb, String backendName) {
        backendFromName(backendName).ifPresent(backend -> {
            sb.append("      targetCoverageTruth:\n");
            for (GpuTargetCoverageTruth.Row row : GpuTargetCoverageTruth.rowsFor(backend)) {
                sb.append("        - op=").append(row.opType())
                        .append(" family=").append(row.family())
                        .append(" matrixStatus=").append(row.matrixStatus())
                        .append(" executionStatus=").append(row.executionStatus())
                        .append(" detail=").append(row.detail())
                        .append('\n');
            }
        });
    }

    private static java.util.Optional<backend.contract.ComputeBackend> backendFromName(String backendName) {
        if ("GPU_METAL".equals(backendName)) {
            return java.util.Optional.of(backend.contract.ComputeBackend.GPU_METAL);
        }
        if ("GPU_CUDA".equals(backendName)) {
            return java.util.Optional.of(backend.contract.ComputeBackend.GPU_CUDA);
        }
        return java.util.Optional.empty();
    }

    static GpuCoverageNativeEvidence nativeEvidence(
            String backend,
            GpuCoverageSummary.BackendCoverage coverage
    ) {
        if (coverage != null && coverage.nativeBufferStepCount() > 0) {
            return GpuCoverageNativeEvidence.passed(backend, "native buffer binding observed");
        }
        return GpuCoverageNativeEvidence.capabilitySkipped(backend, "native buffer evidence not available in this report");
    }

    private static void appendBackendSelectionCost(
            StringBuilder sb,
            graph.execution.trace.BackendSelectionTrace trace
    ) {
        if (trace == null || trace.decisions().isEmpty()) {
            return;
        }
        java.util.List<graph.execution.trace.BackendSelectionDecisionTrace> selected = trace.decisions().stream()
                .filter(decision -> decision.selected() && decision.costSummary() != null)
                .toList();
        java.util.List<graph.execution.trace.PartitionDecisionTrace.CandidateCostTrace> finalists =
                rejectedFinalists(trace);
        if (selected.isEmpty() && finalists.isEmpty()) {
            return;
        }
        sb.append("  backendSelectionCost:\n");
        for (var decision : selected) {
            var summary = decision.costSummary();
            sb.append("    selectedBackend=").append(decision.selectedBackend())
                    .append(" nodeIds=").append(decision.nodeIds())
                    .append(" preset=").append(summary.preset())
                    .append(" finalScore=").append(formatScore(summary.finalScore()))
                    .append(" boundaryCount=").append(summary.boundaryCount())
                    .append(" estimatedTransferBytes=").append(summary.estimatedTransferBytes())
                    .append(" layoutFallbackBytes=").append(summary.layoutFallbackBytes())
                    .append(" estimatedComputeWork=").append(summary.estimatedComputeWork())
                    .append(" reason=").append(decision.reason())
                    .append('\n');
            sb.append("      ")
                    .append(CostExplanationTextRenderer.renderCompact(summary.toCostScore().explain(summary.reasonCode())))
                    .append('\n');
            if (decision.gpuLoweredRegionManifest() != null) {
                appendIndentedBlock(
                        sb,
                        GpuLoweredRegionManifestRenderer.renderCompact(decision.gpuLoweredRegionManifest()),
                        "      "
                );
            }
        }
        if (!finalists.isEmpty()) {
            sb.append("    rejectedFinalists:\n");
            for (var finalist : finalists) {
                sb.append("      - nodeIds=").append(finalist.nodeIds())
                        .append(" preset=").append(finalist.preset())
                        .append(" finalScore=").append(formatScore(finalist.finalScore()))
                        .append(" boundaryCount=").append(finalist.boundaryCount())
                        .append(" estimatedTransferBytes=").append(finalist.estimatedTransferBytes())
                        .append(" layoutFallbackBytes=").append(finalist.layoutFallbackBytes())
                        .append(" estimatedComputeWork=").append(finalist.estimatedComputeWork())
                        .append(" reason=").append(finalist.reason())
                        .append('\n');
                sb.append("        ")
                        .append(CostExplanationTextRenderer.renderCompact(finalist.toCostScore().explain(finalist.reason())))
                        .append('\n');
            }
        }
    }

    private static void appendOptimizerCost(
            StringBuilder sb,
            graph.optimizer.state.OptimizerTrace trace
    ) {
        if (trace == null || trace.costExplanations().isEmpty()) {
            return;
        }
        sb.append("  optimizerCost:\n");
        for (var explanation : trace.costExplanations()) {
            sb.append("    ")
                    .append(CostExplanationTextRenderer.renderCompact(explanation))
                    .append('\n');
        }
    }

    private static void appendIndentedBlock(StringBuilder sb, String block, String indent) {
        if (block == null || block.isBlank()) {
            return;
        }
        for (String line : block.split("\\R")) {
            if (!line.isEmpty()) {
                sb.append(indent).append(line).append('\n');
            }
        }
    }

    private static java.util.List<graph.execution.trace.PartitionDecisionTrace.CandidateCostTrace> rejectedFinalists(
            graph.execution.trace.BackendSelectionTrace trace
    ) {
        java.util.List<graph.execution.trace.PartitionDecisionTrace.CandidateCostTrace> out = new java.util.ArrayList<>();
        trace.decisions().stream()
                .flatMap(decision -> decision.finalists().stream())
                .forEach(out::add);
        for (var decision : trace.decisions()) {
            if (decision.selected() || decision.costSummary() == null) {
                continue;
            }
            var summary = decision.costSummary();
            out.add(new graph.execution.trace.PartitionDecisionTrace.CandidateCostTrace(
                    decision.nodeIds(),
                    decision.reason(),
                    summary.finalScore(),
                    summary.boundaryCount(),
                    summary.estimatedTransferBytes(),
                    summary.layoutFallbackBytes(),
                    summary.estimatedComputeWork(),
                    summary.preset()
            ));
        }
        return out.stream().limit(3).toList();
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
                    appendMetalRouteCostSummary(sb, step);
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
                appendMetalRouteCostLine(sb, metadata.attributes(), "      ");
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
                        .append(" blasProvider=").append(metadata.matMul().blasProvider())
                        .append(" blasSymbol=").append(metadata.matMul().blasSymbol())
                        .append(" blasRoute=").append(metadata.matMul().blasRoute())
                        .append(" route=").append(metadata.matMul().route())
                        .append(" cpuStorageProfile=").append(metadata.matMul().cpuStorageProfile())
                        .append(" nativeCpuFailurePolicy=").append(metadata.matMul().nativeCpuFailurePolicy())
                        .append(" requestedCpuStorage=").append(metadata.matMul().requestedCpuStorage())
                        .append(" actualCpuStorage=").append(metadata.matMul().actualCpuStorage())
                        .append(" nativeCpuFallbackReason=").append(metadata.matMul().nativeCpuFallbackReason())
                        .append(" openblasSgemmAvailable=").append(metadata.matMul().openblasSgemmAvailable())
                        .append(" openblasDgemmAvailable=").append(metadata.matMul().openblasDgemmAvailable())
                        .append(" openblasSbgemmAvailable=").append(metadata.matMul().openblasSbgemmAvailable())
                        .append(" openblasBgemmAvailable=").append(metadata.matMul().openblasBgemmAvailable())
                        .append(" bf16ContinuationRoute=").append(metadata.matMul().bf16ContinuationRoute())
                        .append(" bf16OutputRoute=").append(metadata.matMul().bf16OutputRoute())
                        .append(" bf16ComputePrecision=").append(metadata.matMul().bf16ComputePrecision())
                        .append(" bf16OutputPrecision=").append(metadata.matMul().bf16OutputPrecision())
                        .append(" copyInBytes=").append(metadata.matMul().copyInBytes())
                        .append(" copyOutBytes=").append(metadata.matMul().copyOutBytes())
                        .append(" nativeTempBytes=").append(metadata.matMul().nativeTempBytes())
                        .append(" threadPolicy=").append(metadata.matMul().threadPolicy())
                        .append(" fallbackReason=").append(metadata.matMul().fallbackReason())
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
                        .append("numericContract=").append(metadata.fused().numericContract())
                        .append(" lowCostHint=").append(metadata.fused().lowCostHint())
                        .append(" dispatchFamily=").append(metadata.fused().dispatchFamily())
                        .append(" schedulerSignature=").append(metadata.fused().schedulerSignature())
                        .append(" vectorFallbackReason=").append(metadata.fused().vectorFallbackReason())
                        .append(" executionBackend=").append(metadata.fused().executionBackend())
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

    private static String formatScore(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.6f", value) : "null";
    }

    private static String formatCompilePolicy(BenchmarkCandidateReport candidate) {
        if (candidate == null || candidate.entry() == null || candidate.entry().profile() == null) {
            return "{}";
        }
        var compile = candidate.entry().profile().compile();
        return "{graphStages=" + graphStages(compile.graphOptimization())
                + ", backendDiscovery=" + compile.backendPlanning().discoveryMode()
                + ", backendTargets=" + compile.backendPlanning().targets()
                + ", ownershipPlanner=" + compile.backendPlanning().ownershipPlanner()
                + ", regionOptimization=" + compile.regionOptimization().enabled()
                + ", memoryPlanning=" + compile.memoryPlanning().enabled()
                + "}";
    }

    private static java.util.List<String> graphStages(config.compile.GraphOptimizationConfig graph) {
        java.util.List<String> stages = new java.util.ArrayList<>();
        if (graph.algebraicRewrite()) {
            stages.add("AR");
        }
        if (graph.constantFolding()) {
            stages.add("CF");
        }
        if (graph.commonSubexpressionElimination()) {
            stages.add("CSE");
        }
        if (graph.deadCodeElimination()) {
            stages.add("DCE");
        }
        if (graph.optionalLowering()) {
            stages.add("LOWER");
        }
        return stages;
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

    private static int listSize(Object value) {
        if (value instanceof java.util.Collection<?> collection) {
            return collection.size();
        }
        return 0;
    }

    private static int intAttr(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value instanceof String text) {
            try {
                return Integer.parseInt(text);
            } catch (NumberFormatException ignored) {
                return 0;
            }
        }
        return 0;
    }

    private static void addStrings(java.util.LinkedHashSet<String> out, Object value) {
        if (out == null || !(value instanceof java.util.Collection<?> collection)) {
            return;
        }
        for (Object item : collection) {
            String text = String.valueOf(item);
            if (!text.isBlank()) {
                out.add(text);
            }
        }
    }

    private static void mergeStringCounts(java.util.LinkedHashMap<String, Integer> out, Object value) {
        if (out == null || !(value instanceof Map<?, ?> map)) {
            return;
        }
        for (var entry : map.entrySet()) {
            String key = String.valueOf(entry.getKey());
            int count = intAttr(entry.getValue());
            if (!key.isBlank() && count > 0) {
                out.merge(key, count, Integer::sum);
            }
        }
    }

    private static void increment(java.util.LinkedHashMap<String, Integer> out, String key) {
        if (out != null && key != null && !key.isBlank()) {
            out.merge(key, 1, Integer::sum);
        }
    }

    private static Map<String, Integer> orderedMap(java.util.LinkedHashMap<String, Integer> values) {
        if (values == null || values.isEmpty()) {
            return java.util.Map.of();
        }
        return java.util.Collections.unmodifiableMap(new java.util.LinkedHashMap<>(values));
    }

    private static void mergeNestedStringCounts(java.util.LinkedHashMap<String, Integer> out, Object value) {
        if (out == null || !(value instanceof java.util.Collection<?> outer)) {
            return;
        }
        for (Object item : outer) {
            if (item instanceof java.util.Collection<?> inner) {
                for (Object nested : inner) {
                    String key = String.valueOf(nested);
                    if (!key.isBlank()) {
                        out.merge(key, 1, Integer::sum);
                    }
                }
            } else {
                String key = String.valueOf(item);
                if (!key.isBlank()) {
                    out.merge(key, 1, Integer::sum);
                }
            }
        }
    }

    private static int trueCount(Object value) {
        if (!(value instanceof java.util.Collection<?> collection)) {
            return 0;
        }
        int count = 0;
        for (Object item : collection) {
            if (Boolean.TRUE.equals(item) || "true".equalsIgnoreCase(String.valueOf(item))) {
                count++;
            }
        }
        return count;
    }

    private static int nativeCpuRegionSegmentScalarNodeCount(Map<String, Object> attrs) {
        int explicitCount = listSize(attrs.get("nativeCpuRegionSegmentScalarNodes"));
        if (explicitCount > 0) {
            return explicitCount;
        }
        Object kernels = attrs.get("nativeCpuRegionPhysicalKernels");
        int count = 0;
        if (kernels instanceof java.util.Collection<?> collection) {
            for (Object kernel : collection) {
                String family = String.valueOf(kernel);
                if ("SEGMENT_SCALAR".equals(family)
                        || "SEGMENT_DENSE_SCALAR".equals(family)
                        || "SEGMENT_STRIDED_SCALAR".equals(family)) {
                    count++;
                }
            }
        }
        if (count > 0) {
            return count;
        }
        Object families = attrs.get("nativeCpuRegionSegmentKernelFamilies");
        if (families instanceof java.util.Collection<?> familyCollection) {
            for (Object family : familyCollection) {
                String value = String.valueOf(family);
                if ("SEGMENT_DENSE_SCALAR".equals(value) || "SEGMENT_STRIDED_SCALAR".equals(value)) {
                    count++;
                }
            }
        }
        return count;
    }

    private static String nativeCpuRegionBenchmarkRow(Map<String, Object> attrs) {
        String route = String.valueOf(attrs.getOrDefault("nativeCpuRegionRoute", ""));
        int providerNodes = listSize(attrs.get("nativeCpuRegionProviderNodes"));
        int localKernelNodes = listSize(attrs.get("nativeCpuRegionLocalKernelNodes"));
        int segmentScalarNodes = nativeCpuRegionSegmentScalarNodeCount(attrs);
        int stridedNodes = intAttr(attrs.get("nativeCpuStridedNodeCount"));
        if ("NATIVE".equals(route)) {
            if (providerNodes > 0 && localKernelNodes > 0) {
                return "native provider + local native";
            }
            if (providerNodes > 0) {
                return "native provider only";
            }
            if (segmentScalarNodes > 0) {
                return "native segment scalar";
            }
            if (containsKernelFamily(attrs, "SEGMENT_PARALLEL")
                    || containsKernelFamily(attrs, "NATIVE_SEGMENT_PARALLEL")) {
                return "native segment parallel";
            }
            return "native segment fused";
        }
        if (providerNodes > 0) {
            return "native provider + array fallback";
        }
        if (stridedNodes > 0) {
            return "array strided";
        }
        return "array dense";
    }

    private static boolean containsKernelFamily(Map<String, Object> attrs, String expected) {
        if (attrs == null || expected == null || expected.isBlank()) {
            return false;
        }
        return containsValue(attrs.get("nativeCpuRegionSegmentKernelFamilies"), expected)
                || containsValue(attrs.get("nativeCpuRegionPhysicalKernels"), expected);
    }

    private static boolean containsValue(Object value, String expected) {
        if (!(value instanceof java.util.Collection<?> collection)) {
            return false;
        }
        for (Object item : collection) {
            if (expected.equals(String.valueOf(item))) {
                return true;
            }
        }
        return false;
    }

    private static void appendMetalRouteCostSummary(StringBuilder sb, graph.execution.trace.ExecutionStepTrace step) {
        if (step == null || step.metadata() == null || step.metadata().attributes() == null) {
            return;
        }
        Map<String, Object> attrs = step.metadata().attributes();
        Object model = attrs.get("metalRouteCostModel");
        if (model == null || String.valueOf(model).isBlank()) {
            return;
        }
        sb.append(" metalRouteCost=")
                .append(model)
                .append("/")
                .append(attrs.getOrDefault("metalRouteCostReason", ""))
                .append(" top=")
                .append(attrs.getOrDefault("metalRouteCostTopContributors", java.util.List.of()));
    }

    private static void appendMetalRouteCostLine(StringBuilder sb, Map<String, Object> attrs, String indent) {
        if (attrs == null || !attrs.containsKey("metalRouteCostModel")) {
            return;
        }
        sb.append(indent)
                .append("metalRouteCost: model=")
                .append(attrs.get("metalRouteCostModel"))
                .append(" input=")
                .append(attrs.getOrDefault("metalRouteCostInputKind", ""))
                .append(" reason=")
                .append(attrs.getOrDefault("metalRouteCostReason", ""))
                .append(" comparison=")
                .append(attrs.getOrDefault("metalRouteCostComparison", ""))
                .append(" top=")
                .append(attrs.getOrDefault("metalRouteCostTopContributors", java.util.List.of()))
                .append('\n');
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
