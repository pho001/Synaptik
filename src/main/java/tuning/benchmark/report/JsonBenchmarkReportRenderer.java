package tuning.benchmark.report;

import backend.accelerator.lowering.GpuFusionSubpatternSummary;
import backend.accelerator.lowering.GpuLoweredPrimitiveManifest;
import backend.accelerator.lowering.GpuLoweredRegionManifest;
import backend.accelerator.lowering.GpuLoweredRegionOriginalOp;
import backend.accelerator.lowering.GpuLoweredRegionRejection;
import backend.accelerator.lowering.GpuLoweredRegionValueAssumption;

import java.util.Locale;
import java.util.Map;

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
            sb.append("      \"compile\": ").append(compilePolicyJson(candidate)).append(",\n");
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
                sb.append("        \"cpuMaterializationCount\": ").append(trace.run().cpuMaterializations().size()).append(",\n");
                sb.append("        \"parallelUsed\": ").append(usesParallel(trace.run().steps())).append(",\n");
                sb.append("        \"vectorUsed\": ").append(usesVector(trace.run().steps())).append(",\n");
                sb.append("        \"nativeCpu\": ").append(nativeCpuTraceSummaryJson(
                        NativeCpuTraceSummary.fromSteps(trace.run().steps())
                )).append(",\n");
                sb.append("        \"runtimeCopy\": ").append(runtimeCopyTraceSummaryJson(
                        RuntimeCopyTraceSummary.fromRun(trace.run())
                )).append(",\n");
                sb.append("        \"nativeCpuMemory\": ")
                        .append(nativeCpuMemoryTraceJson(trace.run().nativeCpuMemory()))
                        .append(",\n");
                sb.append("        \"accelerator\": ").append(acceleratorSummaryJson(
                        AcceleratorTraceSummary.fromSteps(trace.run().steps())
                )).append(",\n");
                sb.append("        \"coverage\": ").append(gpuCoverageSummaryJson(
                        GpuCoverageSummary.fromTrace(trace)
                )).append(",\n");
                sb.append("        \"routerEvidence\": ").append(crossBackendRouterEvidenceJson(
                        CrossBackendRouterEvidence.fromTrace(trace)
                )).append(",\n");
                sb.append("        \"backendSelectionCost\": ")
                        .append(backendSelectionCostJson(trace.prepare().backendSelection()))
                        .append(",\n");
                sb.append("        \"optimizerCost\": ")
                        .append(optimizerCostJson(trace.compile().optimizerTrace()))
                        .append(",\n");
                sb.append("        \"cpuMaterializations\": [\n");
                var materializations = trace.run().cpuMaterializations();
                for (int j = 0; j < materializations.size(); j++) {
                    if (j > 0) {
                        sb.append(",\n");
                    }
                    appendCpuMaterializationJson(sb, materializations.get(j), "          ");
                }
                sb.append("\n        ],\n");
                sb.append("        \"steps\": [\n");
                var allSteps = trace.run().steps();
                for (int j = 0; j < allSteps.size(); j++) {
                    var step = allSteps.get(j);
                    if (j > 0) {
                        sb.append(",\n");
                    }
                    appendStepJson(sb, step, "          ");
                }
                sb.append("\n        ],\n");
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
                    appendStepJson(sb, step, "          ");
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

    private static String nativeCpuTraceSummaryJson(NativeCpuTraceSummary summary) {
        if (summary == null || !summary.present()) {
            return "null";
        }
        return "{"
                + "\"nativeKernelCount\": " + summary.nativeKernelCount() + ", "
                + "\"arrayKernelCount\": " + summary.arrayKernelCount() + ", "
                + "\"fallbackCount\": " + summary.fallbackCount()
                + "}";
    }

    private static String runtimeCopyTraceSummaryJson(RuntimeCopyTraceSummary summary) {
        if (summary == null || !summary.present()) {
            return "null";
        }
        return "{"
                + "\"cpuMaterializationBytes\": " + summary.cpuMaterializationBytes() + ", "
                + "\"cpuMaterializationDurationNs\": " + summary.cpuMaterializationDurationNs() + ", "
                + "\"matMulCopyInBytes\": " + summary.matMulCopyInBytes() + ", "
                + "\"matMulCopyOutBytes\": " + summary.matMulCopyOutBytes() + ", "
                + "\"matMulNativeTempBytes\": " + summary.matMulNativeTempBytes()
                + "}";
    }

    private static String nativeCpuMemoryTraceJson(graph.execution.trace.NativeCpuMemoryTrace trace) {
        if (trace == null || !trace.present()) {
            return "null";
        }
        return "{"
                + "\"allocationCount\": " + trace.allocationCount() + ", "
                + "\"releaseCount\": " + trace.releaseCount() + ", "
                + "\"retainCount\": " + trace.retainCount() + ", "
                + "\"allocationFailureCount\": " + trace.allocationFailureCount() + ", "
                + "\"requestedBytes\": " + trace.requestedBytes() + ", "
                + "\"allocatedBytes\": " + trace.allocatedBytes() + ", "
                + "\"currentLiveBytes\": " + trace.currentLiveBytes() + ", "
                + "\"peakLiveBytes\": " + trace.peakLiveBytes() + ", "
                + "\"retainedBytes\": " + trace.retainedBytes()
                + "}";
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

    private static String compilePolicyJson(BenchmarkCandidateReport candidate) {
        if (candidate == null || candidate.entry() == null || candidate.entry().profile() == null) {
            return "{}";
        }
        var compile = candidate.entry().profile().compile();
        var graph = compile.graphOptimization();
        var backend = compile.backendPlanning();
        var region = compile.regionOptimization();
        var memory = compile.memoryPlanning();
        StringBuilder sb = new StringBuilder("{");
        sb.append("\"graphStages\": ").append(graphStagesJson(graph)).append(", ");
        sb.append("\"backendDiscovery\": \"").append(backend.discoveryMode().name()).append("\", ");
        sb.append("\"backendTargets\": ").append(stringSetJson(backend.targets().stream()
                .map(Enum::name)
                .sorted()
                .toList())).append(", ");
        sb.append("\"ownershipPlanner\": \"").append(backend.ownershipPlanner().name()).append("\", ");
        sb.append("\"regionOptimization\": ").append(region.enabled()).append(", ");
        sb.append("\"memoryPlanning\": ").append(memory.enabled());
        sb.append('}');
        return sb.toString();
    }

    private static String graphStagesJson(config.compile.GraphOptimizationConfig graph) {
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
        return stringSetJson(stages);
    }

    private static String stringSetJson(java.util.List<String> values) {
        if (values == null || values.isEmpty()) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append('"').append(escape(values.get(i))).append('"');
        }
        sb.append(']');
        return sb.toString();
    }

    private static String acceleratorSummaryJson(AcceleratorTraceSummary summary) {
        if (summary == null || !summary.present()) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder("{");
        int i = 0;
        for (var entry : summary.backends().entrySet()) {
            if (i++ > 0) {
                sb.append(", ");
            }
            var backend = entry.getValue();
            sb.append('"').append(escape(entry.getKey())).append("\": {")
                    .append("\"steps\": ").append(backend.steps()).append(", ")
                    .append("\"bufferBindingSteps\": ").append(backend.bufferBindingSteps()).append(", ")
                    .append("\"tensorArraySteps\": ").append(backend.tensorArraySteps()).append(", ")
                    .append("\"cpuFallbackSteps\": ").append(backend.cpuFallbackSteps()).append(", ")
                    .append("\"unavailableSteps\": ").append(backend.unavailableSteps()).append(", ")
                    .append("\"preparedInputSteps\": ").append(backend.preparedInputSteps()).append(", ")
                    .append("\"inputBytes\": ").append(backend.inputBytes()).append(", ")
                    .append("\"outputBytes\": ").append(backend.outputBytes()).append(", ")
                    .append("\"javaToNativeCopyNs\": ").append(backend.javaToNativeCopyNs()).append(", ")
                    .append("\"nativeToJavaCopyNs\": ").append(backend.nativeToJavaCopyNs()).append(", ")
                    .append("\"nativeDeviceCopyNs\": ").append(backend.nativeDeviceCopyNs()).append(", ")
                    .append("\"nativeCopyStrategies\": ").append(stringListJson(backend.nativeCopyStrategies())).append(", ")
                    .append("\"outputBufferWriteStatuses\": ").append(stringListJson(backend.outputBufferWriteStatuses())).append(", ")
                    .append("\"executionRouteCounts\": ").append(intMapJson(backend.executionRouteCounts())).append(", ")
                    .append("\"rejectedRouteReasonCounts\": ").append(intMapJson(backend.rejectedRouteReasonCounts())).append(", ")
                    .append("\"reasonCodes\": ").append(stringListJson(backend.reasonCodes())).append(", ")
                    .append("\"fallbackReasons\": ").append(stringListJson(backend.fallbackReasons()))
                    .append("}");
        }
        sb.append('}');
        return sb.toString();
    }

    private static String gpuCoverageSummaryJson(GpuCoverageSummary summary) {
        if (summary == null || !summary.present()) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder("{");
        int i = 0;
        for (var entry : summary.backends().entrySet()) {
            if (i++ > 0) {
                sb.append(", ");
            }
            var backend = entry.getValue();
            GpuCoverageGateResult gate = GpuCoverageRegressionGate.evaluate(
                    summary,
                    GpuCoverageGatePolicy.reportNativeBufferTarget(entry.getKey(), backend)
            );
            GpuCoverageNativeEvidence nativeEvidence = TextBenchmarkReportRenderer.nativeEvidence(entry.getKey(), backend);
            sb.append('"').append(escape(entry.getKey())).append("\": {")
                    .append("\"totalStepCount\": ").append(backend.totalStepCount()).append(", ")
                    .append("\"acceleratorStepCount\": ").append(backend.acceleratorStepCount()).append(", ")
                    .append("\"gpuCoverageRatio\": ").append(format(backend.gpuCoverageRatio())).append(", ")
                    .append("\"selectedRegionCount\": ").append(backend.selectedRegionCount()).append(", ")
                    .append("\"multiOpGpuRegionCount\": ").append(backend.multiOpGpuRegionCount()).append(", ")
                    .append("\"maxSelectedRegionLength\": ").append(backend.maxSelectedRegionLength()).append(", ")
                    .append("\"averageSelectedRegionLength\": ")
                    .append(format(backend.averageSelectedRegionLength())).append(", ")
                    .append("\"loweredPrimitiveCount\": ").append(backend.loweredPrimitiveCount()).append(", ")
                    .append("\"rejectedCandidateCount\": ").append(backend.rejectedCandidateCount()).append(", ")
                    .append("\"rejectedCandidateReasonCounts\": ")
                    .append(intMapJson(backend.rejectedCandidateReasonCounts())).append(", ")
                    .append("\"bufferBindingStepCount\": ").append(backend.bufferBindingStepCount()).append(", ")
                    .append("\"nativeBufferStepCount\": ").append(backend.nativeBufferStepCount()).append(", ")
                    .append("\"tensorArrayStepCount\": ").append(backend.tensorArrayStepCount()).append(", ")
                    .append("\"cpuFallbackStepCount\": ").append(backend.cpuFallbackStepCount()).append(", ")
                    .append("\"fallbackCount\": ").append(backend.fallbackCount()).append(", ")
                    .append("\"cpuMaterializationCount\": ").append(backend.cpuMaterializationCount()).append(", ")
                    .append("\"internalCpuMaterializationCount\": ")
                    .append(backend.internalCpuMaterializationCount()).append(", ")
                    .append("\"gradientPublicationMaterializationCount\": ")
                    .append(backend.gradientPublicationMaterializationCount()).append(", ")
                    .append("\"cpuMaterializationReasonCounts\": ")
                    .append(intMapJson(backend.cpuMaterializationReasonCounts())).append(", ")
                    .append("\"cpuMaterializationBytes\": ").append(backend.cpuMaterializationBytes()).append(", ")
                    .append("\"cpuMaterializationDurationNs\": ")
                    .append(backend.cpuMaterializationDurationNs()).append(", ")
                    .append("\"copyDurationNs\": ").append(backend.copyDurationNs()).append(", ")
                    .append("\"nativeCopyStrategyCounts\": ")
                    .append(intMapJson(backend.nativeCopyStrategyCounts())).append(", ")
                    .append("\"executionRouteCounts\": ")
                    .append(intMapJson(backend.executionRouteCounts())).append(", ")
                    .append("\"rejectedRouteReasonCounts\": ")
                    .append(intMapJson(backend.rejectedRouteReasonCounts())).append(", ")
                    .append("\"deviceHandoffCount\": ").append(backend.deviceHandoffCount()).append(", ")
                    .append("\"gpuLayoutMaterializationCount\": ").append(backend.gpuLayoutMaterializationCount()).append(", ")
                    .append("\"gpuLayoutMaterializationBytes\": ").append(backend.gpuLayoutMaterializationBytes()).append(", ")
                    .append("\"gpuLayoutTransformKindCounts\": ")
                    .append(intMapJson(backend.gpuLayoutTransformKindCounts())).append(", ")
                    .append("\"gpuLayoutTargetLayoutClassCounts\": ")
                    .append(intMapJson(backend.gpuLayoutTargetLayoutClassCounts())).append(", ")
                    .append("\"storageResidencyCounts\": ").append(intMapJson(backend.storageResidencyCounts())).append(", ")
                    .append("\"dtypeResidencyEvidence\": ").append(intMapJson(backend.dtypeResidencyReasons())).append(", ")
                    .append("\"gpuFusedSubpatternCount\": ").append(backend.gpuFusedSubpatternCount()).append(", ")
                    .append("\"gpuFusedSubpatternTypes\": ").append(stringListJson(backend.gpuFusedSubpatternTypes())).append(", ")
                    .append("\"gpuFusedSubpatternOriginalNodeIds\": ").append(stringListJson(backend.gpuFusedSubpatternOriginalNodeIds())).append(", ")
                    .append("\"gpuFusedSubpatternLoweredPrimitiveCount\": ").append(backend.gpuFusedSubpatternLoweredPrimitiveCount()).append(", ")
                    .append("\"gpuFusedSubpatternReasons\": ").append(stringListJson(backend.gpuFusedSubpatternReasons())).append(", ")
                    .append("\"reasonCodes\": ").append(stringListJson(backend.reasonCodes())).append(", ")
                    .append("\"fallbackReasons\": ").append(stringListJson(backend.fallbackReasons())).append(", ")
                    .append("\"coverageGate\": ").append(gateResultJson(gate)).append(", ")
                    .append("\"nativeEvidence\": ").append(nativeEvidenceJson(nativeEvidence)).append(", ")
                    .append("\"targetCoverageTruth\": ").append(targetCoverageTruthJson(entry.getKey()))
                    .append("}");
        }
        sb.append('}');
        return sb.toString();
    }

    private static String crossBackendRouterEvidenceJson(CrossBackendRouterEvidence evidence) {
        if (evidence == null || !evidence.present()) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder("{");
        int i = 0;
        for (var entry : evidence.backends().entrySet()) {
            if (i++ > 0) {
                sb.append(", ");
            }
            var backend = entry.getValue();
            sb.append('"').append(escape(entry.getKey())).append("\": {")
                    .append("\"acceleratorPathCounts\": ").append(intMapJson(backend.acceleratorPathCounts())).append(", ")
                    .append("\"backendRouteCounts\": ").append(intMapJson(backend.backendRouteCounts())).append(", ")
                    .append("\"rejectedRouteReasonCounts\": ").append(intMapJson(backend.rejectedRouteReasonCounts())).append(", ")
                    .append("\"reasonCodeCounts\": ").append(intMapJson(backend.reasonCodeCounts())).append(", ")
                    .append("\"fallbackReasonCounts\": ").append(intMapJson(backend.fallbackReasonCounts())).append(", ")
                    .append("\"nativeCopyStrategyCounts\": ").append(intMapJson(backend.nativeCopyStrategyCounts())).append(", ")
                    .append("\"outputBufferWriteStatusCounts\": ")
                    .append(intMapJson(backend.outputBufferWriteStatusCounts())).append(", ")
                    .append("\"selectedRegionCount\": ").append(backend.selectedRegionCount()).append(", ")
                    .append("\"maxSelectedRegionLength\": ").append(backend.maxSelectedRegionLength()).append(", ")
                    .append("\"loweredPrimitiveCount\": ").append(backend.loweredPrimitiveCount()).append(", ")
                    .append("\"gpuFusedSubpatternCount\": ").append(backend.gpuFusedSubpatternCount()).append(", ")
                    .append("\"tensorArrayStepCount\": ").append(backend.tensorArrayStepCount()).append(", ")
                    .append("\"cpuFallbackStepCount\": ").append(backend.cpuFallbackStepCount()).append(", ")
                    .append("\"cpuMaterializationCount\": ").append(backend.cpuMaterializationCount()).append(", ")
                    .append("\"internalCpuMaterializationCount\": ")
                    .append(backend.internalCpuMaterializationCount()).append(", ")
                    .append("\"deviceHandoffCount\": ").append(backend.deviceHandoffCount())
                    .append("}");
        }
        sb.append('}');
        return sb.toString();
    }

    private static String targetCoverageTruthJson(String backendName) {
        backend.ComputeBackend computeBackend = switch (backendName) {
            case "GPU_METAL" -> backend.ComputeBackend.GPU_METAL;
            case "GPU_CUDA" -> backend.ComputeBackend.GPU_CUDA;
            default -> null;
        };
        if (computeBackend == null) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder("[");
        java.util.List<GpuTargetCoverageTruth.Row> rows = GpuTargetCoverageTruth.rowsFor(computeBackend);
        for (int i = 0; i < rows.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            GpuTargetCoverageTruth.Row row = rows.get(i);
            sb.append("{")
                    .append("\"op\": \"").append(escape(row.opType().name())).append("\", ")
                    .append("\"family\": \"").append(escape(row.family().name())).append("\", ")
                    .append("\"matrixStatus\": \"").append(escape(row.matrixStatus().name())).append("\", ")
                    .append("\"executionStatus\": \"").append(escape(row.executionStatus().name())).append("\", ")
                    .append("\"detail\": \"").append(escape(row.detail())).append("\"")
                    .append("}");
        }
        sb.append("]");
        return sb.toString();
    }

    private static String gateResultJson(GpuCoverageGateResult result) {
        return "{"
                + "\"gatePassed\": " + (result != null && result.passed()) + ", "
                + "\"gateFailures\": " + stringListJson(result == null ? java.util.List.of() : result.failures())
                + "}";
    }

    private static String nativeEvidenceJson(GpuCoverageNativeEvidence evidence) {
        if (evidence == null) {
            evidence = GpuCoverageNativeEvidence.capabilitySkipped("", "native evidence unavailable");
        }
        return "{"
                + "\"backend\": \"" + escape(evidence.backend()) + "\", "
                + "\"nativeStatus\": \"" + escape(evidence.nativeStatus()) + "\", "
                + "\"capabilitySkipped\": " + "capabilitySkipped".equals(evidence.nativeStatus()) + ", "
                + "\"detail\": \"" + escape(evidence.detail()) + "\""
                + "}";
    }

    private static String backendSelectionCostJson(graph.execution.trace.BackendSelectionTrace trace) {
        if (trace == null || trace.decisions().isEmpty()) {
            return "{}";
        }
        var selected = trace.decisions().stream()
                .filter(decision -> decision.selected() && decision.costSummary() != null)
                .toList();
        var finalists = rejectedFinalists(trace);
        if (selected.isEmpty() && finalists.isEmpty()) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder("{");
        sb.append("\"selected\": [");
        for (int i = 0; i < selected.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(selectedDecisionJson(selected.get(i)));
        }
        sb.append("], \"rejectedFinalists\": [");
        for (int i = 0; i < finalists.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(finalistJson(finalists.get(i)));
        }
        sb.append("]}");
        return sb.toString();
    }

    private static String selectedDecisionJson(graph.execution.trace.BackendSelectionDecisionTrace decision) {
        var summary = decision.costSummary();
        return "{"
                + "\"nodeIds\": " + intListJson(decision.nodeIds()) + ", "
                + "\"selectedBackend\": \"" + escape(String.valueOf(decision.selectedBackend())) + "\", "
                + "\"reason\": \"" + escape(decision.reason()) + "\", "
                + "\"finalScore\": " + format(summary.finalScore()) + ", "
                + "\"boundaryCount\": " + summary.boundaryCount() + ", "
                + "\"estimatedTransferBytes\": " + summary.estimatedTransferBytes() + ", "
                + "\"layoutFallbackBytes\": " + summary.layoutFallbackBytes() + ", "
                + "\"estimatedComputeWork\": " + summary.estimatedComputeWork() + ", "
                + "\"preset\": \"" + escape(summary.preset()) + "\", "
                + "\"cost_explanation\": " + CostExplanationJsonRenderer.render(summary.toCostScore().explain(summary.reasonCode()))
                + manifestJsonSuffix(decision.gpuLoweredRegionManifest())
                + "}";
    }

    private static String manifestJsonSuffix(GpuLoweredRegionManifest manifest) {
        if (manifest == null) {
            return "";
        }
        return ", \"gpuLoweredRegionManifest\": " + manifestJson(manifest);
    }

    private static String manifestJson(GpuLoweredRegionManifest manifest) {
        return "{"
                + "\"regionId\": \"" + escape(manifest.regionId()) + "\", "
                + "\"backend\": \"" + escape(String.valueOf(manifest.backend())) + "\", "
                + "\"selectedRegionLength\": " + manifest.selectedRegionLength() + ", "
                + "\"originalOps\": " + originalOpsJson(manifest.originalOps()) + ", "
                + "\"loweredPrimitives\": " + loweredPrimitivesJson(manifest.loweredPrimitives()) + ", "
                + "\"valueAssumptions\": " + valueAssumptionsJson(manifest) + ", "
                + "\"dtypeResidencyEvidence\": " + stringMapJson(dtypeResidencyEvidence(manifest)) + ", "
                + "\"fusedSubpatterns\": " + fusedSubpatternsJson(manifest.fusedSubpatterns()) + ", "
                + "\"rejections\": " + rejectionsJson(manifest.rejections()) + ", "
                + "\"candidateSpan\": " + candidateSpanJson(manifest.candidateSpan())
                + "}";
    }

    private static Map<String, String> dtypeResidencyEvidence(GpuLoweredRegionManifest manifest) {
        if (manifest == null || manifest.backendExtensions().isEmpty()) {
            return Map.of();
        }
        java.util.LinkedHashMap<String, String> out = new java.util.LinkedHashMap<>();
        manifest.backendExtensions().forEach((key, value) -> {
            if (key.startsWith("dtypeResidency.")) {
                out.put(key, value);
            }
        });
        return out;
    }

    private static String originalOpsJson(java.util.List<GpuLoweredRegionOriginalOp> ops) {
        if (ops == null || ops.isEmpty()) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < ops.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            GpuLoweredRegionOriginalOp op = ops.get(i);
            sb.append("{")
                    .append("\"nodeId\": ").append(op.nodeId()).append(", ")
                    .append("\"opType\": \"").append(escape(op.opType())).append("\", ")
                    .append("\"inputNodeIds\": ").append(intListJson(op.inputNodeIds())).append(", ")
                    .append("\"outputNodeIds\": ").append(intListJson(op.outputNodeIds())).append(", ")
                    .append("\"dataType\": \"").append(op.dataType().name()).append("\", ")
                    .append("\"shape\": ").append(intListJson(op.shape())).append(", ")
                    .append("\"loweredPrimitiveIds\": ").append(stringListJson(op.loweredPrimitiveIds())).append(", ")
                    .append("\"aggregatedReasons\": ").append(reasonListJson(op.aggregatedReasons()))
                    .append("}");
        }
        sb.append(']');
        return sb.toString();
    }

    private static String loweredPrimitivesJson(java.util.List<GpuLoweredPrimitiveManifest> primitives) {
        if (primitives == null || primitives.isEmpty()) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < primitives.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            GpuLoweredPrimitiveManifest primitive = primitives.get(i);
            sb.append("{")
                    .append("\"primitiveId\": \"").append(escape(primitive.primitiveId())).append("\", ")
                    .append("\"primitiveType\": \"").append(escape(primitive.primitiveType())).append("\", ")
                    .append("\"sourceOriginalNodeIds\": ").append(intListJson(primitive.sourceOriginalNodeIds())).append(", ")
                    .append("\"inputRefs\": ").append(stringListJson(primitive.inputRefs())).append(", ")
                    .append("\"outputRef\": \"").append(escape(primitive.outputRef())).append("\", ")
                    .append("\"dataType\": \"").append(primitive.dataType().name()).append("\", ")
                    .append("\"shape\": ").append(intListJson(primitive.shape())).append(", ")
                    .append("\"reasons\": ").append(reasonListJson(primitive.reasons()))
                    .append("}");
        }
        sb.append(']');
        return sb.toString();
    }

    private static String valueAssumptionsJson(GpuLoweredRegionManifest manifest) {
        java.util.List<String> items = new java.util.ArrayList<>();
        for (GpuLoweredRegionValueAssumption assumption : manifest.inputAssumptions()) {
            items.add(valueAssumptionJson("input", assumption));
        }
        for (GpuLoweredRegionValueAssumption assumption : manifest.outputAssumptions()) {
            items.add(valueAssumptionJson("output", assumption));
        }
        return "[" + String.join(", ", items) + "]";
    }

    private static String valueAssumptionJson(String scope, GpuLoweredRegionValueAssumption assumption) {
        return "{"
                + "\"scope\": \"" + escape(scope) + "\", "
                + "\"nodeId\": " + assumption.nodeId() + ", "
                + "\"role\": \"" + escape(assumption.role()) + "\", "
                + "\"dataType\": \"" + assumption.dataType().name() + "\", "
                + "\"rank\": " + assumption.rank() + ", "
                + "\"shape\": " + intListJson(assumption.shape()) + ", "
                + "\"layout\": \"" + escape(assumption.layout()) + "\", "
                + "\"contiguous\": " + assumption.contiguous() + ", "
                + "\"hasStorageOffset\": " + assumption.hasStorageOffset() + ", "
                + "\"storageOffset\": " + assumption.storageOffset()
                + "}";
    }

    private static String fusedSubpatternsJson(java.util.List<GpuFusionSubpatternSummary> subpatterns) {
        if (subpatterns == null || subpatterns.isEmpty()) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < subpatterns.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            GpuFusionSubpatternSummary subpattern = subpatterns.get(i);
            sb.append("{")
                    .append("\"patternType\": \"").append(subpattern.patternType().name()).append("\", ")
                    .append("\"supported\": ").append(subpattern.supported()).append(", ")
                    .append("\"reason\": \"").append(subpattern.reason().name()).append("\", ")
                    .append("\"originalOperationNodeIds\": ").append(intListJson(subpattern.originalOperationNodeIds())).append(", ")
                    .append("\"loweredPrimitiveIds\": ").append(stringListJson(subpattern.loweredPrimitiveIds())).append(", ")
                    .append("\"loweredPrimitiveCount\": ").append(subpattern.loweredPrimitiveCount()).append(", ")
                    .append("\"detail\": \"").append(escape(subpattern.detail())).append("\"")
                    .append("}");
        }
        sb.append(']');
        return sb.toString();
    }

    private static String rejectionsJson(java.util.List<GpuLoweredRegionRejection> rejections) {
        if (rejections == null || rejections.isEmpty()) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < rejections.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            GpuLoweredRegionRejection rejection = rejections.get(i);
            sb.append("{")
                    .append("\"level\": \"").append(escape(rejection.level())).append("\", ")
                    .append("\"originalNodeId\": ").append(rejection.originalNodeId()).append(", ")
                    .append("\"primitiveId\": \"").append(escape(rejection.primitiveId())).append("\", ")
                    .append("\"fusedPatternType\": \"").append(escape(rejection.fusedPatternType())).append("\", ")
                    .append("\"reason\": \"").append(rejection.reason().name()).append("\", ")
                    .append("\"detail\": \"").append(escape(rejection.detail())).append("\"")
                    .append("}");
        }
        sb.append(']');
        return sb.toString();
    }

    private static String candidateSpanJson(backend.accelerator.lowering.GpuLoweredRegionCandidateSpan span) {
        if (span == null) {
            return "{}";
        }
        return "{"
                + "\"originalCandidateNodeIds\": " + intListJson(span.originalCandidateNodeIds()) + ", "
                + "\"acceptedNodeIds\": " + intListJson(span.acceptedNodeIds()) + ", "
                + "\"rejectedOriginalNodeId\": " + span.rejectedOriginalNodeId() + ", "
                + "\"rejectedPrimitiveId\": \"" + escape(span.rejectedPrimitiveId()) + "\", "
                + "\"reason\": \"" + span.reason().name() + "\""
                + "}";
    }

    private static String reasonListJson(java.util.List<backend.accelerator.lowering.GpuLoweringUnsupportedReason> reasons) {
        if (reasons == null || reasons.isEmpty()) {
            return "[]";
        }
        return stringListJson(reasons.stream().map(Enum::name).toList());
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

    private static String finalistJson(graph.execution.trace.PartitionDecisionTrace.CandidateCostTrace finalist) {
        return "{"
                + "\"nodeIds\": " + intListJson(finalist.nodeIds()) + ", "
                + "\"reason\": \"" + escape(finalist.reason()) + "\", "
                + "\"finalScore\": " + format(finalist.finalScore()) + ", "
                + "\"boundaryCount\": " + finalist.boundaryCount() + ", "
                + "\"estimatedTransferBytes\": " + finalist.estimatedTransferBytes() + ", "
                + "\"layoutFallbackBytes\": " + finalist.layoutFallbackBytes() + ", "
                + "\"estimatedComputeWork\": " + finalist.estimatedComputeWork() + ", "
                + "\"preset\": \"" + escape(finalist.preset()) + "\", "
                + "\"cost_explanation\": " + CostExplanationJsonRenderer.render(finalist.toCostScore().explain(finalist.reason()))
                + "}";
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

    private static void appendCpuMaterializationJson(
            StringBuilder sb,
            graph.execution.trace.CpuMaterializationTrace materialization,
            String indent
    ) {
        sb.append(indent).append("{\n");
        sb.append(indent).append("  \"nodeId\": ").append(materialization.nodeId()).append(",\n");
        sb.append(indent).append("  \"reason\": \"").append(materialization.reason().name()).append("\",\n");
        sb.append(indent).append("  \"materializedFrom\": \"").append(escape(materialization.materializedFrom())).append("\",\n");
        sb.append(indent).append("  \"sourceResidency\": \"").append(materialization.sourceResidency().name()).append("\",\n");
        sb.append(indent).append("  \"bytes\": ").append(materialization.bytes()).append(",\n");
        sb.append(indent).append("  \"durationMs\": ").append(format(nanosToMs(materialization.durationNs()))).append(",\n");
        sb.append(indent).append("  \"durationNs\": ").append(materialization.durationNs()).append(",\n");
        sb.append(indent).append("  \"completed\": ").append(materialization.completed()).append(",\n");
        sb.append(indent).append("  \"detail\": \"").append(escape(materialization.detail())).append("\"\n");
        sb.append(indent).append("}");
    }

    private static void appendStepJson(StringBuilder sb, graph.execution.trace.ExecutionStepTrace step, String indent) {
        sb.append(indent).append("{\n");
        sb.append(indent).append("  \"index\": ").append(step.index()).append(",\n");
        sb.append(indent).append("  \"label\": \"").append(escape(step.label())).append("\",\n");
        sb.append(indent).append("  \"opType\": \"").append(escape(step.opType())).append("\",\n");
        sb.append(indent).append("  \"shape\": ").append(intListJson(step.shape())).append(",\n");
        sb.append(indent).append("  \"dataType\": \"").append(step.dataType() == null ? "" : escape(step.dataType().name())).append("\",\n");
        sb.append(indent).append("  \"backend\": \"").append(escape(step.backend())).append("\",\n");
        sb.append(indent).append("  \"kernel\": \"").append(escape(step.kernel())).append("\",\n");
        sb.append(indent).append("  \"durationMs\": ").append(format(nanosToMs(step.durationNs()))).append(",\n");
        appendStepMetadataJson(sb, step.metadata(), indent + "  ");
        sb.append('\n').append(indent).append('}');
    }

    private static void appendStepMetadataJson(
            StringBuilder sb,
            graph.execution.trace.StepExecutionMetadata metadata,
            String indent
    ) {
        if (metadata == null) {
            sb.append(indent).append("\"metadata\": null");
            return;
        }
        sb.append(indent).append("\"metadata\": {\n");
        sb.append(indent).append("  \"kind\": \"").append(escape(metadata.kind())).append("\",\n");
        sb.append(indent).append("  \"attributes\": ").append(mapJson(metadata.attributes())).append(",\n");
        sb.append(indent).append("  \"metalRouteCostExplanation\": ")
                .append(metalRouteCostExplanationJson(metadata.attributes()))
                .append(",\n");
        sb.append(indent).append("  \"compute\": ").append(computeJson(metadata.compute())).append(",\n");
        sb.append(indent).append("  \"layout\": ").append(layoutJson(metadata.layout())).append(",\n");
        sb.append(indent).append("  \"dispatch\": ").append(dispatchJson(metadata.dispatch())).append(",\n");
        sb.append(indent).append("  \"reduction\": ").append(reductionJson(metadata.reduction())).append(",\n");
        sb.append(indent).append("  \"matMul\": ").append(matMulJson(metadata.matMul())).append(",\n");
        sb.append(indent).append("  \"conv\": ").append(convJson(metadata.conv())).append(",\n");
        sb.append(indent).append("  \"fused\": ").append(fusedJson(metadata.fused())).append('\n');
        sb.append(indent).append("}");
    }

    private static String optimizerCostJson(graph.optimizer.state.OptimizerTrace trace) {
        if (trace == null) {
            return "{\"events\": [], \"cost_explanations\": []}";
        }
        StringBuilder sb = new StringBuilder("{");
        sb.append("\"events\": ").append(stringListJson(trace.events())).append(", ");
        sb.append("\"cost_explanations\": [");
        for (int i = 0; i < trace.costExplanations().size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(CostExplanationJsonRenderer.render(trace.costExplanations().get(i)));
        }
        sb.append("]}");
        return sb.toString();
    }

    private static String metalRouteCostExplanationJson(Map<String, Object> attrs) {
        if (attrs == null || !attrs.containsKey("metalRouteCostModel")) {
            return "null";
        }
        return "{"
                + "\"model\": " + valueJson(attrs.get("metalRouteCostModel")) + ", "
                + "\"input_kind\": " + valueJson(attrs.get("metalRouteCostInputKind")) + ", "
                + "\"reason\": " + valueJson(attrs.get("metalRouteCostReason")) + ", "
                + "\"comparison\": " + valueJson(attrs.get("metalRouteCostComparison")) + ", "
                + "\"top_contributors\": " + valueJson(attrs.get("metalRouteCostTopContributors")) + ", "
                + "\"components\": " + valueJson(attrs.get("metalRouteCostComponents"))
                + "}";
    }

    private static String computeJson(graph.execution.trace.ComputeTraceMetadata compute) {
        if (compute == null) {
            return "null";
        }
        return "{"
                + "\"mode\": \"" + escape(compute.mode()) + "\", "
                + "\"storageType\": \"" + escape(compute.storageType()) + "\", "
                + "\"computeType\": \"" + escape(compute.computeType()) + "\", "
                + "\"backend\": \"" + escape(compute.backend()) + "\", "
                + "\"accumulateType\": \"" + escape(compute.accumulateType()) + "\""
                + "}";
    }

    private static String layoutJson(graph.execution.trace.LayoutTraceMetadata layout) {
        if (layout == null) {
            return "null";
        }
        return "{"
                + "\"storageOffset\": " + layout.storageOffset() + ", "
                + "\"contiguous\": " + layout.contiguous() + ", "
                + "\"stridedPath\": " + layout.stridedPath() + ", "
                + "\"targetType\": \"" + escape(layout.targetType()) + "\""
                + "}";
    }

    private static String dispatchJson(graph.execution.trace.DispatchTraceMetadata dispatch) {
        if (dispatch == null) {
            return "null";
        }
        return "{"
                + "\"mode\": \"" + escape(dispatch.mode()) + "\", "
                + "\"vectorWidth\": " + dispatch.vectorWidth() + ", "
                + "\"plannedWorkers\": " + dispatch.plannedWorkers() + ", "
                + "\"scalarChunkSize\": " + dispatch.scalarChunkSize() + ", "
                + "\"vectorChunkSize\": " + dispatch.vectorChunkSize()
                + "}";
    }

    private static String reductionJson(graph.execution.trace.ReductionTraceMetadata reduction) {
        if (reduction == null) {
            return "null";
        }
        return "{"
                + "\"mode\": \"" + escape(reduction.mode()) + "\", "
                + "\"plannedWorkers\": " + reduction.plannedWorkers() + ", "
                + "\"chunkSize\": " + reduction.chunkSize() + ", "
                + "\"vectorWidth\": " + reduction.vectorWidth() + ", "
                + "\"accuracyMode\": \"" + escape(reduction.accuracyMode()) + "\""
                + "}";
    }

    private static String matMulJson(graph.execution.trace.MatMulTraceMetadata matMul) {
        if (matMul == null) {
            return "null";
        }
        return "{"
                + "\"useBlas\": " + matMul.useBlas() + ", "
                + "\"useBatchedBlas\": " + matMul.useBatchedBlas() + ", "
                + "\"blasProvider\": \"" + escape(matMul.blasProvider()) + "\", "
                + "\"blasSymbol\": \"" + escape(matMul.blasSymbol()) + "\", "
                + "\"blasRoute\": \"" + escape(matMul.blasRoute()) + "\", "
                + "\"route\": \"" + escape(matMul.route()) + "\", "
                + "\"cpuStorageProfile\": \"" + escape(matMul.cpuStorageProfile()) + "\", "
                + "\"nativeCpuFailurePolicy\": \"" + escape(matMul.nativeCpuFailurePolicy()) + "\", "
                + "\"requestedCpuStorage\": \"" + escape(matMul.requestedCpuStorage()) + "\", "
                + "\"actualCpuStorage\": \"" + escape(matMul.actualCpuStorage()) + "\", "
                + "\"nativeCpuFallbackReason\": \"" + escape(matMul.nativeCpuFallbackReason()) + "\", "
                + "\"openblasSgemmAvailable\": " + matMul.openblasSgemmAvailable() + ", "
                + "\"openblasDgemmAvailable\": " + matMul.openblasDgemmAvailable() + ", "
                + "\"openblasSbgemmAvailable\": " + matMul.openblasSbgemmAvailable() + ", "
                + "\"openblasBgemmAvailable\": " + matMul.openblasBgemmAvailable() + ", "
                + "\"bf16ContinuationRoute\": \"" + escape(matMul.bf16ContinuationRoute()) + "\", "
                + "\"bf16OutputRoute\": \"" + escape(matMul.bf16OutputRoute()) + "\", "
                + "\"bf16ComputePrecision\": \"" + escape(matMul.bf16ComputePrecision()) + "\", "
                + "\"bf16OutputPrecision\": \"" + escape(matMul.bf16OutputPrecision()) + "\", "
                + "\"copyInBytes\": " + matMul.copyInBytes() + ", "
                + "\"copyOutBytes\": " + matMul.copyOutBytes() + ", "
                + "\"nativeTempBytes\": " + matMul.nativeTempBytes() + ", "
                + "\"threadPolicy\": \"" + escape(matMul.threadPolicy()) + "\", "
                + "\"fallbackReason\": \"" + escape(matMul.fallbackReason()) + "\", "
                + "\"parallel\": " + matMul.parallel() + ", "
                + "\"tileM\": " + matMul.tileM() + ", "
                + "\"tileN\": " + matMul.tileN() + ", "
                + "\"tileK\": " + matMul.tileK() + ", "
                + "\"plannedWorkers\": " + matMul.plannedWorkers() + ", "
                + "\"work\": " + matMul.work() + ", "
                + "\"microKernel\": \"" + escape(matMul.microKernel()) + "\""
                + "}";
    }

    private static String convJson(graph.execution.trace.ConvTraceMetadata conv) {
        if (conv == null) {
            return "null";
        }
        return "{"
                + "\"executionKind\": \"" + escape(conv.executionKind()) + "\", "
                + "\"lowered\": " + conv.lowered() + ", "
                + "\"blasUsed\": " + conv.blasUsed() + ", "
                + "\"blasProvider\": \"" + escape(conv.blasProvider()) + "\", "
                + "\"m\": " + conv.m() + ", "
                + "\"n\": " + conv.n() + ", "
                + "\"k\": " + conv.k() + ", "
                + "\"blasCalls\": " + conv.blasCalls() + ", "
                + "\"javaCalls\": " + conv.javaCalls()
                + "}";
    }

    private static String fusedJson(graph.execution.trace.FusedTraceMetadata fused) {
        if (fused == null) {
            return "null";
        }
        return "{"
                + "\"precisionMode\": " + fused.precisionMode() + ", "
                + "\"lowCostHint\": " + fused.lowCostHint() + ", "
                + "\"dispatchFamily\": \"" + escape(fused.dispatchFamily()) + "\", "
                + "\"schedulerSignature\": \"" + escape(fused.schedulerSignature()) + "\", "
                + "\"executionBackend\": \"" + escape(fused.executionBackend()) + "\", "
                + "\"dispatchScale\": " + fused.dispatchScale() + ", "
                + "\"fusedNodeCount\": " + fused.fusedNodeCount() + ", "
                + "\"fusedInputCount\": " + fused.fusedInputCount()
                + "}";
    }

    private static String intListJson(java.util.List<Integer> values) {
        if (values == null || values.isEmpty()) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(values.get(i));
        }
        sb.append(']');
        return sb.toString();
    }

    private static String stringListJson(java.util.List<String> values) {
        if (values == null || values.isEmpty()) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append('"').append(escape(values.get(i))).append('"');
        }
        sb.append(']');
        return sb.toString();
    }

    private static String mapJson(Map<String, Object> map) {
        if (map == null || map.isEmpty()) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder("{");
        int index = 0;
        for (var entry : map.entrySet()) {
            if (index++ > 0) {
                sb.append(", ");
            }
            sb.append('"').append(escape(entry.getKey())).append('"').append(": ").append(valueJson(entry.getValue()));
        }
        sb.append('}');
        return sb.toString();
    }

    private static String intMapJson(Map<String, Integer> map) {
        if (map == null || map.isEmpty()) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder("{");
        int index = 0;
        for (var entry : map.entrySet()) {
            if (index++ > 0) {
                sb.append(", ");
            }
            sb.append('"').append(escape(entry.getKey())).append('"').append(": ").append(entry.getValue());
        }
        sb.append('}');
        return sb.toString();
    }

    private static String stringMapJson(Map<String, String> map) {
        if (map == null || map.isEmpty()) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder("{");
        int index = 0;
        for (var entry : map.entrySet()) {
            if (index++ > 0) {
                sb.append(", ");
            }
            sb.append('"').append(escape(entry.getKey())).append('"')
                    .append(": \"").append(escape(entry.getValue())).append('"');
        }
        sb.append('}');
        return sb.toString();
    }

    private static String valueJson(Object value) {
        if (value == null) {
            return "null";
        }
        if (value instanceof Number || value instanceof Boolean) {
            return String.valueOf(value);
        }
        if (value instanceof java.util.List<?> list) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(valueJson(list.get(i)));
            }
            sb.append(']');
            return sb.toString();
        }
        return "\"" + escape(String.valueOf(value)) + "\"";
    }
}
