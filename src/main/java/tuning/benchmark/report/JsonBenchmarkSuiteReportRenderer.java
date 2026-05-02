package tuning.benchmark.report;

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
        sb.append("  \"targetCoverageGates\": [\n");
        java.util.List<GpuCoverageHotPathExpectation> expectations = GpuHotPathCoverageTargets.defaultExpectations();
        java.util.List<GpuCoverageGateResult> targetGates = GpuCoverageRegressionGate.evaluateTargets(report, expectations);
        for (int i = 0; i < expectations.size(); i++) {
            if (i > 0) {
                sb.append(",\n");
            }
            GpuCoverageHotPathExpectation expectation = expectations.get(i);
            GpuCoverageGateResult gate = targetGates.get(i);
            sb.append("    {");
            sb.append("\"workloadName\": \"").append(escape(expectation.workloadName())).append("\", ");
            sb.append("\"backend\": \"").append(escape(expectation.backend())).append("\", ");
            sb.append("\"nativeEvidenceRequired\": ").append(expectation.nativeEvidenceRequired()).append(", ");
            sb.append("\"expectedVisibleReasons\": ").append(stringListJson(expectation.expectedVisibleReasons())).append(", ");
            sb.append("\"policy\": ").append(policyJson(expectation.policy())).append(", ");
            sb.append("\"coverageGate\": {");
            sb.append("\"gatePassed\": ").append(gate.passed()).append(", ");
            sb.append("\"gateFailures\": ").append(stringListJson(gate.failures())).append(", ");
            sb.append("\"coverage\": ").append(coverageJson(gate.coverage()));
            sb.append("}");
            sb.append("}");
        }
        sb.append("\n  ],\n");
        sb.append("  \"coverageSummary\": [\n");
        int coverageIndex = 0;
        for (var entry : report.bestCoverageByBackend().entrySet()) {
            if (coverageIndex++ > 0) {
                sb.append(",\n");
            }
            var coverage = entry.getValue();
            GpuCoverageNativeEvidence nativeEvidence = TextBenchmarkReportRenderer.nativeEvidence(entry.getKey(), coverage);
            sb.append("    {");
            sb.append("\"backend\": \"").append(escape(entry.getKey())).append("\", ");
            sb.append("\"gpuCoverageRatio\": ").append(format(coverage.gpuCoverageRatio())).append(", ");
            sb.append("\"maxSelectedRegionLength\": ").append(coverage.maxSelectedRegionLength()).append(", ");
            sb.append("\"loweredPrimitiveCount\": ").append(coverage.loweredPrimitiveCount()).append(", ");
            sb.append("\"nativeBufferStepCount\": ").append(coverage.nativeBufferStepCount()).append(", ");
            sb.append("\"tensorArrayStepCount\": ").append(coverage.tensorArrayStepCount()).append(", ");
            sb.append("\"cpuFallbackStepCount\": ").append(coverage.cpuFallbackStepCount()).append(", ");
            sb.append("\"cpuMaterializationCount\": ").append(coverage.cpuMaterializationCount()).append(", ");
            sb.append("\"internalCpuMaterializationCount\": ")
                    .append(coverage.internalCpuMaterializationCount()).append(", ");
            sb.append("\"gradientPublicationMaterializationCount\": ")
                    .append(coverage.gradientPublicationMaterializationCount()).append(", ");
            sb.append("\"fallbackCount\": ").append(coverage.fallbackCount()).append(", ");
            sb.append("\"deviceHandoffCount\": ").append(coverage.deviceHandoffCount()).append(", ");
            sb.append("\"dtypeResidencyEvidence\": ").append(intMapJson(coverage.dtypeResidencyReasons())).append(", ");
            sb.append("\"nativeEvidence\": {");
            sb.append("\"backend\": \"").append(escape(nativeEvidence.backend())).append("\", ");
            sb.append("\"nativeStatus\": \"").append(escape(nativeEvidence.nativeStatus())).append("\", ");
            sb.append("\"capabilitySkipped\": ").append("capabilitySkipped".equals(nativeEvidence.nativeStatus())).append(", ");
            sb.append("\"detail\": \"").append(escape(nativeEvidence.detail())).append("\"");
            sb.append("}, ");
            sb.append("\"coverageDeltaVsBaseline\": ").append(comparisonJson(GpuCoverageComparison.compare(
                    GpuCoverageBaseline.v14Closure(entry.getKey()),
                    coverage
            )));
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

    private static String policyJson(GpuCoverageGatePolicy policy) {
        if (policy == null) {
            return "{}";
        }
        return "{"
                + "\"backend\": \"" + escape(policy.backend()) + "\", "
                + "\"minGpuCoverageRatio\": " + format(policy.minGpuCoverageRatio()) + ", "
                + "\"minMaxSelectedRegionLength\": " + policy.minMaxSelectedRegionLength() + ", "
                + "\"minMultiOpGpuRegionCount\": " + policy.minMultiOpGpuRegionCount() + ", "
                + "\"minLoweredPrimitiveCount\": " + policy.minLoweredPrimitiveCount() + ", "
                + "\"minGpuFusedSubpatternCount\": " + policy.minGpuFusedSubpatternCount() + ", "
                + "\"maxCpuMaterializationCount\": " + policy.maxCpuMaterializationCount() + ", "
                + "\"maxInternalCpuMaterializationCount\": " + policy.maxInternalCpuMaterializationCount() + ", "
                + "\"maxGradientPublicationMaterializationCount\": "
                + policy.maxGradientPublicationMaterializationCount() + ", "
                + "\"maxFallbackCount\": " + policy.maxFallbackCount() + ", "
                + "\"maxTensorArrayStepCount\": " + policy.maxTensorArrayStepCount() + ", "
                + "\"maxDeviceHandoffCount\": " + policy.maxDeviceHandoffCount() + ", "
                + "\"requireNativeBufferBinding\": " + policy.requireNativeBufferBinding()
                + "}";
    }

    private static String coverageJson(GpuCoverageSummary.BackendCoverage coverage) {
        if (coverage == null) {
            return "null";
        }
        return "{"
                + "\"gpuCoverageRatio\": " + format(coverage.gpuCoverageRatio()) + ", "
                + "\"maxSelectedRegionLength\": " + coverage.maxSelectedRegionLength() + ", "
                + "\"loweredPrimitiveCount\": " + coverage.loweredPrimitiveCount() + ", "
                + "\"nativeBufferStepCount\": " + coverage.nativeBufferStepCount() + ", "
                + "\"tensorArrayStepCount\": " + coverage.tensorArrayStepCount() + ", "
                + "\"cpuFallbackStepCount\": " + coverage.cpuFallbackStepCount() + ", "
                + "\"cpuMaterializationCount\": " + coverage.cpuMaterializationCount() + ", "
                + "\"internalCpuMaterializationCount\": " + coverage.internalCpuMaterializationCount() + ", "
                + "\"gradientPublicationMaterializationCount\": "
                + coverage.gradientPublicationMaterializationCount() + ", "
                + "\"fallbackCount\": " + coverage.fallbackCount() + ", "
                + "\"deviceHandoffCount\": " + coverage.deviceHandoffCount() + ", "
                + "\"dtypeResidencyEvidence\": " + intMapJson(coverage.dtypeResidencyReasons()) + ", "
                + "\"reasonCodes\": " + stringListJson(coverage.reasonCodes()) + ", "
                + "\"fallbackReasons\": " + stringListJson(coverage.fallbackReasons())
                + "}";
    }

    private static String intMapJson(java.util.Map<String, Integer> values) {
        if (values == null || values.isEmpty()) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder("{");
        int i = 0;
        for (var entry : values.entrySet()) {
            if (i++ > 0) {
                sb.append(", ");
            }
            sb.append('"').append(escape(entry.getKey())).append("\": ").append(entry.getValue());
        }
        sb.append('}');
        return sb.toString();
    }

    private static String comparisonJson(GpuCoverageComparison comparison) {
        if (comparison == null) {
            return "null";
        }
        return "{"
                + "\"baselineName\": \"" + escape(comparison.baselineName()) + "\", "
                + "\"backend\": \"" + escape(comparison.backend()) + "\", "
                + "\"passes\": " + comparison.passes() + ", "
                + "\"baselineMaxSelectedRegionLength\": " + comparison.baselineMaxSelectedRegionLength() + ", "
                + "\"currentMaxSelectedRegionLength\": " + comparison.currentMaxSelectedRegionLength() + ", "
                + "\"baselineCpuMaterializationCount\": " + comparison.baselineCpuMaterializationCount() + ", "
                + "\"currentCpuMaterializationCount\": " + comparison.currentCpuMaterializationCount() + ", "
                + "\"baselineFallbackCount\": " + comparison.baselineFallbackCount() + ", "
                + "\"currentFallbackCount\": " + comparison.currentFallbackCount() + ", "
                + "\"baselineDeviceHandoffCount\": " + comparison.baselineDeviceHandoffCount() + ", "
                + "\"currentDeviceHandoffCount\": " + comparison.currentDeviceHandoffCount() + ", "
                + "\"improvements\": " + stringListJson(comparison.improvements()) + ", "
                + "\"regressions\": " + stringListJson(comparison.regressions())
                + "}";
    }
}
