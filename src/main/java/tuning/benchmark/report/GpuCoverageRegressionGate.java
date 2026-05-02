package tuning.benchmark.report;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Fail-fast evaluator for GPU coverage regressions.
 */
public final class GpuCoverageRegressionGate {
    private GpuCoverageRegressionGate() {
    }

    public static GpuCoverageGateResult evaluate(GpuCoverageSummary summary, GpuCoverageGatePolicy policy) {
        if (policy == null) {
            policy = GpuCoverageGatePolicy.nativeBufferTarget("", 0.0d, 0);
        }
        var failures = new ArrayList<String>();
        GpuCoverageSummary.BackendCoverage coverage = summary == null ? null : summary.backends().get(policy.backend());
        if (coverage == null) {
            failures.add("missing coverage summary");
            return new GpuCoverageGateResult(false, failures, null);
        }
        if (coverage.gpuCoverageRatio() < policy.minGpuCoverageRatio()) {
            failures.add("lost GPU coverage");
        }
        if (coverage.maxSelectedRegionLength() < policy.minMaxSelectedRegionLength()) {
            failures.add("lost GPU coverage");
        }
        if (coverage.multiOpGpuRegionCount() < policy.minMultiOpGpuRegionCount()) {
            failures.add("lost multi-op GPU region coverage");
        }
        if (coverage.loweredPrimitiveCount() < policy.minLoweredPrimitiveCount()) {
            failures.add("lost lowered primitive coverage");
        }
        if (coverage.gpuFusedSubpatternCount() < policy.minGpuFusedSubpatternCount()) {
            failures.add("lost fused subpattern coverage");
        }
        if (coverage.cpuMaterializationCount() > policy.maxCpuMaterializationCount()) {
            failures.add("unexpected CPU materialization");
        }
        if (coverage.fallbackCount() > policy.maxFallbackCount()) {
            failures.add("lost GPU coverage");
        }
        if (coverage.cpuFallbackStepCount() > 0) {
            failures.add("unexpected CPU fallback");
        }
        if (coverage.tensorArrayStepCount() > policy.maxTensorArrayStepCount()) {
            failures.add("hidden tensor-array fallback");
        }
        if (policy.requireNativeBufferBinding() && coverage.bufferBindingStepCount() == 0) {
            failures.add("lost native buffer binding");
        }
        if (coverage.deviceHandoffCount() > policy.maxDeviceHandoffCount()) {
            failures.add("unexpected device handoff");
        }
        return new GpuCoverageGateResult(failures.isEmpty(), failures, coverage);
    }

    public static void requirePass(GpuCoverageSummary summary, GpuCoverageGatePolicy policy) {
        GpuCoverageGateResult result = evaluate(summary, policy);
        if (!result.passed()) {
            throw new IllegalStateException(String.join("; ", result.failures()));
        }
    }

    public static List<GpuCoverageGateResult> evaluateTargets(
            BenchmarkSuiteReport suiteReport,
            List<GpuCoverageHotPathExpectation> expectations
    ) {
        if (expectations == null || expectations.isEmpty()) {
            return List.of();
        }
        var results = new ArrayList<GpuCoverageGateResult>();
        for (GpuCoverageHotPathExpectation expectation : expectations) {
            GpuCoverageSummary.BackendCoverage coverage = findCoverage(suiteReport, expectation);
            if (coverage == null) {
                results.add(new GpuCoverageGateResult(
                        false,
                        List.of("missing target coverage summary workload="
                                + expectation.workloadName()
                                + " backend="
                                + expectation.backend()),
                        null
                ));
                continue;
            }
            GpuCoverageGateResult result = evaluate(
                    new GpuCoverageSummary(Map.of(expectation.backend(), coverage)),
                    expectation.policy()
            );
            if (!expectation.expectedVisibleReasons().isEmpty()
                    && !hasExpectedVisibleReason(coverage, expectation.expectedVisibleReasons())) {
                var failures = new ArrayList<>(result.failures());
                failures.add("missing expected visible reason workload="
                        + expectation.workloadName()
                        + " backend="
                        + expectation.backend());
                result = new GpuCoverageGateResult(false, failures, coverage);
            }
            results.add(result);
        }
        return List.copyOf(results);
    }

    public static void requireTargetsPass(
            BenchmarkSuiteReport suiteReport,
            List<GpuCoverageHotPathExpectation> expectations
    ) {
        List<GpuCoverageGateResult> results = evaluateTargets(suiteReport, expectations);
        List<String> failures = results.stream()
                .filter(result -> !result.passed())
                .flatMap(result -> result.failures().stream())
                .toList();
        if (!failures.isEmpty()) {
            throw new IllegalStateException(String.join("; ", failures));
        }
    }

    private static GpuCoverageSummary.BackendCoverage findCoverage(
            BenchmarkSuiteReport suiteReport,
            GpuCoverageHotPathExpectation expectation
    ) {
        if (suiteReport == null || expectation == null) {
            return null;
        }
        for (BenchmarkReport workloadReport : suiteReport.workloadReports()) {
            if (!expectation.workloadName().equals(workloadReport.workloadName())) {
                continue;
            }
            for (BenchmarkCandidateReport candidate : workloadReport.candidates()) {
                if (candidate.measurement() == null) {
                    continue;
                }
                GpuCoverageSummary.BackendCoverage coverage = GpuCoverageSummary.fromTrace(candidate.measurement().trace())
                        .backends()
                        .get(expectation.backend());
                if (coverage != null) {
                    return coverage;
                }
            }
        }
        return null;
    }

    private static boolean hasExpectedVisibleReason(
            GpuCoverageSummary.BackendCoverage coverage,
            List<String> expectedVisibleReasons
    ) {
        String visibleReasons = String.join(
                " ",
                coverage.rejectedCandidateReasonCounts().keySet().toString(),
                coverage.cpuMaterializationReasonCounts().keySet().toString(),
                coverage.fallbackReasons().toString(),
                coverage.reasonCodes().toString()
        );
        return expectedVisibleReasons.stream()
                .filter(reason -> reason != null && !reason.isBlank())
                .anyMatch(visibleReasons::contains);
    }
}
