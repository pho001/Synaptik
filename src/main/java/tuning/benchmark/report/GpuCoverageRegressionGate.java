package tuning.benchmark.report;

import java.util.ArrayList;

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
        if (coverage.cpuMaterializationCount() > policy.maxCpuMaterializationCount()) {
            failures.add("unexpected CPU materialization");
        }
        if (coverage.fallbackCount() > policy.maxFallbackCount()) {
            failures.add("lost GPU coverage");
        }
        if (coverage.tensorArrayStepCount() > policy.maxTensorArrayStepCount()) {
            failures.add("hidden tensor-array fallback");
        }
        if (policy.requireNativeBufferBinding() && coverage.bufferBindingStepCount() == 0) {
            failures.add("lost GPU coverage");
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
}
