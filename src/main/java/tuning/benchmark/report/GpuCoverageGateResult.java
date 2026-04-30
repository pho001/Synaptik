package tuning.benchmark.report;

import java.util.List;

/**
 * Result of evaluating a GPU coverage regression gate.
 *
 * @param passed whether coverage satisfies the policy
 * @param failures stable failure strings
 * @param coverage backend coverage used for evaluation, or null when missing
 */
public record GpuCoverageGateResult(
        boolean passed,
        List<String> failures,
        GpuCoverageSummary.BackendCoverage coverage
) {
    public GpuCoverageGateResult {
        failures = failures == null ? List.of() : List.copyOf(failures);
    }
}
