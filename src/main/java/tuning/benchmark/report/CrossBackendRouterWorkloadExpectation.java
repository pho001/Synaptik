package tuning.benchmark.report;

/**
 * Workload-level router evidence expectation for benchmark suite gates.
 *
 * @param workloadName benchmark workload name
 * @param policy router gate policy
 */
public record CrossBackendRouterWorkloadExpectation(
        String workloadName,
        CrossBackendRouterGatePolicy policy
) {
    public CrossBackendRouterWorkloadExpectation {
        workloadName = workloadName == null ? "" : workloadName;
    }
}
