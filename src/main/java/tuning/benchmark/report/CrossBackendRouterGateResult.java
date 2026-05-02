package tuning.benchmark.report;

import java.util.List;

/**
 * Result of evaluating one cross-backend router evidence gate.
 *
 * @param passed whether the evidence satisfies the policy
 * @param failures stable failure messages
 * @param evidence evaluated backend evidence, or null when missing
 */
public record CrossBackendRouterGateResult(
        boolean passed,
        List<String> failures,
        CrossBackendRouterEvidence.BackendEvidence evidence
) {
    public CrossBackendRouterGateResult {
        failures = failures == null ? List.of() : List.copyOf(failures);
    }
}
