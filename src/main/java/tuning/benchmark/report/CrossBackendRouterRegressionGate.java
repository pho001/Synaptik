package tuning.benchmark.report;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Fail-fast evaluator for backend router calibration and representative workload gates.
 */
public final class CrossBackendRouterRegressionGate {
    private CrossBackendRouterRegressionGate() {
    }

    public static CrossBackendRouterGateResult evaluate(
            CrossBackendRouterEvidence evidence,
            CrossBackendRouterGatePolicy policy
    ) {
        if (policy == null) {
            policy = CrossBackendRouterGatePolicy.nativeHotPath("");
        }
        var failures = new ArrayList<String>();
        CrossBackendRouterEvidence.BackendEvidence backendEvidence = evidence == null
                ? null
                : evidence.backends().get(policy.backend());
        if (backendEvidence == null) {
            failures.add("missing cross-backend router evidence");
            return new CrossBackendRouterGateResult(false, failures, null);
        }

        if (backendEvidence.tensorArrayStepCount() > policy.maxTensorArrayStepCount()) {
            failures.add("hidden tensor-array replay");
        }
        if (backendEvidence.cpuFallbackStepCount() > policy.maxCpuFallbackStepCount()) {
            failures.add("unexpected CPU fallback");
        }
        if (backendEvidence.cpuMaterializationCount() > policy.maxCpuMaterializationCount()) {
            failures.add("unexpected CPU materialization");
        }
        if (backendEvidence.internalCpuMaterializationCount() > policy.maxInternalCpuMaterializationCount()) {
            failures.add("unexpected internal CPU materialization");
        }
        if (backendEvidence.deviceHandoffCount() > policy.maxDeviceHandoffCount()) {
            failures.add("unexpected device handoff");
        }
        if (policy.requireNativeBufferBinding() && backendEvidence.bufferBindingStepCount() == 0) {
            failures.add("lost native buffer binding");
        }
        if (backendEvidence.maxSelectedRegionLength() < policy.minMaxSelectedRegionLength()) {
            failures.add("lost selected region length");
        }
        if (backendEvidence.loweredPrimitiveCount() < policy.minLoweredPrimitiveCount()) {
            failures.add("lost lowered primitive coverage");
        }
        for (String requiredRoute : policy.requiredRoutes()) {
            if (!backendEvidence.hasRouteOrPath(requiredRoute)) {
                failures.add("missing required selected route " + requiredRoute);
            }
        }
        for (String requiredReason : policy.requiredVisibleReasons()) {
            if (!backendEvidence.hasTextEvidence(requiredReason)) {
                failures.add("missing required visible reason " + requiredReason);
            }
        }
        addMissingRequiredSetFailures(
                failures,
                "missing required native copy strategy ",
                policy.requiredNativeCopyStrategies(),
                backendEvidence.nativeCopyStrategyCounts()
        );
        addUnexpectedSetFailures(
                failures,
                "unexpected native copy strategy ",
                policy.allowedNativeCopyStrategies(),
                backendEvidence.nativeCopyStrategyCounts()
        );
        addMissingRequiredSetFailures(
                failures,
                "missing required output-buffer write status ",
                policy.requiredOutputBufferWriteStatuses(),
                backendEvidence.outputBufferWriteStatusCounts()
        );
        addUnexpectedSetFailures(
                failures,
                "unexpected output-buffer write status ",
                policy.allowedOutputBufferWriteStatuses(),
                backendEvidence.outputBufferWriteStatusCounts()
        );
        if (policy.rejectUnsupportedRouteOverclaims() && hasUnsupportedRouteOverclaim(backendEvidence)) {
            failures.add("unsupported route overclaim");
        }
        return new CrossBackendRouterGateResult(failures.isEmpty(), failures, backendEvidence);
    }

    public static void requirePass(CrossBackendRouterEvidence evidence, CrossBackendRouterGatePolicy policy) {
        CrossBackendRouterGateResult result = evaluate(evidence, policy);
        if (!result.passed()) {
            throw new IllegalStateException(String.join("; ", result.failures()));
        }
    }

    public static List<CrossBackendRouterGateResult> evaluateTargets(
            BenchmarkSuiteReport suiteReport,
            List<CrossBackendRouterWorkloadExpectation> expectations
    ) {
        if (expectations == null || expectations.isEmpty()) {
            return List.of();
        }
        var results = new ArrayList<CrossBackendRouterGateResult>();
        for (CrossBackendRouterWorkloadExpectation expectation : expectations) {
            CrossBackendRouterGateResult result = evaluateTarget(suiteReport, expectation);
            results.add(result);
        }
        return List.copyOf(results);
    }

    public static void requireTargetsPass(
            BenchmarkSuiteReport suiteReport,
            List<CrossBackendRouterWorkloadExpectation> expectations
    ) {
        List<String> failures = evaluateTargets(suiteReport, expectations).stream()
                .filter(result -> !result.passed())
                .flatMap(result -> result.failures().stream())
                .toList();
        if (!failures.isEmpty()) {
            throw new IllegalStateException(String.join("; ", failures));
        }
    }

    private static CrossBackendRouterGateResult evaluateTarget(
            BenchmarkSuiteReport suiteReport,
            CrossBackendRouterWorkloadExpectation expectation
    ) {
        if (suiteReport == null || expectation == null || expectation.policy() == null) {
            return new CrossBackendRouterGateResult(false, List.of("missing workload router expectation"), null);
        }
        for (BenchmarkReport workloadReport : suiteReport.workloadReports()) {
            if (!expectation.workloadName().equals(workloadReport.workloadName())) {
                continue;
            }
            for (BenchmarkCandidateReport candidate : workloadReport.candidates()) {
                if (candidate.measurement() == null) {
                    continue;
                }
                CrossBackendRouterGateResult result = evaluate(
                        CrossBackendRouterEvidence.fromTrace(candidate.measurement().trace()),
                        expectation.policy()
                );
                if (!result.passed()) {
                    var failures = new ArrayList<String>();
                    for (String failure : result.failures()) {
                        failures.add("workload=" + expectation.workloadName()
                                + " backend=" + expectation.policy().backend()
                                + " " + failure);
                    }
                    return new CrossBackendRouterGateResult(false, failures, result.evidence());
                }
                return result;
            }
        }
        return new CrossBackendRouterGateResult(
                false,
                List.of("missing router target workload="
                        + expectation.workloadName()
                        + " backend="
                        + expectation.policy().backend()),
                null
        );
    }

    private static void addMissingRequiredSetFailures(
            List<String> failures,
            String prefix,
            Set<String> required,
            Map<String, Integer> observed
    ) {
        for (String value : required) {
            if (!observed.containsKey(value)) {
                failures.add(prefix + value);
            }
        }
    }

    private static void addUnexpectedSetFailures(
            List<String> failures,
            String prefix,
            Set<String> allowed,
            Map<String, Integer> observed
    ) {
        if (allowed.isEmpty()) {
            return;
        }
        for (String value : observed.keySet()) {
            if (!allowed.contains(value)) {
                failures.add(prefix + value);
            }
        }
    }

    private static boolean hasUnsupportedRouteOverclaim(CrossBackendRouterEvidence.BackendEvidence evidence) {
        if (evidence.backendRouteCounts().containsKey("MPS_GRAPH")
                && (evidence.nativeCopyStrategyCounts().containsKey("TRUE_OUTPUT_BUFFER_WRITE")
                || evidence.outputBufferWriteStatusCounts().containsKey("PROVEN_TRUE_WRITE"))) {
            return true;
        }
        if ("GPU_CUDA".equals(evidence.backend())
                && evidence.hasSupportClaim()
                && evidence.hasTextEvidence("CAPABILITY_MISSING")) {
            return true;
        }
        return evidence.hasSupportClaim()
                && (evidence.hasTextEvidence("UNSUPPORTED_ROUTE_OVERCLAIM")
                || evidence.hasTextEvidence("ROUTE_UNSUPPORTED"));
    }
}
