package tuning.benchmark.report;

import java.util.List;

/**
 * Hard or partial coverage expectation for a selected v1.3 GPU hot-path workload.
 *
 * @param workloadName benchmark workload name
 * @param backend accelerator backend name
 * @param policy coverage gate policy to evaluate when native evidence is expected
 * @param expectedVisibleReasons stable visible blockers accepted for partially supported targets
 * @param nativeEvidenceRequired whether this target requires positive native GPU evidence
 */
public record GpuCoverageHotPathExpectation(
        String workloadName,
        String backend,
        GpuCoverageGatePolicy policy,
        List<String> expectedVisibleReasons,
        boolean nativeEvidenceRequired
) {
    public GpuCoverageHotPathExpectation {
        workloadName = workloadName == null ? "" : workloadName;
        backend = backend == null ? "" : backend;
        policy = policy == null ? GpuCoverageGatePolicy.nativeBufferTarget(backend, 0.0d, 0) : policy;
        expectedVisibleReasons = expectedVisibleReasons == null ? List.of() : List.copyOf(expectedVisibleReasons);
    }
}
