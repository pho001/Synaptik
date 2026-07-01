package tuning.benchmark.report;

import java.util.List;

/**
 * v1.3 hot-path workload target for coverage-driven GPU partition expansion.
 *
 * @param workloadName standard workload name
 * @param targetKind coarse target family
 * @param requirementFamilies downstream requirement families exercised by the target
 * @param ownerPhase downstream phase that primarily owns the target
 * @param rationale short reason this workload is a target
 */
public record GpuHotPathCoverageTarget(
        String workloadName,
        String targetKind,
        List<String> requirementFamilies,
        int ownerPhase,
        String rationale
) {
    public GpuHotPathCoverageTarget {
        workloadName = workloadName == null || workloadName.isBlank() ? "unknown" : workloadName;
        targetKind = targetKind == null || targetKind.isBlank() ? "generic" : targetKind;
        requirementFamilies = requirementFamilies == null ? List.of() : List.copyOf(requirementFamilies);
        ownerPhase = Math.max(0, ownerPhase);
        rationale = rationale == null || rationale.isBlank() ? "unspecified" : rationale;
    }
}
