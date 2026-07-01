package tuning.benchmark.report;

/**
 * Ranked GPU coverage gap for one workload, candidate, backend, and reason.
 *
 * @param workloadName workload that produced the gap
 * @param candidateName candidate/profile name that produced the gap
 * @param backend accelerator backend name
 * @param category stable gap category
 * @param reason stable reason string
 * @param count raw count contributing to the gap
 * @param severityScore deterministic triage score
 * @param maxSelectedPartitionLength maximum selected partition length from coverage evidence
 * @param cpuMaterializationCount raw CPU materialization count
 * @param fallbackCount raw fallback count
 * @param deviceHandoffCount raw device handoff count
 * @param requirementFamily downstream v1.3 requirement family
 */
public record GpuCoverageGap(
        String workloadName,
        String candidateName,
        String backend,
        GpuCoverageGapCategory category,
        String reason,
        int count,
        int severityScore,
        int maxSelectedPartitionLength,
        int cpuMaterializationCount,
        int fallbackCount,
        int deviceHandoffCount,
        String requirementFamily
) {
    public GpuCoverageGap {
        workloadName = workloadName == null ? "" : workloadName;
        candidateName = candidateName == null ? "" : candidateName;
        backend = backend == null ? "" : backend;
        category = category == null ? GpuCoverageGapCategory.REJECTED_CANDIDATE : category;
        reason = reason == null || reason.isBlank() ? "unspecified" : reason;
        count = Math.max(0, count);
        severityScore = Math.max(0, severityScore);
        maxSelectedPartitionLength = Math.max(0, maxSelectedPartitionLength);
        cpuMaterializationCount = Math.max(0, cpuMaterializationCount);
        fallbackCount = Math.max(0, fallbackCount);
        deviceHandoffCount = Math.max(0, deviceHandoffCount);
        requirementFamily = requirementFamily == null || requirementFamily.isBlank()
                ? "GPUHARDEN"
                : requirementFamily;
    }
}
