package tuning.benchmark.report;

/**
 * Stable coverage baseline for comparing GPU coverage behavior without timing thresholds.
 *
 * @param baselineName baseline label, for example {@code v1.1}
 * @param backend accelerator backend name
 * @param maxSelectedRegionLength baseline maximum selected GPU region length
 * @param cpuMaterializationCount baseline CPU materialization count
 * @param fallbackCount baseline tensor-array plus CPU-fallback count
 * @param deviceHandoffCount baseline device handoff count
 */
public record GpuCoverageBaseline(
        String baselineName,
        String backend,
        int maxSelectedRegionLength,
        int cpuMaterializationCount,
        int fallbackCount,
        int deviceHandoffCount
) {
    public GpuCoverageBaseline {
        baselineName = baselineName == null || baselineName.isBlank() ? "baseline" : baselineName;
        backend = backend == null ? "" : backend;
        maxSelectedRegionLength = Math.max(0, maxSelectedRegionLength);
        cpuMaterializationCount = Math.max(0, cpuMaterializationCount);
        fallbackCount = Math.max(0, fallbackCount);
        deviceHandoffCount = Math.max(0, deviceHandoffCount);
    }
}
