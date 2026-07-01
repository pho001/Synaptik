package tuning.benchmark.report;

/**
 * Stable coverage gap categories used by GPU coverage triage.
 */
public enum GpuCoverageGapCategory {
    REJECTED_CANDIDATE,
    CPU_MATERIALIZATION,
    TENSOR_ARRAY_FALLBACK,
    CPU_FALLBACK,
    DEVICE_HANDOFF,
    LOW_PARTITION_LENGTH,
    LOW_GPU_COVERAGE,
    STORAGE_RESIDENCY
}
