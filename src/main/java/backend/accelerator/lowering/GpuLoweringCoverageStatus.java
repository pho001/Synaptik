package backend.accelerator.lowering;

/**
 * Source-level coverage status for GPU lowering matrix rows.
 */
public enum GpuLoweringCoverageStatus {
    SUPPORTED,
    FALLBACK,
    UNSUPPORTED
}
