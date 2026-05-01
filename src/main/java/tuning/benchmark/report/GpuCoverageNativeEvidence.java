package tuning.benchmark.report;

/**
 * Native accelerator evidence status for GPU coverage reports.
 *
 * @param backend accelerator backend name
 * @param nativeStatus one of passed, capabilitySkipped, or failed
 * @param detail stable human-readable evidence detail
 */
public record GpuCoverageNativeEvidence(
        String backend,
        String nativeStatus,
        String detail
) {
    public GpuCoverageNativeEvidence {
        backend = backend == null ? "" : backend;
        nativeStatus = nativeStatus == null ? "" : nativeStatus;
        detail = detail == null ? "" : detail;
    }

    public static GpuCoverageNativeEvidence passed(String backend, String detail) {
        return new GpuCoverageNativeEvidence(backend, "passed", detail);
    }

    public static GpuCoverageNativeEvidence capabilitySkipped(String backend, String detail) {
        return new GpuCoverageNativeEvidence(backend, "capabilitySkipped", detail);
    }

    public static GpuCoverageNativeEvidence failed(String backend, String detail) {
        return new GpuCoverageNativeEvidence(backend, "failed", detail);
    }
}
