package tuning.benchmark.report;

/**
 * Fail-fast policy for GPU coverage regression checks.
 *
 * @param backend backend name to check
 * @param minGpuCoverageRatio minimum executed accelerator-step ratio
 * @param minMaxSelectedRegionLength minimum selected region length
 * @param maxCpuMaterializationCount maximum allowed CPU materialization boundaries
 * @param maxFallbackCount maximum allowed tensor-array plus CPU fallback count
 * @param maxTensorArrayStepCount maximum allowed tensor-array bridge steps
 * @param maxDeviceHandoffCount maximum allowed device handoffs
 * @param requireNativeBufferBinding whether at least one native buffer-binding step is required
 */
public record GpuCoverageGatePolicy(
        String backend,
        double minGpuCoverageRatio,
        int minMaxSelectedRegionLength,
        int maxCpuMaterializationCount,
        int maxFallbackCount,
        int maxTensorArrayStepCount,
        int maxDeviceHandoffCount,
        boolean requireNativeBufferBinding
) {
    public GpuCoverageGatePolicy {
        backend = backend == null ? "" : backend;
        minGpuCoverageRatio = Math.max(0.0d, minGpuCoverageRatio);
        minMaxSelectedRegionLength = Math.max(0, minMaxSelectedRegionLength);
        maxCpuMaterializationCount = Math.max(0, maxCpuMaterializationCount);
        maxFallbackCount = Math.max(0, maxFallbackCount);
        maxTensorArrayStepCount = Math.max(0, maxTensorArrayStepCount);
        maxDeviceHandoffCount = Math.max(0, maxDeviceHandoffCount);
    }

    public static GpuCoverageGatePolicy nativeBufferTarget(
            String backend,
            double minGpuCoverageRatio,
            int minMaxSelectedRegionLength
    ) {
        return new GpuCoverageGatePolicy(
                backend,
                minGpuCoverageRatio,
                minMaxSelectedRegionLength,
                0,
                0,
                0,
                1,
                true
        );
    }
}
