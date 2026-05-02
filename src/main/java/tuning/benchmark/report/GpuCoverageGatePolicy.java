package tuning.benchmark.report;

/**
 * Fail-fast policy for GPU coverage regression checks.
 *
 * @param backend backend name to check
 * @param minGpuCoverageRatio minimum executed accelerator-step ratio
 * @param minMaxSelectedRegionLength minimum selected region length
 * @param minMultiOpGpuRegionCount minimum selected multi-op GPU regions
 * @param minLoweredPrimitiveCount minimum lowered primitive count
 * @param minGpuFusedSubpatternCount minimum GPU fused subpattern count
 * @param maxCpuMaterializationCount maximum allowed CPU materialization boundaries
 * @param maxInternalCpuMaterializationCount maximum allowed avoidable internal CPU materialization boundaries
 * @param maxGradientPublicationMaterializationCount maximum allowed gradient-publication materialization boundaries
 * @param maxFallbackCount maximum allowed tensor-array plus CPU fallback count
 * @param maxTensorArrayStepCount maximum allowed tensor-array bridge steps
 * @param maxDeviceHandoffCount maximum allowed device handoffs
 * @param requireNativeBufferBinding whether at least one native buffer-binding step is required
 * @param requiredNativeCopyStrategy required native copy strategy, or blank when not checked
 */
public record GpuCoverageGatePolicy(
        String backend,
        double minGpuCoverageRatio,
        int minMaxSelectedRegionLength,
        int minMultiOpGpuRegionCount,
        int minLoweredPrimitiveCount,
        int minGpuFusedSubpatternCount,
        int maxCpuMaterializationCount,
        int maxInternalCpuMaterializationCount,
        int maxGradientPublicationMaterializationCount,
        int maxFallbackCount,
        int maxTensorArrayStepCount,
        int maxDeviceHandoffCount,
        boolean requireNativeBufferBinding,
        String requiredNativeCopyStrategy
) {
    public GpuCoverageGatePolicy(
            String backend,
            double minGpuCoverageRatio,
            int minMaxSelectedRegionLength,
            int minMultiOpGpuRegionCount,
            int minLoweredPrimitiveCount,
            int minGpuFusedSubpatternCount,
            int maxCpuMaterializationCount,
            int maxFallbackCount,
            int maxTensorArrayStepCount,
            int maxDeviceHandoffCount,
            boolean requireNativeBufferBinding
    ) {
        this(
                backend,
                minGpuCoverageRatio,
                minMaxSelectedRegionLength,
                minMultiOpGpuRegionCount,
                minLoweredPrimitiveCount,
                minGpuFusedSubpatternCount,
                maxCpuMaterializationCount,
                maxCpuMaterializationCount == Integer.MAX_VALUE ? Integer.MAX_VALUE : 0,
                maxCpuMaterializationCount == Integer.MAX_VALUE ? Integer.MAX_VALUE : 0,
                maxFallbackCount,
                maxTensorArrayStepCount,
                maxDeviceHandoffCount,
                requireNativeBufferBinding,
                ""
        );
    }

    public GpuCoverageGatePolicy {
        backend = backend == null ? "" : backend;
        minGpuCoverageRatio = Math.max(0.0d, minGpuCoverageRatio);
        minMaxSelectedRegionLength = Math.max(0, minMaxSelectedRegionLength);
        minMultiOpGpuRegionCount = Math.max(0, minMultiOpGpuRegionCount);
        minLoweredPrimitiveCount = Math.max(0, minLoweredPrimitiveCount);
        minGpuFusedSubpatternCount = Math.max(0, minGpuFusedSubpatternCount);
        maxCpuMaterializationCount = Math.max(0, maxCpuMaterializationCount);
        maxInternalCpuMaterializationCount = Math.max(0, maxInternalCpuMaterializationCount);
        maxGradientPublicationMaterializationCount = Math.max(0, maxGradientPublicationMaterializationCount);
        maxFallbackCount = Math.max(0, maxFallbackCount);
        maxTensorArrayStepCount = Math.max(0, maxTensorArrayStepCount);
        maxDeviceHandoffCount = Math.max(0, maxDeviceHandoffCount);
        requiredNativeCopyStrategy = requiredNativeCopyStrategy == null ? "" : requiredNativeCopyStrategy.trim();
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
                0,
                0,
                0,
                0,
                0,
                1,
                true,
                ""
        );
    }

    public static GpuCoverageGatePolicy hotPathTarget(
            String backend,
            double minGpuCoverageRatio,
            int minMaxSelectedRegionLength,
            int minMultiOpGpuRegionCount,
            int minLoweredPrimitiveCount,
            int minGpuFusedSubpatternCount
    ) {
        return new GpuCoverageGatePolicy(
                backend,
                minGpuCoverageRatio,
                minMaxSelectedRegionLength,
                minMultiOpGpuRegionCount,
                minLoweredPrimitiveCount,
                minGpuFusedSubpatternCount,
                0,
                0,
                0,
                0,
                0,
                1,
                true,
                ""
        );
    }

    public static GpuCoverageGatePolicy trainingHotPathTarget(
            String backend,
            double minGpuCoverageRatio,
            int minMaxSelectedRegionLength,
            int minMultiOpGpuRegionCount,
            int minLoweredPrimitiveCount,
            int minGpuFusedSubpatternCount,
            int maxGradientPublicationMaterializationCount
    ) {
        int gradientBudget = Math.max(0, maxGradientPublicationMaterializationCount);
        return new GpuCoverageGatePolicy(
                backend,
                minGpuCoverageRatio,
                minMaxSelectedRegionLength,
                minMultiOpGpuRegionCount,
                minLoweredPrimitiveCount,
                minGpuFusedSubpatternCount,
                gradientBudget,
                0,
                gradientBudget,
                0,
                0,
                Math.max(1, gradientBudget + 1),
                true,
                ""
        );
    }

    public GpuCoverageGatePolicy withRequiredNativeCopyStrategy(String strategy) {
        return new GpuCoverageGatePolicy(
                backend,
                minGpuCoverageRatio,
                minMaxSelectedRegionLength,
                minMultiOpGpuRegionCount,
                minLoweredPrimitiveCount,
                minGpuFusedSubpatternCount,
                maxCpuMaterializationCount,
                maxInternalCpuMaterializationCount,
                maxGradientPublicationMaterializationCount,
                maxFallbackCount,
                maxTensorArrayStepCount,
                maxDeviceHandoffCount,
                requireNativeBufferBinding,
                strategy
        );
    }
}
