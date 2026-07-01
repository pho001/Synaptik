package tuning.benchmark.report;

/**
 * Fail-fast policy for GPU coverage regression checks.
 *
 * @param backend backend name to check
 * @param minGpuCoverageRatio minimum executed accelerator-step ratio
 * @param minMaxSelectedPartitionLength minimum selected partition length
 * @param minMultiOpGpuPartitionCount minimum selected multi-op GPU partitions
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
        int minMaxSelectedPartitionLength,
        int minMultiOpGpuPartitionCount,
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
            int minMaxSelectedPartitionLength,
            int minMultiOpGpuPartitionCount,
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
                minMaxSelectedPartitionLength,
                minMultiOpGpuPartitionCount,
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
        minMaxSelectedPartitionLength = Math.max(0, minMaxSelectedPartitionLength);
        minMultiOpGpuPartitionCount = Math.max(0, minMultiOpGpuPartitionCount);
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
            int minMaxSelectedPartitionLength
    ) {
        return new GpuCoverageGatePolicy(
                backend,
                minGpuCoverageRatio,
                minMaxSelectedPartitionLength,
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

    public static GpuCoverageGatePolicy reportNativeBufferTarget(
            String backend,
            GpuCoverageSummary.BackendCoverage coverage
    ) {
        int publicationMaterializations = coverage == null ? 0 : coverage.publicationCpuMaterializationCount();
        int gradientPublications = coverage == null ? 0 : coverage.gradientPublicationMaterializationCount();
        int deviceHandoffBudget = Math.max(1, publicationMaterializations + 1);
        return new GpuCoverageGatePolicy(
                backend,
                0.0d,
                0,
                0,
                0,
                0,
                publicationMaterializations,
                0,
                gradientPublications,
                0,
                0,
                deviceHandoffBudget,
                true,
                ""
        );
    }

    public static GpuCoverageGatePolicy hotPathTarget(
            String backend,
            double minGpuCoverageRatio,
            int minMaxSelectedPartitionLength,
            int minMultiOpGpuPartitionCount,
            int minLoweredPrimitiveCount,
            int minGpuFusedSubpatternCount
    ) {
        return new GpuCoverageGatePolicy(
                backend,
                minGpuCoverageRatio,
                minMaxSelectedPartitionLength,
                minMultiOpGpuPartitionCount,
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
            int minMaxSelectedPartitionLength,
            int minMultiOpGpuPartitionCount,
            int minLoweredPrimitiveCount,
            int minGpuFusedSubpatternCount,
            int maxGradientPublicationMaterializationCount
    ) {
        int gradientBudget = Math.max(0, maxGradientPublicationMaterializationCount);
        return new GpuCoverageGatePolicy(
                backend,
                minGpuCoverageRatio,
                minMaxSelectedPartitionLength,
                minMultiOpGpuPartitionCount,
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
                minMaxSelectedPartitionLength,
                minMultiOpGpuPartitionCount,
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
