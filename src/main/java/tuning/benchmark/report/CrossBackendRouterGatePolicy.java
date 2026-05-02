package tuning.benchmark.report;

import java.util.Set;

/**
 * Gate policy for cross-backend router evidence.
 *
 * @param backend backend name to evaluate
 * @param maxTensorArrayStepCount maximum allowed tensor-array replay steps
 * @param maxCpuFallbackStepCount maximum allowed accelerator CPU fallback steps
 * @param maxCpuMaterializationCount maximum allowed CPU materialization boundaries
 * @param maxInternalCpuMaterializationCount maximum allowed avoidable internal CPU materializations
 * @param maxDeviceHandoffCount maximum allowed device handoffs
 * @param requireNativeBufferBinding whether at least one native buffer-binding step is required
 * @param minMaxSelectedRegionLength minimum selected region length
 * @param minLoweredPrimitiveCount minimum lowered primitive count
 * @param requiredRoutes required backend route or common accelerator path names
 * @param requiredVisibleReasons required visible reason substrings across rejected routes/fallbacks/reasons
 * @param requiredNativeCopyStrategies required native copy strategies
 * @param allowedNativeCopyStrategies allowed native copy strategies, or empty when unchecked
 * @param requiredOutputBufferWriteStatuses required output-buffer write statuses
 * @param allowedOutputBufferWriteStatuses allowed output-buffer write statuses, or empty when unchecked
 * @param rejectUnsupportedRouteOverclaims whether unsupported/copy overclaims should fail the gate
 */
public record CrossBackendRouterGatePolicy(
        String backend,
        int maxTensorArrayStepCount,
        int maxCpuFallbackStepCount,
        int maxCpuMaterializationCount,
        int maxInternalCpuMaterializationCount,
        int maxDeviceHandoffCount,
        boolean requireNativeBufferBinding,
        int minMaxSelectedRegionLength,
        int minLoweredPrimitiveCount,
        Set<String> requiredRoutes,
        Set<String> requiredVisibleReasons,
        Set<String> requiredNativeCopyStrategies,
        Set<String> allowedNativeCopyStrategies,
        Set<String> requiredOutputBufferWriteStatuses,
        Set<String> allowedOutputBufferWriteStatuses,
        boolean rejectUnsupportedRouteOverclaims
) {
    public CrossBackendRouterGatePolicy {
        backend = backend == null ? "" : backend.trim();
        maxTensorArrayStepCount = Math.max(0, maxTensorArrayStepCount);
        maxCpuFallbackStepCount = Math.max(0, maxCpuFallbackStepCount);
        maxCpuMaterializationCount = Math.max(0, maxCpuMaterializationCount);
        maxInternalCpuMaterializationCount = Math.max(0, maxInternalCpuMaterializationCount);
        maxDeviceHandoffCount = Math.max(0, maxDeviceHandoffCount);
        minMaxSelectedRegionLength = Math.max(0, minMaxSelectedRegionLength);
        minLoweredPrimitiveCount = Math.max(0, minLoweredPrimitiveCount);
        requiredRoutes = clean(requiredRoutes);
        requiredVisibleReasons = clean(requiredVisibleReasons);
        requiredNativeCopyStrategies = clean(requiredNativeCopyStrategies);
        allowedNativeCopyStrategies = clean(allowedNativeCopyStrategies);
        requiredOutputBufferWriteStatuses = clean(requiredOutputBufferWriteStatuses);
        allowedOutputBufferWriteStatuses = clean(allowedOutputBufferWriteStatuses);
    }

    public static CrossBackendRouterGatePolicy nativeHotPath(String backend) {
        return new CrossBackendRouterGatePolicy(
                backend,
                0,
                0,
                0,
                0,
                1,
                true,
                1,
                1,
                Set.of("BUFFER_BINDING"),
                Set.of(),
                Set.of(),
                Set.of(),
                Set.of(),
                Set.of(),
                true
        );
    }

    public CrossBackendRouterGatePolicy withRequiredRoutes(Set<String> routes) {
        return new CrossBackendRouterGatePolicy(
                backend,
                maxTensorArrayStepCount,
                maxCpuFallbackStepCount,
                maxCpuMaterializationCount,
                maxInternalCpuMaterializationCount,
                maxDeviceHandoffCount,
                requireNativeBufferBinding,
                minMaxSelectedRegionLength,
                minLoweredPrimitiveCount,
                routes,
                requiredVisibleReasons,
                requiredNativeCopyStrategies,
                allowedNativeCopyStrategies,
                requiredOutputBufferWriteStatuses,
                allowedOutputBufferWriteStatuses,
                rejectUnsupportedRouteOverclaims
        );
    }

    public CrossBackendRouterGatePolicy withRequiredVisibleReasons(Set<String> reasons) {
        return new CrossBackendRouterGatePolicy(
                backend,
                maxTensorArrayStepCount,
                maxCpuFallbackStepCount,
                maxCpuMaterializationCount,
                maxInternalCpuMaterializationCount,
                maxDeviceHandoffCount,
                requireNativeBufferBinding,
                minMaxSelectedRegionLength,
                minLoweredPrimitiveCount,
                requiredRoutes,
                reasons,
                requiredNativeCopyStrategies,
                allowedNativeCopyStrategies,
                requiredOutputBufferWriteStatuses,
                allowedOutputBufferWriteStatuses,
                rejectUnsupportedRouteOverclaims
        );
    }

    public CrossBackendRouterGatePolicy withNativeCopyStrategies(Set<String> required, Set<String> allowed) {
        return new CrossBackendRouterGatePolicy(
                backend,
                maxTensorArrayStepCount,
                maxCpuFallbackStepCount,
                maxCpuMaterializationCount,
                maxInternalCpuMaterializationCount,
                maxDeviceHandoffCount,
                requireNativeBufferBinding,
                minMaxSelectedRegionLength,
                minLoweredPrimitiveCount,
                requiredRoutes,
                requiredVisibleReasons,
                required,
                allowed,
                requiredOutputBufferWriteStatuses,
                allowedOutputBufferWriteStatuses,
                rejectUnsupportedRouteOverclaims
        );
    }

    public CrossBackendRouterGatePolicy withOutputBufferWriteStatuses(Set<String> required, Set<String> allowed) {
        return new CrossBackendRouterGatePolicy(
                backend,
                maxTensorArrayStepCount,
                maxCpuFallbackStepCount,
                maxCpuMaterializationCount,
                maxInternalCpuMaterializationCount,
                maxDeviceHandoffCount,
                requireNativeBufferBinding,
                minMaxSelectedRegionLength,
                minLoweredPrimitiveCount,
                requiredRoutes,
                requiredVisibleReasons,
                requiredNativeCopyStrategies,
                allowedNativeCopyStrategies,
                required,
                allowed,
                rejectUnsupportedRouteOverclaims
        );
    }

    private static Set<String> clean(Set<String> values) {
        if (values == null || values.isEmpty()) {
            return Set.of();
        }
        return values.stream()
                .filter(value -> value != null && !value.isBlank())
                .map(String::trim)
                .collect(java.util.stream.Collectors.toUnmodifiableSet());
    }
}
