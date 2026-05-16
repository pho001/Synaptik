package backend.cpu.nativecpu;

import config.runtime.CpuStorageProfile;

import java.util.Objects;

/**
 * Prepare-time native CPU decision for one CPU node.
 */
public record PreparedNativeCpuPlan(
        PreparedNativeCpuRoute route,
        PreparedNativeCpuInputPolicy inputPolicy,
        NativeCpuCoverageEntry coverageEntry,
        CpuStorageProfile requestedStorage,
        String fallbackReason,
        int chainSegmentId,
        NativeCpuChainDecision chainDecision,
        String chainReason
) {
    public PreparedNativeCpuPlan {
        route = route == null ? PreparedNativeCpuRoute.NONE : route;
        inputPolicy = inputPolicy == null ? PreparedNativeCpuInputPolicy.ALL_CPU : inputPolicy;
        requestedStorage = requestedStorage == null ? CpuStorageProfile.CPU_ARRAY : requestedStorage;
        fallbackReason = fallbackReason == null ? "" : fallbackReason;
        chainSegmentId = Math.max(-1, chainSegmentId);
        chainDecision = chainDecision == null ? NativeCpuChainDecision.NONE : chainDecision;
        chainReason = chainReason == null ? "" : chainReason;
        if (route != PreparedNativeCpuRoute.NONE && coverageEntry == null) {
            throw new IllegalArgumentException("coverageEntry is required for native CPU route " + route);
        }
        if ((route == PreparedNativeCpuRoute.NONE || route == PreparedNativeCpuRoute.FALLBACK_ONLY)
                && inputPolicy != PreparedNativeCpuInputPolicy.ALL_CPU) {
            throw new IllegalArgumentException("non-native routes must use ALL_CPU input policy");
        }
        Objects.requireNonNull(inputPolicy, "inputPolicy cannot be null");
    }

    public PreparedNativeCpuPlan(
            PreparedNativeCpuRoute route,
            PreparedNativeCpuInputPolicy inputPolicy,
            NativeCpuCoverageEntry coverageEntry,
            CpuStorageProfile requestedStorage,
            String fallbackReason
    ) {
        this(route, inputPolicy, coverageEntry, requestedStorage, fallbackReason, -1, NativeCpuChainDecision.NONE, "");
    }

    public static PreparedNativeCpuPlan none(CpuStorageProfile requestedStorage, String reason) {
        return new PreparedNativeCpuPlan(
                PreparedNativeCpuRoute.NONE,
                PreparedNativeCpuInputPolicy.ALL_CPU,
                null,
                requestedStorage,
                reason,
                -1,
                NativeCpuChainDecision.NONE,
                ""
        );
    }

    public static PreparedNativeCpuPlan fallbackOnly(
            NativeCpuCoverageEntry coverageEntry,
            CpuStorageProfile requestedStorage,
            String reason
    ) {
        return new PreparedNativeCpuPlan(
                PreparedNativeCpuRoute.FALLBACK_ONLY,
                PreparedNativeCpuInputPolicy.ALL_CPU,
                coverageEntry,
                requestedStorage,
                reason,
                -1,
                NativeCpuChainDecision.NONE,
                ""
        );
    }

    public static PreparedNativeCpuPlan nativeExecutable(
            NativeCpuCoverageEntry coverageEntry,
            CpuStorageProfile requestedStorage
    ) {
        return new PreparedNativeCpuPlan(
                PreparedNativeCpuRoute.NATIVE_EXECUTABLE,
                PreparedNativeCpuInputPolicy.ALL_NATIVE,
                coverageEntry,
                requestedStorage,
                "",
                -1,
                NativeCpuChainDecision.NONE,
                ""
        );
    }

    public static PreparedNativeCpuPlan conditionArrayInputNativeOutput(
            NativeCpuCoverageEntry coverageEntry,
            CpuStorageProfile requestedStorage
    ) {
        return new PreparedNativeCpuPlan(
                PreparedNativeCpuRoute.CONDITION_ARRAY_INPUT_NATIVE_OUTPUT,
                PreparedNativeCpuInputPolicy.CONDITION_CPU_VALUES_NATIVE,
                coverageEntry,
                requestedStorage,
                "",
                -1,
                NativeCpuChainDecision.NONE,
                ""
        );
    }

    public static PreparedNativeCpuPlan viewAlias(
            NativeCpuCoverageEntry coverageEntry,
            CpuStorageProfile requestedStorage
    ) {
        return new PreparedNativeCpuPlan(
                PreparedNativeCpuRoute.VIEW_ALIAS,
                PreparedNativeCpuInputPolicy.ALL_NATIVE,
                coverageEntry,
                requestedStorage,
                "",
                -1,
                NativeCpuChainDecision.NONE,
                ""
        );
    }

    public boolean allowsNativeInputs() {
        return inputPolicy == PreparedNativeCpuInputPolicy.ALL_NATIVE
                || inputPolicy == PreparedNativeCpuInputPolicy.CONDITION_CPU_VALUES_NATIVE;
    }

    public PreparedNativeCpuPlan withChain(int segmentId, NativeCpuChainDecision decision, String reason) {
        return new PreparedNativeCpuPlan(
                route,
                inputPolicy,
                coverageEntry,
                requestedStorage,
                fallbackReason,
                segmentId,
                decision,
                reason
        );
    }

    public PreparedNativeCpuPlan withRoute(
            PreparedNativeCpuRoute newRoute,
            PreparedNativeCpuInputPolicy newInputPolicy,
            NativeCpuCoverageEntry newCoverageEntry,
            String newFallbackReason
    ) {
        return new PreparedNativeCpuPlan(
                newRoute,
                newInputPolicy,
                newCoverageEntry,
                requestedStorage,
                newFallbackReason,
                chainSegmentId,
                chainDecision,
                chainReason
        );
    }
}
