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
        String fallbackReason
) {
    public PreparedNativeCpuPlan {
        route = route == null ? PreparedNativeCpuRoute.NONE : route;
        inputPolicy = inputPolicy == null ? PreparedNativeCpuInputPolicy.ALL_CPU : inputPolicy;
        requestedStorage = requestedStorage == null ? CpuStorageProfile.CPU_ARRAY : requestedStorage;
        fallbackReason = fallbackReason == null ? "" : fallbackReason;
        if (route != PreparedNativeCpuRoute.NONE && coverageEntry == null) {
            throw new IllegalArgumentException("coverageEntry is required for native CPU route " + route);
        }
        if ((route == PreparedNativeCpuRoute.NONE || route == PreparedNativeCpuRoute.FALLBACK_ONLY)
                && inputPolicy != PreparedNativeCpuInputPolicy.ALL_CPU) {
            throw new IllegalArgumentException("non-native routes must use ALL_CPU input policy");
        }
        Objects.requireNonNull(inputPolicy, "inputPolicy cannot be null");
    }

    public static PreparedNativeCpuPlan none(CpuStorageProfile requestedStorage, String reason) {
        return new PreparedNativeCpuPlan(
                PreparedNativeCpuRoute.NONE,
                PreparedNativeCpuInputPolicy.ALL_CPU,
                null,
                requestedStorage,
                reason
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
                reason
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
                ""
        );
    }

    public boolean allowsNativeInputs() {
        return inputPolicy == PreparedNativeCpuInputPolicy.ALL_NATIVE
                || inputPolicy == PreparedNativeCpuInputPolicy.CONDITION_CPU_VALUES_NATIVE;
    }
}
