package backend.cpu.nativecpu;

import operations.Operation;
import tensor.DataType;

import java.util.Objects;

/**
 * Single operation/dtype row in the native CPU coverage matrix.
 */
public record NativeCpuCoverageEntry(
        Operation.OpType opType,
        DataType dataType,
        NativeCpuCoverageLayoutScope layoutScope,
        NativeCpuKernelPerformanceStatus status,
        NativeCpuKernelFamily family,
        boolean nativeSupported,
        boolean preservesNativeStorage,
        String fallbackReason
) {
    public NativeCpuCoverageEntry {
        opType = Objects.requireNonNull(opType, "opType cannot be null");
        dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        layoutScope = Objects.requireNonNull(layoutScope, "layoutScope cannot be null");
        status = Objects.requireNonNull(status, "status cannot be null");
        family = Objects.requireNonNull(family, "family cannot be null");
        fallbackReason = fallbackReason == null ? "" : fallbackReason;
    }

    /**
     * Returns whether this row is eligible for default AUTO native selection without benchmark evidence.
     */
    public boolean autoFastEligible() {
        return status == NativeCpuKernelPerformanceStatus.NATIVE_FAST
                || status == NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER
                || status == NativeCpuKernelPerformanceStatus.VIEW_ONLY;
    }
}
