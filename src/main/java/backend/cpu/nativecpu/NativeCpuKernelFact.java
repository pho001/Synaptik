package backend.cpu.nativecpu;

import operations.Operation;
import tensor.DataType;

import java.util.Objects;

/**
 * Planner-visible native CPU coverage fact for one operation/dtype pair.
 *
 * @param opType semantic operation
 * @param dataType logical output dtype considered by the planner
 * @param status native coverage/performance status
 * @param family physical execution family, or {@code ARRAY_ONLY} when native is unavailable
 * @param reason stable machine-readable reason or condition
 */
public record NativeCpuKernelFact(
        Operation.OpType opType,
        DataType dataType,
        NativeCpuKernelPerformanceStatus status,
        NativeCpuKernelFamily family,
        String reason
) {
    public NativeCpuKernelFact {
        opType = Objects.requireNonNull(opType, "opType cannot be null");
        dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        status = Objects.requireNonNull(status, "status cannot be null");
        family = Objects.requireNonNull(family, "family cannot be null");
        reason = reason == null ? "" : reason;
    }

    /**
     * Returns whether this fact represents native compute, not just a metadata view.
     */
    public boolean nativeComputeEligible() {
        return status == NativeCpuKernelPerformanceStatus.NATIVE_FAST
                || status == NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW
                || status == NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER;
    }

    /**
     * Returns whether an existing native storage value can remain native across this operation.
     */
    public boolean preservesNativeStorage() {
        return dataType != DataType.BOOL && (nativeComputeEligible() || status == NativeCpuKernelPerformanceStatus.VIEW_ONLY);
    }
}
