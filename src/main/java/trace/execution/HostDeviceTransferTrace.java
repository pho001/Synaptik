package trace.execution;

import runtime.contract.StorageResidency;
import runtime.contract.HostDeviceTransferKind;

import java.util.Objects;

/**
 * Trace entry for a host/device transfer or fallback bridge.
 */
public record HostDeviceTransferTrace(
        int nodeId,
        String backend,
        String dataType,
        StorageResidency sourceResidency,
        StorageResidency targetResidency,
        HostDeviceTransferKind transferKind,
        long bytes,
        long javaArrayBytes,
        long nativeBytes,
        long deviceBytes,
        long durationNs,
        boolean syncOnly,
        boolean directTransferSupported,
        boolean success,
        String fallbackReason,
        String detail
) {
    public HostDeviceTransferTrace {
        backend = backend == null ? "" : backend;
        dataType = dataType == null ? "" : dataType;
        Objects.requireNonNull(sourceResidency, "sourceResidency cannot be null");
        Objects.requireNonNull(targetResidency, "targetResidency cannot be null");
        Objects.requireNonNull(transferKind, "transferKind cannot be null");
        bytes = Math.max(0L, bytes);
        javaArrayBytes = Math.max(0L, javaArrayBytes);
        nativeBytes = Math.max(0L, nativeBytes);
        deviceBytes = Math.max(0L, deviceBytes);
        durationNs = Math.max(0L, durationNs);
        fallbackReason = fallbackReason == null ? "" : fallbackReason;
        detail = detail == null ? "" : detail;
    }
}
