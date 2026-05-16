package graph.execution.trace;

/**
 * Runtime host/device transfer route observed during execution.
 */
public enum HostDeviceTransferKind {
    CPU_ARRAY_TO_DEVICE_COPY,
    DEVICE_TO_CPU_ARRAY_COPY,
    NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE,
    DEVICE_TO_ARRAY_TO_NATIVE_BRIDGE,
    NATIVE_SEGMENT_TO_DEVICE_COPY,
    DEVICE_TO_NATIVE_SEGMENT_COPY,
    MAPPING,
    SYNC_ONLY
}
