package backend.memory;

/**
 * Physical residency class for a runtime tensor value.
 *
 * <p>This enum describes where the newest materialized representation may live.
 * It does not imply dtype support or backend legality by itself; backend
 * planners still own those checks. The residency value exists so execution,
 * tracing, and later zero-copy paths can distinguish CPU-array tensors from
 * shared host/device buffers and device-owned buffers.</p>
 */
public enum StorageResidency {
    /**
     * The current value is represented as the tensor's normal CPU-side typed array.
     */
    CPU_ARRAY,

    /**
     * The current value is represented as native CPU memory backed by a {@code MemorySegment}.
     */
    CPU_NATIVE,

    /**
     * The value is represented by a host-visible buffer that can also be used by a device backend.
     */
    HOST_SHARED_DEVICE_BUFFER,

    /**
     * The value is primarily owned by a device backend and requires materialization before CPU array reads.
     */
    DEVICE_OWNED
}
