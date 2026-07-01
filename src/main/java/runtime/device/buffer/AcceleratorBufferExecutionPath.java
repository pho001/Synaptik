package runtime.device.buffer;

/**
 * High-level execution path selected by accelerator buffer policy.
 */
public enum AcceleratorBufferExecutionPath {
    /**
     * Native backend buffer bindings are used.
     */
    BUFFER_BINDING,

    /**
     * Legacy tensor-array bridge path is used.
     */
    TENSOR_ARRAY,

    /**
     * The accelerator executable replays the partition on CPU.
     */
    CPU_FALLBACK,

    /**
     * The buffer path was not evaluated or is not available.
     */
    UNAVAILABLE
}
