package backend.memory;

/**
 * Backend-neutral descriptor for a runtime tensor value that has a device-visible buffer.
 *
 * <p>This interface is intentionally smaller than backend-specific buffer handles. Execution state
 * only needs to know which compiled node the binding represents, which backend owns it, whether it
 * is usable, and how many logical bytes it covers. Metal, CUDA, or another accelerator can carry
 * richer native-handle details in their own implementation classes.</p>
 */
public interface DeviceBufferBinding {
    /**
     * Returns the compiled node id represented by this binding.
     *
     * @return compiled node id
     */
    int nodeId();

    /**
     * Returns the backend id that owns or can consume the buffer, for example {@code GPU_METAL}.
     *
     * @return backend id
     */
    String backendId();

    /**
     * Returns the logical tensor payload size in bytes.
     *
     * @return logical byte length
     */
    long logicalByteLength();

    /**
     * Returns whether this binding can be used by its backend.
     *
     * @return true when the binding has a valid backend handle and covers the logical payload
     */
    boolean available();

    /**
     * Returns a concise diagnostic summary suitable for traces and failure messages.
     *
     * @return diagnostic description
     */
    String describe();
}
