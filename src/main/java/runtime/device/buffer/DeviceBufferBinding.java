package runtime.device.buffer;

/**
 * Backend-neutral descriptor for a runtime tensor value that has a device-visible buffer.
 *
 * <p>This interface is intentionally smaller than backend-specific buffer handles. Execution state
 * only needs to know which compiled node the binding represents, which backend owns it, the logical
 * tensor layout it covers, whether it is usable, and a diagnostic native identity. Metal, CUDA, or another accelerator can carry
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
     * Returns the logical tensor layout represented by this binding.
     *
     * @return backend-neutral layout metadata
     */
    AcceleratorBufferLayout layout();

    /**
     * Returns the backend-neutral access mode requested for this binding.
     *
     * @return access mode
     */
    AcceleratorBufferAccessMode accessMode();

    /**
     * Returns an opaque diagnostic identity for the backend-owned native handle.
     *
     * @return native identity string without exposing backend-native handle objects
     */
    String nativeHandleIdentity();

    /**
     * Returns the logical tensor payload size in bytes.
     *
     * @return logical byte length
     */
    default long logicalByteLength() {
        return layout().logicalByteLength();
    }

    /**
     * Creates a metadata-only logical view over this binding's native buffer, when supported by the backend.
     *
     * <p>The default returns {@code null} so shared graph execution code can ask for a view without importing concrete
     * backend binding classes. Backend implementations that support borrowed-handle logical views should override this
     * method and preserve the native lifetime ownership of the source binding.</p>
     *
     * @param targetNodeId compiled node id for the logical view
     * @param targetLayout logical layout represented by the view
     * @param accessMode backend-neutral access intent for the view
     * @return view binding, or {@code null} when this binding cannot create metadata-only views
     */
    default DeviceBufferBinding viewOf(
            int targetNodeId,
            AcceleratorBufferLayout targetLayout,
            AcceleratorBufferAccessMode accessMode
    ) {
        return null;
    }

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
