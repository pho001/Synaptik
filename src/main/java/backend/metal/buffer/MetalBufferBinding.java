package backend.metal.buffer;

import backend.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferAccessMode;
import backend.accelerator.buffer.AcceleratorLayoutAbiV2Descriptor;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.memory.DeviceBufferBinding;

import java.util.Objects;

/**
 * Runtime binding between one compiled graph value and a Metal-compatible buffer.
 *
 * <p>This is the Java-side contract used by the native Metal buffer bridge.
 * It carries only execution facts: node id, shared logical layout metadata,
 * buffer handle, and access intent. Tensor materialization policy remains in execution
 * state; native code receives buffer bindings rather than semantic tensors.</p>
 *
 * @param nodeId compiled node id represented by the buffer
 * @param layout backend-neutral logical tensor layout
 * @param handle native buffer handle
 * @param access native access intent
 */
public record MetalBufferBinding(
        int nodeId,
        AcceleratorBufferLayout layout,
        MetalBufferHandle handle,
        MetalBufferAccess access
) implements DeviceBufferBinding {
    public MetalBufferBinding {
        Objects.requireNonNull(layout, "layout cannot be null");
        Objects.requireNonNull(handle, "handle cannot be null");
        access = access == null ? MetalBufferAccess.READ : access;
    }

    /**
     * Creates a logical view binding over an existing Metal handle.
     *
     * <p>The returned binding borrows the source handle and does not imply resource ownership. The
     * allocation that produced {@code source} remains responsible for native lifetime management.</p>
     *
     * @param nodeId target compiled node id
     * @param layout target logical layout
     * @param source source binding whose handle is reused
     * @param access target access mode; defaults to source access when null
     * @return view binding over the same native handle
     */
    public static MetalBufferBinding viewOf(
            int nodeId,
            AcceleratorBufferLayout layout,
            MetalBufferBinding source,
            MetalBufferAccess access
    ) {
        Objects.requireNonNull(layout, "layout cannot be null");
        Objects.requireNonNull(source, "source cannot be null");
        if (!source.available()) {
            throw new IllegalArgumentException("source binding is unavailable: " + source.describe());
        }
        MetalBufferHandle viewHandle = new MetalBufferHandle(
                source.handle().nativeHandle(),
                source.handle().byteLength(),
                source.handle().storageMode(),
                source.handle().owner() + ":logical-view",
                false
        );
        return new MetalBufferBinding(
                nodeId,
                layout,
                viewHandle,
                access == null ? source.access() : access
        );
    }

    /**
     * Returns the expected byte size implied by the shared layout.
     *
     * @return logical payload byte size
     */
    @Override
    public long logicalByteLength() {
        return layout.logicalByteLength();
    }

    /**
     * Returns the backend id for Metal buffer bindings.
     *
     * @return {@code GPU_METAL}
     */
    @Override
    public String backendId() {
        return ComputeBackend.GPU_METAL.name();
    }

    /**
     * Returns whether the attached native buffer can hold the physical source span.
     *
     * <p>Logical views can require fewer physical bytes than their logical dense payload
     * length, for example zero-stride broadcast views. Dense bindings have identical
     * logical and physical byte requirements.</p>
     *
     * @return true when the handle is available and large enough for the physical layout span
     */
    public boolean bufferCoversLogicalPayload() {
        return handle.available()
                && handle.byteLength() >= requiredBufferByteLength();
    }

    private long requiredBufferByteLength() {
        if (handle.owner().contains(":logical-view")) {
            return AcceleratorLayoutAbiV2Descriptor.physicalByteSpan(layout);
        }
        return logicalByteLength();
    }

    @Override
    public AcceleratorBufferAccessMode accessMode() {
        return switch (access) {
            case READ -> AcceleratorBufferAccessMode.READ;
            case WRITE -> AcceleratorBufferAccessMode.WRITE;
            case READ_WRITE -> AcceleratorBufferAccessMode.READ_WRITE;
        };
    }

    @Override
    public String nativeHandleIdentity() {
        return backendId()
                + ":owner=" + handle.owner()
                + ",storageMode=" + handle.storageMode()
                + ",bytes=" + handle.byteLength();
    }

    /**
     * Returns whether the binding can be used by Metal.
     *
     * @return true when the native handle is available and large enough
     */
    @Override
    public boolean available() {
        return bufferCoversLogicalPayload();
    }

    /**
     * Returns a concise diagnostic string for trace and failure messages.
     *
     * @return binding summary
     */
    public String describe() {
        return "nodeId=" + nodeId
                + ", " + layout.describe()
                + ", access=" + access
                + ", handleBytes=" + handle.byteLength();
    }
}
