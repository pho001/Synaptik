package backend.metal.buffer;

import runtime.memory.ExecutionResource;

import java.util.Objects;

/**
 * Run-scoped owner for a Metal buffer handle allocated during prepared execution.
 */
public final class MetalBufferResource implements ExecutionResource {
    private final MetalBufferAllocator allocator;
    private final MetalBufferHandle handle;
    private boolean closed;

    /**
     * Creates a resource wrapper around an owned Metal buffer handle.
     *
     * @param allocator allocator that created the handle
     * @param handle native handle to destroy when the run ends
     */
    public MetalBufferResource(MetalBufferAllocator allocator, MetalBufferHandle handle) {
        this.allocator = Objects.requireNonNull(allocator, "allocator cannot be null");
        this.handle = Objects.requireNonNull(handle, "handle cannot be null");
    }

    /**
     * Destroys the handle exactly once.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        allocator.destroy(handle);
        closed = true;
    }
}
