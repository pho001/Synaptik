package backend.cuda.buffer;

import runtime.memory.ExecutionResource;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Run-scoped cleanup wrapper for an owned CUDA buffer.
 */
public final class CudaBufferResource implements ExecutionResource {
    private final CudaBufferAllocator allocator;
    private final CudaBufferHandle handle;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    public CudaBufferResource(CudaBufferAllocator allocator, CudaBufferHandle handle) {
        this.allocator = Objects.requireNonNull(allocator, "allocator cannot be null");
        this.handle = Objects.requireNonNull(handle, "handle cannot be null");
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            allocator.destroy(handle);
        }
    }
}
