package io.github.pho001.synaptik.backend.metal.internal;

import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Run-owned Metal shared-storage buffer with exact logical byte geometry.
 *
 * <p>The representation owns one opaque native buffer handle and one child lease on its device
 * context. Upload and download expose only explicitly requested byte ranges and interpret no
 * tensor element type. Access, close, and context close are coordinated so an admitted access
 * completes before its relevant close gate proceeds. Closing is thread-safe, idempotent, and
 * consumes the native handle at most once.</p>
 */
final class MetalBufferRepresentation implements BufferRepresentation {
    private final MetalDeviceContext context;
    private final MetalNativeApi api;
    private final MetalNativeApi.Handle handle;
    private final long logicalByteSize;
    private boolean closed;

    /**
     * Creates one open owner around an already allocated native buffer and acquired context lease.
     *
     * @param context non-null context that owns the child lease
     * @param api non-null native seam shared with the context
     * @param handle non-null opaque buffer handle consumed by this owner
     * @param logicalByteSize exact non-negative logical byte extent
     */
    MetalBufferRepresentation(
            MetalDeviceContext context,
            MetalNativeApi api,
            MetalNativeApi.Handle handle,
            long logicalByteSize) {
        this.context = Objects.requireNonNull(context, "context");
        this.api = Objects.requireNonNull(api, "api");
        this.handle = Objects.requireNonNull(handle, "handle");
        this.logicalByteSize = logicalByteSize;
    }

    /** @return the exact non-negative logical byte extent, including zero */
    long byteSize() {
        return logicalByteSize;
    }

    /** @return whether this live representation belongs to the exact supplied device context */
    synchronized boolean belongsTo(MetalDeviceContext expected) {
        return !closed && context == expected;
    }

    /**
     * Returns the opaque handle only to the package-private prepared execution bridge.
     *
     * @return the non-null live native buffer handle
     * @throws IllegalStateException if close has begun
     */
    synchronized MetalNativeApi.Handle executionHandle() {
        requireOpen();
        return handle;
    }

    /** @return whether close has begun; safe to query concurrently */
    synchronized boolean isClosed() {
        return closed;
    }

    /**
     * Uploads exactly one bounded range from caller-owned native memory.
     *
     * @param bufferOffset non-negative destination offset in this logical buffer
     * @param source non-null live, current-thread-accessible native source segment; retained only
     *     for the duration of this call
     * @param sourceOffset non-negative byte offset in {@code source}
     * @param byteCount non-negative number of bytes to copy
     * @throws NullPointerException if {@code source} is {@code null}
     * @throws IllegalStateException if this resource or its context is closed, or the segment is
     *     not alive or accessible by the current thread
     * @throws IllegalArgumentException if an offset/count is negative, either range is out of
     *     bounds, or {@code source} is not native memory
     * @throws RuntimeException if the native copy fails
     */
    synchronized void upload(
            long bufferOffset, MemorySegment source, long sourceOffset, long byteCount) {
        requireOpen();
        Objects.requireNonNull(source, "source");
        requireReadableNative(source, "source");
        requireRange(bufferOffset, byteCount, logicalByteSize, "buffer");
        requireRange(sourceOffset, byteCount, source.byteSize(), "source");
        MemorySegment bytes = byteCount == 0L
                ? MemorySegment.NULL : source.asSlice(sourceOffset, byteCount);
        context.access(() -> api.upload(handle, bufferOffset, bytes, byteCount));
    }

    /**
     * Downloads exactly one bounded range into caller-owned writable native memory.
     *
     * @param bufferOffset non-negative source offset in this logical buffer
     * @param destination non-null live, current-thread-accessible writable native destination;
     *     retained only for the duration of this call
     * @param destinationOffset non-negative byte offset in {@code destination}
     * @param byteCount non-negative number of bytes to copy
     * @throws NullPointerException if {@code destination} is {@code null}
     * @throws IllegalStateException if this resource or its context is closed, or the segment is
     *     not alive or accessible by the current thread
     * @throws IllegalArgumentException if an offset/count is negative, either range is out of
     *     bounds, or {@code destination} is non-native or read-only
     * @throws RuntimeException if the native copy fails
     */
    synchronized void download(
            long bufferOffset,
            MemorySegment destination,
            long destinationOffset,
            long byteCount) {
        requireOpen();
        Objects.requireNonNull(destination, "destination");
        requireWritableNative(destination, "destination");
        requireRange(bufferOffset, byteCount, logicalByteSize, "buffer");
        requireRange(destinationOffset, byteCount, destination.byteSize(), "destination");
        MemorySegment bytes = byteCount == 0L
                ? MemorySegment.NULL : destination.asSlice(destinationOffset, byteCount);
        context.access(() -> api.download(handle, bufferOffset, bytes, byteCount));
    }

    /**
     * Marks this representation closed, consumes its native handle once, and ends its lease.
     *
     * <p>Repeated and concurrent calls are inert after the first attempt. A native buffer-release
     * failure remains primary; a distinct deferred context-cleanup failure is suppressed on it.
     * No failed release is retried.</p>
     *
     * @throws RuntimeException if buffer or deferred context cleanup fails
     * @throws Error if cleanup reports an error
     */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        Throwable failure = null;
        try {
            api.releaseBuffer(handle);
        } catch (RuntimeException | Error releaseFailure) {
            failure = releaseFailure;
        }
        try {
            context.releaseChild(failure);
        } catch (RuntimeException | Error contextFailure) {
            if (failure == null) {
                failure = contextFailure;
            } else if (contextFailure != failure) {
                failure.addSuppressed(contextFailure);
            }
        }
        if (failure != null) {
            rethrow(failure);
        }
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("Metal buffer is closed");
        }
    }

    private static void requireReadableNative(MemorySegment segment, String name) {
        if (!segment.scope().isAlive() || !segment.isAccessibleBy(Thread.currentThread())) {
            throw new IllegalStateException(name + " is not alive and accessible");
        }
        if (!segment.isNative()) {
            throw new IllegalArgumentException(name + " must be native memory");
        }
    }

    private static void requireWritableNative(MemorySegment segment, String name) {
        requireReadableNative(segment, name);
        if (segment.isReadOnly()) {
            throw new IllegalArgumentException(name + " must be writable");
        }
    }

    private static void requireRange(long offset, long count, long size, String name) {
        if (offset < 0L || count < 0L || offset > size || count > size - offset) {
            throw new IllegalArgumentException(name + " byte range is out of bounds");
        }
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        throw (Error) failure;
    }
}
