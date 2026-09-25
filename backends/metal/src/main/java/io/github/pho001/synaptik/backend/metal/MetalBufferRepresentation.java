package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Metal shared-storage buffer owner with exact logical byte geometry.
 *
 * <p>The representation owns one opaque native buffer handle and one child lease on its device
 * context. It may belong directly to a run or to an immutable prepared splat resource whose fresh
 * read-only run bindings never expose this owner for mutation. Upload and download expose only
 * explicitly requested byte ranges and interpret no tensor element type. Access, close, and
 * context close are coordinated so an admitted access completes before its relevant close gate
 * proceeds. Closing is thread-safe, idempotent, and consumes the native handle at most once.</p>
 */
final class MetalBufferRepresentation implements BufferRepresentation {
    private final MetalDeviceContext context;
    private final MetalNativeApi api;
    private final MetalNativeApi.Handle handle;
    private final long logicalByteSize;
    /** Optional caller storage retained for the complete borrowed-input wrapper lifetime. */
    @SuppressWarnings("unused")
    private final Object retainedBorrow;
    private final DenseAffinePublication denseAffinePublication;
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
        this(context, api, handle, logicalByteSize, null);
    }

    /**
     * Creates one open owner that also retains a non-owning caller-storage borrow.
     *
     * @param context non-null context that owns the child lease
     * @param api non-null native seam shared with the context
     * @param handle non-null opaque buffer handle consumed by this owner
     * @param logicalByteSize exact non-negative logical byte extent
     * @param retainedBorrow optional caller-owned object retained but never closed
     */
    MetalBufferRepresentation(
            MetalDeviceContext context,
            MetalNativeApi api,
            MetalNativeApi.Handle handle,
            long logicalByteSize,
            Object retainedBorrow) {
        this.context = Objects.requireNonNull(context, "context");
        this.api = Objects.requireNonNull(api, "api");
        this.handle = Objects.requireNonNull(handle, "handle");
        this.logicalByteSize = logicalByteSize;
        if (retainedBorrow instanceof DenseAffinePublication publication) {
            this.retainedBorrow = null;
            this.denseAffinePublication = publication;
        } else {
            this.retainedBorrow = retainedBorrow;
            this.denseAffinePublication = null;
        }
    }

    /** @return the exact non-negative logical byte extent, including zero */
    long byteSize() {
        return logicalByteSize;
    }

    /** @return whether this live representation belongs to the exact supplied device context */
    synchronized boolean belongsTo(MetalDeviceContext expected) {
        return !closed && context == expected;
    }

    /** Returns whether this live buffer is an authenticated dense target for the exact view. */
    synchronized boolean authenticatesDenseAffinePublication(TensorDescriptor descriptor) {
        Objects.requireNonNull(descriptor, "descriptor");
        return !closed
                && denseAffinePublication != null
                && denseAffinePublication.authenticates(
                        context, descriptor, logicalByteSize);
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

    /**
     * Unforgeable outside this package: exact finalized-route evidence for one dense physical
     * affine target whose public descriptor remains a logical view. Evidence binds the executable,
     * plan, context, value identity, producer kind, target position, descriptor, and byte extent.
     */
    static final class DenseAffinePublication {
        private final MetalNegPreparedExecutable executable;
        private final MetalNegPreparationPlan plan;
        private final int targetPosition;
        private final ValueId valueId;
        private final MetalMpsGraphProgram.NodeKind producerKind;
        private final TensorDescriptor descriptor;
        private final long byteSize;
        DenseAffinePublication(
                MetalNegPreparedExecutable executable,
                int targetPosition,
                ValueId valueId,
                MetalMpsGraphProgram.NodeKind producerKind,
                TensorDescriptor descriptor,
                long byteSize) {
            this.executable = Objects.requireNonNull(executable, "executable");
            this.plan = executable.preparationPlan();
            this.targetPosition = targetPosition;
            this.valueId = Objects.requireNonNull(valueId, "valueId");
            this.producerKind = Objects.requireNonNull(producerKind, "producerKind");
            this.descriptor = Objects.requireNonNull(descriptor, "descriptor");
            this.byteSize = byteSize;
            if (plan.denseAffineProducerKind(
                    targetPosition, valueId, descriptor, byteSize)
                    .filter(producerKind::equals)
                    .isEmpty()) {
                throw new IllegalArgumentException(
                        "Metal dense affine publication is not an exact finalized route target");
            }
        }

        private boolean authenticates(
                MetalDeviceContext context,
                TensorDescriptor expectedDescriptor,
                long expectedByteSize) {
            return executable.preparationPlan() == plan
                    && plan.context() == context
                    && descriptor.equals(expectedDescriptor)
                    && byteSize == expectedByteSize
                    && plan.denseAffineProducerKind(
                            targetPosition, valueId, expectedDescriptor, expectedByteSize)
                            .filter(producerKind::equals)
                            .isPresent();
        }
    }
}
