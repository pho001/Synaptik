package io.github.pho001.synaptik.backend.metal.internal;

import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import java.util.Objects;

/**
 * Run-owned Metal shared-storage scratch workspace.
 *
 * <p>The workspace owns one opaque native buffer and one context child lease. It exposes no host
 * byte access, logical-buffer validity, tensor type, or transfer behavior. Closing is thread-safe,
 * idempotent, and consumes the native buffer handle at most once.</p>
 */
final class MetalWorkspaceRepresentation implements WorkspaceRepresentation {
    private final MetalDeviceContext context;
    private final MetalNativeApi api;
    private final MetalNativeApi.Handle handle;
    private final long byteSize;
    private boolean closed;

    /**
     * Creates one open owner around allocated native scratch and an acquired context lease.
     *
     * @param context non-null context that owns the child lease
     * @param api non-null native seam shared with the context
     * @param handle non-null opaque buffer handle consumed by this owner
     * @param byteSize exact non-negative scratch extent in bytes
     */
    MetalWorkspaceRepresentation(
            MetalDeviceContext context,
            MetalNativeApi api,
            MetalNativeApi.Handle handle,
            long byteSize) {
        this.context = Objects.requireNonNull(context, "context");
        this.api = Objects.requireNonNull(api, "api");
        this.handle = Objects.requireNonNull(handle, "handle");
        this.byteSize = byteSize;
    }

    /** @return the exact non-negative scratch byte extent, including zero */
    long byteSize() {
        return byteSize;
    }

    /** @return whether close has begun; safe to query concurrently */
    synchronized boolean isClosed() {
        return closed;
    }

    /**
     * Marks this workspace closed, consumes its native handle once, and ends its context lease.
     *
     * <p>Repeated and concurrent calls are inert. A native buffer-release failure remains
     * primary; a distinct deferred context-cleanup failure is suppressed on it. Failed cleanup
     * is never retried.</p>
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
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (failure != null) {
            throw (Error) failure;
        }
    }
}
