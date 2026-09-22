package io.github.pho001.synaptik.backend.metal.internal;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Owns one native Metal device/command-queue context and leases held by its child resources.
 *
 * <p>The context begins with one owner reference. Every successful buffer or workspace
 * allocation adds one child lease. Closing marks the owner closed before releasing its reference,
 * rejects later allocation and access, and defers the single native context release until the
 * last child closes. The lifecycle gate is thread-safe; distinct open child resources otherwise
 * remain independently usable.</p>
 */
final class MetalDeviceContext implements AutoCloseable {
    private final MetalNativeApi api;
    private final MetalNativeApi.Handle handle;
    private boolean closed;
    private boolean nativeReleased;
    private int childLeases;

    private MetalDeviceContext(MetalNativeApi api, MetalNativeApi.Handle handle) {
        this.api = api;
        this.handle = handle;
    }

    /**
     * Loads the exact absolute dylib path and creates one default-device context.
     *
     * @param absoluteLibraryPath caller-selected absolute dylib path; must not be {@code null}
     * @return a new open context owning the library lookup and native context; never {@code null}
     * @throws NullPointerException if {@code absoluteLibraryPath} is {@code null}
     * @throws IllegalArgumentException if the path is not absolute
     * @throws RuntimeException if ABI loading, validation, or native context creation fails
     * @throws Error if loading, creation, or failure cleanup reports an error
     */
    static MetalDeviceContext open(Path absoluteLibraryPath) {
        return open(MetalNativeApi.open(absoluteLibraryPath));
    }

    /**
     * Creates one context through an injected API and takes ownership of that API's lifetime.
     *
     * @param api the non-null open native seam to own
     * @return a new open context; never {@code null}
     * @throws NullPointerException if {@code api} or its successful handle is {@code null}
     * @throws RuntimeException if native creation fails
     * @throws Error if creation or cleanup reports an error
     */
    static MetalDeviceContext open(MetalNativeApi api) {
        Objects.requireNonNull(api, "api");
        try {
            MetalNativeApi.Handle context = Objects.requireNonNull(
                    api.createContext(), "native context handle");
            return new MetalDeviceContext(api, context);
        } catch (RuntimeException | Error failure) {
            suppressDistinct(failure, api::close);
            throw failure;
        }
    }

    /**
     * Allocates one fresh run-owned Metal buffer and acquires one child lease.
     *
     * @param logicalByteSize exact non-negative logical size in bytes
     * @return a new open buffer representation; never {@code null}
     * @throws IllegalStateException if context close has begun
     * @throws IllegalArgumentException if {@code logicalByteSize} is negative
     * @throws RuntimeException if native allocation fails
     */
    synchronized MetalBufferRepresentation createBuffer(long logicalByteSize) {
        requireOpen();
        requireNonNegative(logicalByteSize);
        MetalNativeApi.Handle buffer = Objects.requireNonNull(
                api.createBuffer(handle, logicalByteSize), "native buffer handle");
        childLeases++;
        return new MetalBufferRepresentation(this, api, buffer, logicalByteSize);
    }

    /**
     * Allocates one fresh run-owned Metal scratch workspace and acquires one child lease.
     *
     * @param logicalByteSize exact non-negative logical size in bytes
     * @return a new open workspace representation; never {@code null}
     * @throws IllegalStateException if context close has begun
     * @throws IllegalArgumentException if {@code logicalByteSize} is negative
     * @throws RuntimeException if native allocation fails
     */
    synchronized MetalWorkspaceRepresentation createWorkspace(long logicalByteSize) {
        requireOpen();
        requireNonNegative(logicalByteSize);
        MetalNativeApi.Handle buffer = Objects.requireNonNull(
                api.createBuffer(handle, logicalByteSize), "native workspace handle");
        childLeases++;
        return new MetalWorkspaceRepresentation(this, api, buffer, logicalByteSize);
    }

    /**
     * Compiles one persistent whole-partition NEG executable under a provisional child lease.
     *
     * <p>Lease acquisition is atomic with owner close. Compilation runs outside the lifecycle
     * monitor while the lease keeps the native context alive. Successful wrapper construction
     * adopts that same lease; every failure releases native executable state before the lease.</p>
     *
     * @param plan non-null immutable shape-specialized lowering facts
     * @return a new open persistent executable resource; never {@code null}
     * @throws NullPointerException if {@code plan} is {@code null}
     * @throws IllegalArgumentException if {@code plan} retains another context identity
     * @throws IllegalStateException if owner close has begun
     * @throws RuntimeException if native compilation or cleanup fails
     * @throws Error if compilation or cleanup reports an error
     */
    MetalMpsGraphExecutableResource createNegExecutable(MetalNegPreparationPlan plan) {
        Objects.requireNonNull(plan, "plan");
        if (plan.context() != this) {
            throw new IllegalArgumentException(
                    "Metal NEG executable plan belongs to another device context");
        }
        if (plan.route() != MetalNegPreparationPlan.Route.MPSGRAPH) {
            throw new IllegalArgumentException(
                    "Metal NEG MPSGraph executable requires the MPSGraph route");
        }
        ChildLease lease = acquireChildLease();
        MetalNativeApi.Handle executable = null;
        try {
            executable = api.createNegExecutable(handle,
                    plan.valueRanks(), plan.valueDimensions(),
                    plan.nodeInputValueIndices(), plan.nodeOutputValueIndices(),
                    plan.feedValueIndices(), plan.targetValueIndices());
            return new MetalMpsGraphExecutableResource(
                    this, api, executable, lease,
                    plan.feedRequiredBytes(), plan.targetRequiredBytes());
        } catch (RuntimeException | Error failure) {
            if (executable != null) {
                try {
                    api.releaseExecutable(executable);
                } catch (RuntimeException | Error cleanup) {
                    if (cleanup != failure) failure.addSuppressed(cleanup);
                }
            }
            lease.closeAfter(failure);
            throw failure;
        }
    }

    /**
     * Compiles one persistent custom NEG pipeline under a provisional child lease.
     *
     * <p>The analysis-selected custom route and its checked element geometry are validated before
     * native creation. Successful wrapper construction adopts the lease; failure releases a
     * published pipeline before ending the lease.</p>
     *
     * @param plan non-null custom-route preparation plan retaining this exact context
     * @return a new open typed custom-pipeline resource; never {@code null}
     * @throws NullPointerException if {@code plan} is {@code null}
     * @throws IllegalArgumentException if route or context identity disagrees
     * @throws IllegalStateException if owner close has begun
     * @throws RuntimeException if native compilation or cleanup fails
     * @throws Error if compilation or cleanup reports an error
     */
    MetalNegKernelPipelineResource createNegKernelPipeline(MetalNegPreparationPlan plan) {
        Objects.requireNonNull(plan, "plan");
        if (plan.context() != this
                || plan.route() != MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG) {
            throw new IllegalArgumentException(
                    "Metal NEG custom pipeline plan route or context disagrees");
        }
        ChildLease lease = acquireChildLease();
        MetalNativeApi.Handle pipeline = null;
        try {
            pipeline = api.createNegKernelPipeline(handle, plan.customElementCount());
            return new MetalNegKernelPipelineResource(
                    this, api, pipeline, lease, plan.feedRequiredBytes()[0]);
        } catch (RuntimeException | Error failure) {
            if (pipeline != null) {
                try {
                    api.releaseNegKernelPipeline(pipeline);
                } catch (RuntimeException | Error cleanup) {
                    if (cleanup != failure) {
                        failure.addSuppressed(cleanup);
                    }
                }
            }
            lease.closeAfter(failure);
            throw failure;
        }
    }

    /**
     * Admits one already resource-gated native access while the context remains open.
     *
     * <p>Admission is atomic with owner close, but the action runs without holding the context
     * lifecycle gate so distinct open child resources may be accessed concurrently. The calling
     * child resource's gate and retained context lease keep both native handles alive until an
     * admitted action completes.</p>
     *
     * @param access non-null access action that does not retain context state
     * @throws NullPointerException if {@code access} is {@code null}
     * @throws IllegalStateException if context close has begun before admission
     * @throws RuntimeException if the access action fails
     */
    void access(Runnable access) {
        synchronized (this) {
            requireOpen();
            Objects.requireNonNull(access, "access");
        }
        access.run();
    }

    /** @return whether owner close has begun; safe to query concurrently */
    synchronized boolean isClosed() {
        return closed;
    }

    /**
     * Marks this owner closed and releases its native state once all child leases have ended.
     *
     * <p>Repeated and concurrent calls are inert after the first attempt. A native release or
     * lookup-arena cleanup failure propagates from the first call only and is never retried.</p>
     *
     * @throws RuntimeException if native context or lookup cleanup fails
     * @throws Error if cleanup reports an error
     */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (childLeases == 0) {
            releaseNativeContext(null);
        }
    }

    /** Ends one child lease and attaches a distinct deferred context failure to its primary. */
    synchronized void releaseChild(Throwable primary) {
        if (childLeases <= 0) {
            throw new IllegalStateException("Metal context child lease underflow");
        }
        childLeases--;
        if (closed && childLeases == 0) {
            releaseNativeContext(primary);
        }
    }

    private synchronized ChildLease acquireChildLease() {
        requireOpen();
        childLeases++;
        return new ChildLease(this);
    }

    /** One exactly-once provisional or adopted context child lease. */
    static final class ChildLease {
        private final MetalDeviceContext context;
        private boolean closed;

        private ChildLease(MetalDeviceContext context) {
            this.context = context;
        }

        /**
         * Ends this lease once, preserving an existing primary failure when supplied.
         *
         * @param primary triggering failure, or {@code null} when lease cleanup owns failure
         * @throws RuntimeException if deferred context cleanup fails without a primary
         * @throws Error if deferred context cleanup reports an error without a primary
         */
        synchronized void closeAfter(Throwable primary) {
            if (closed) return;
            closed = true;
            context.releaseChild(primary);
        }
    }

    private void releaseNativeContext(Throwable primary) {
        if (nativeReleased) {
            return;
        }
        nativeReleased = true;
        Throwable failure = primary;
        try {
            api.releaseContext(handle);
        } catch (RuntimeException | Error cleanup) {
            if (failure == null) {
                failure = cleanup;
            } else if (cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
        }
        try {
            api.close();
        } catch (RuntimeException | Error cleanup) {
            if (failure == null) {
                failure = cleanup;
            } else if (cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
        }
        if (primary == null && failure != null) {
            rethrow(failure);
        }
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("Metal device context is closed");
        }
    }

    private static void requireNonNegative(long logicalByteSize) {
        if (logicalByteSize < 0L) {
            throw new IllegalArgumentException("logicalByteSize must be non-negative");
        }
    }

    private static void suppressDistinct(Throwable primary, Runnable cleanup) {
        try {
            cleanup.run();
        } catch (RuntimeException | Error failure) {
            if (failure != primary) {
                primary.addSuppressed(failure);
            }
        }
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        throw (Error) failure;
    }
}
