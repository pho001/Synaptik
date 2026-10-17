package io.github.pho001.synaptik.backend.metal;

import java.nio.file.Path;
import java.util.Objects;
import java.util.UUID;

/**
 * Owns one native Metal device/command-queue context and leases held by its child resources.
 *
 * <p>The native seam validates the bridge ABI before a context is published. The context retains
 * only execution and session-compatibility state; route policy is a Java preparation concern.</p>
 *
 * <p>The context begins with one owner reference. Every successful buffer or workspace allocation
 * adds one child lease. Closing marks the owner closed before releasing its reference, rejects
 * later allocation and access, and defers the single native context release until the last child
 * closes. The lifecycle gate is thread-safe; distinct open child resources otherwise remain
 * independently usable.</p>
 */
final class MetalDeviceContext implements AutoCloseable {
    private final MetalNativeApi api;
    private final MetalNativeApi.Handle handle;
    private final SessionNonce sessionNonce;
    private final ResourcePublisher resourcePublisher;
    private boolean closed;
    private boolean nativeReleased;
    private int childLeases;

    MetalDeviceContext(
            MetalNativeApi api,
            MetalNativeApi.Handle handle,
            ResourcePublisher resourcePublisher) {
        this.api = Objects.requireNonNull(api, "api");
        this.handle = Objects.requireNonNull(handle, "handle");
        this.resourcePublisher = Objects.requireNonNull(
                resourcePublisher, "resourcePublisher");
        UUID nonce = UUID.randomUUID();
        this.sessionNonce = new SessionNonce(
                nonce.getMostSignificantBits(), nonce.getLeastSignificantBits());
    }

    /**
     * Returns the immutable nonce that restricts tuning compatibility to this context session.
     *
     * <p>The value contains no native handle or device claim. It is generated once with the
     * context and is deliberately different from the Metal ABI version, which cannot identify a
     * stable device across sessions.</p>
     *
     * @return non-null immutable nonce for this exact context lifetime
     */
    SessionNonce sessionNonce() {
        return sessionNonce;
    }

    /**
     * Private value representation of one context's session compatibility identity.
     *
     * @param highBits high-order random nonce bits
     * @param lowBits low-order random nonce bits
     */
    record SessionNonce(long highBits, long lowBits) { }

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
        return open(MetalNativeApi.open(
                Objects.requireNonNull(absoluteLibraryPath, "absoluteLibraryPath")));
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
        return open(api, ResourcePublisher.DEFAULT, MetalDeviceContext::new);
    }

    static MetalDeviceContext open(
            MetalNativeApi api,
            ResourcePublisher resourcePublisher,
            ContextPublisher contextPublisher) {
        Objects.requireNonNull(api, "api");
        Objects.requireNonNull(resourcePublisher, "resourcePublisher");
        Objects.requireNonNull(contextPublisher, "contextPublisher");
        MetalNativeApi.Handle context = null;
        try {
            context = Objects.requireNonNull(
                    api.createContext(), "native context handle");
            return Objects.requireNonNull(
                    contextPublisher.publish(api, context, resourcePublisher),
                    "Metal device context");
        } catch (RuntimeException | Error failure) {
            if (context != null) {
                MetalNativeApi.Handle acquired = context;
                suppressDistinct(failure, () -> api.releaseContext(acquired));
            }
            suppressDistinct(failure, api::close);
            throw failure;
        }
    }

    /**
     * Allocates one fresh Metal buffer and acquires one child lease. The immediate owner may be a
     * run representation or an immutable prepared resource.
     * @param logicalByteSize exact non-negative logical size in bytes
     * @return a new open buffer representation; never {@code null}
     * @throws IllegalStateException if context close has begun
     * @throws IllegalArgumentException if {@code logicalByteSize} is negative
     * @throws RuntimeException if native allocation fails
     */
    synchronized MetalBufferRepresentation createBuffer(long logicalByteSize) {
        return createBufferWithState(logicalByteSize, null);
    }

    /**
     * Allocates one fresh dense physical target authenticated to an exact affine route result.
     *
     * @param logicalByteSize exact positive full logical byte extent
     * @param publication exact non-null finalized-route evidence retained by the representation
     * @return a new open authenticated target representation
     */
    synchronized MetalBufferRepresentation createBuffer(
            long logicalByteSize,
            MetalBufferRepresentation.DenseAffinePublication publication) {
        return createBufferWithState(
                logicalByteSize, Objects.requireNonNull(publication, "publication"));
    }

    private MetalBufferRepresentation createBufferWithState(
            long logicalByteSize, Object retainedState) {
        requireOpen();
        requireNonNegative(logicalByteSize);
        MetalNativeApi.Handle buffer = Objects.requireNonNull(
                api.createBuffer(handle, logicalByteSize), "native buffer handle");
        childLeases++;
        try {
            return Objects.requireNonNull(
                    resourcePublisher.publishBuffer(
                            this, api, buffer, logicalByteSize, retainedState),
                    "Metal buffer representation");
        } catch (RuntimeException | Error failure) {
            suppressDistinct(failure, () -> api.releaseBuffer(buffer));
            suppressDistinct(failure, () -> releaseChild(failure));
            throw failure;
        }
    }

    /**
     * Allocates one Metal input buffer while retaining a non-owning caller-storage borrow.
     *
     * @param logicalByteSize exact non-negative logical size in bytes
     * @param retainedBorrow non-null caller-owned storage object retained but never closed
     * @return a new open buffer representation; never {@code null}
     * @throws NullPointerException if {@code retainedBorrow} is {@code null}
     * @throws IllegalStateException if context close has begun
     * @throws IllegalArgumentException if {@code logicalByteSize} is negative
     * @throws RuntimeException if native allocation fails
     */
    synchronized MetalBufferRepresentation createBorrowedBuffer(
            long logicalByteSize, Object retainedBorrow) {
        requireOpen();
        requireNonNegative(logicalByteSize);
        Objects.requireNonNull(retainedBorrow, "retainedBorrow");
        MetalNativeApi.Handle buffer = Objects.requireNonNull(
                api.createBuffer(handle, logicalByteSize), "native buffer handle");
        childLeases++;
        try {
            return Objects.requireNonNull(
                    resourcePublisher.publishBuffer(
                            this, api, buffer, logicalByteSize, retainedBorrow),
                    "Metal borrowed-buffer representation");
        } catch (RuntimeException | Error failure) {
            suppressDistinct(failure, () -> api.releaseBuffer(buffer));
            suppressDistinct(failure, () -> releaseChild(failure));
            throw failure;
        }
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
        try {
            return Objects.requireNonNull(
                    resourcePublisher.publishWorkspace(
                            this, api, buffer, logicalByteSize),
                    "Metal workspace representation");
        } catch (RuntimeException | Error failure) {
            suppressDistinct(failure, () -> api.releaseBuffer(buffer));
            suppressDistinct(failure, () -> releaseChild(failure));
            throw failure;
        }
    }

    /**
     * Compiles one persistent whole-partition program executable under a provisional lease.
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
    MetalProgramExecutableResource createProgramExecutable(MetalNegPreparationPlan plan) {
        Objects.requireNonNull(plan, "plan");
        if (plan.context() != this) {
            throw new IllegalArgumentException(
                    "Metal program executable plan belongs to another device context");
        }
        if (plan.route() != MetalPreparedRoute.MPSGRAPH
                && plan.route() != MetalPreparedRoute.CUSTOM_PROGRAM) {
            throw new IllegalArgumentException(
                    "Metal program executable requires an MPSGraph or custom-program route");
        }
        ChildLease lease = acquireChildLease();
        MetalNativeApi.Handle executable = null;
        try {
            executable = api.createProgramExecutable(handle,
            plan.programValueDescriptors(),
            plan.graphProgram(),
            plan.feedValueIndices(),
            plan.targetValueIndices(),
            plan.route(),
            plan.route() == MetalPreparedRoute.CUSTOM_PROGRAM
                    ? plan.pointwiseFusionPlan() : null);
            long[] runInputBytes = plan.route()
                    == MetalPreparedRoute.CUSTOM_PROGRAM
                    ? plan.materializedValueRequiredBytes()
                    : plan.feedRequiredBytes();
            return new MetalProgramExecutableResource(
                    this, api, executable, lease, runInputBytes, plan.targetRequiredBytes(),
                    plan.route() == MetalPreparedRoute.CUSTOM_PROGRAM);
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
                || plan.route() != MetalPreparedRoute.CUSTOM_SINGLE_NEG) {
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

    @FunctionalInterface
    interface ContextPublisher {
        MetalDeviceContext publish(
                MetalNativeApi api,
                MetalNativeApi.Handle handle,
                ResourcePublisher resourcePublisher);
    }

    interface ResourcePublisher {
        ResourcePublisher DEFAULT = new ResourcePublisher() {
            @Override
            public MetalBufferRepresentation publishBuffer(
                    MetalDeviceContext context,
                    MetalNativeApi api,
                    MetalNativeApi.Handle handle,
                    long logicalByteSize,
                    Object retainedBorrow) {
                return retainedBorrow == null
                        ? new MetalBufferRepresentation(
                                context, api, handle, logicalByteSize)
                        : new MetalBufferRepresentation(
                                context, api, handle, logicalByteSize, retainedBorrow);
            }

            @Override
            public MetalWorkspaceRepresentation publishWorkspace(
                    MetalDeviceContext context,
                    MetalNativeApi api,
                    MetalNativeApi.Handle handle,
                    long logicalByteSize) {
                return new MetalWorkspaceRepresentation(
                        context, api, handle, logicalByteSize);
            }
        };

        MetalBufferRepresentation publishBuffer(
                MetalDeviceContext context,
                MetalNativeApi api,
                MetalNativeApi.Handle handle,
                long logicalByteSize,
                Object retainedBorrow);

        MetalWorkspaceRepresentation publishWorkspace(
                MetalDeviceContext context,
                MetalNativeApi api,
                MetalNativeApi.Handle handle,
                long logicalByteSize);
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
