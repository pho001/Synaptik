package io.github.pho001.synaptik.backend.metal.internal;

import io.github.pho001.synaptik.runtime.resource.PreparedResource;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Owns one reusable native MPSGraph executable and its adopted context child lease.
 *
 * <p>The opaque executable handle is consumed once before the lease ends. Close is synchronized,
 * idempotent, and attempt-all. Execution borrows both resources and is valid only while Runtime's
 * enclosing prepared-execution lease prevents this owner from closing.</p>
 */
final class MetalMpsGraphExecutableResource implements PreparedResource {
    private final MetalDeviceContext context;
    private final MetalNativeApi api;
    private final MetalNativeApi.Handle executable;
    private final MetalDeviceContext.ChildLease contextLease;
    private final long[] inputRequiredBytes;
    private final long[] outputRequiredBytes;
    private boolean closed;

    /**
     * Adopts one successfully created executable and the already acquired provisional lease.
     *
     * @param context non-null exact device context required by every bound buffer
     * @param api non-null native seam retained for execution and release
     * @param executable non-null opaque executable handle owned by this wrapper
     * @param contextLease non-null provisional child lease adopted without another increment
     * @param inputRequiredBytes non-null stable feed byte extents to snapshot
     * @param outputRequiredBytes non-null stable target byte extents to snapshot
     * @throws NullPointerException if a required reference is {@code null}
     */
    MetalMpsGraphExecutableResource(
            MetalDeviceContext context,
            MetalNativeApi api,
            MetalNativeApi.Handle executable,
            MetalDeviceContext.ChildLease contextLease,
            long[] inputRequiredBytes,
            long[] outputRequiredBytes) {
        this.context = Objects.requireNonNull(context, "context");
        this.api = Objects.requireNonNull(api, "api");
        this.executable = Objects.requireNonNull(executable, "executable");
        this.contextLease = Objects.requireNonNull(contextLease, "contextLease");
        this.inputRequiredBytes = inputRequiredBytes.clone();
        this.outputRequiredBytes = outputRequiredBytes.clone();
    }

    /** @return the exact device context required by compatible bound buffers */
    MetalDeviceContext context() { return context; }

    /** @return a private copy of stable feed byte extents */
    long[] inputRequiredBytes() { return inputRequiredBytes.clone(); }

    /** @return a private copy of stable target byte extents */
    long[] outputRequiredBytes() { return outputRequiredBytes.clone(); }

    /**
     * Performs one synchronous native executable call through already ordered concrete buffers.
     *
     * @param inputs non-null fixed feed-ordered concrete buffers
     * @param outputs non-null fixed target-ordered concrete buffers
     * @throws IllegalStateException if this resource is closed
     * @throws RuntimeException if native validation or execution fails
     */
    synchronized void run(int inputCount, MemorySegment inputHandles,
            int outputCount, MemorySegment outputHandles) {
        if (closed) throw new IllegalStateException("Metal MPSGraph executable is closed");
        MetalNativeApi.NegExecutableAbi.validateRun(
                inputRequiredBytes.length, inputCount, inputHandles,
                outputRequiredBytes.length, outputCount, outputHandles);
        api.runExecutable(executable, inputCount, inputHandles, outputCount, outputHandles);
    }

    /**
     * Consumes the native executable once and then ends the adopted context lease.
     *
     * @throws RuntimeException if executable or deferred context cleanup fails
     * @throws Error if cleanup reports an error
     */
    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;
        Throwable failure = null;
        try {
            api.releaseExecutable(executable);
        } catch (RuntimeException | Error cleanup) {
            failure = cleanup;
        }
        try {
            contextLease.closeAfter(failure);
        } catch (RuntimeException | Error cleanup) {
            if (failure == null) failure = cleanup;
            else if (cleanup != failure) failure.addSuppressed(cleanup);
        }
        if (failure instanceof RuntimeException runtime) throw runtime;
        if (failure != null) throw (Error) failure;
    }
}
