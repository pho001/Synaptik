package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.runtime.resource.PreparedResource;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Owns one reusable native Metal program executable and its adopted context child lease.
 *
 * <p>The opaque executable handle is consumed once before the lease ends. Close is synchronized,
 * idempotent, and attempt-all. Execution borrows both resources and is valid only while Runtime's
 * enclosing prepared-execution lease prevents this owner from closing.</p>
 */
final class MetalProgramExecutableResource implements PreparedResource {
    private final MetalDeviceContext context;
    private final MetalNativeApi api;
    private final MetalNativeApi.Handle executable;
    private final MetalDeviceContext.ChildLease contextLease;
    private final long[] inputRequiredBytes;
    private final long[] outputRequiredBytes;
    private final boolean materializedValueInputs;
    private boolean closed;

    /**
     * Adopts one successfully created executable and the already acquired provisional lease.
     *
     * @param context non-null exact device context required by every bound buffer
     * @param api non-null native seam retained for execution and release
     * @param executable non-null opaque executable handle owned by this wrapper
     * @param contextLease non-null provisional child lease adopted without another increment
     * @param inputRequiredBytes non-null input byte extents to snapshot: ABI descriptor byte extents
     * in compact materialized-value order for a custom program, or physical feed extents in feed
     * order for an MPSGraph executable
     * @param outputRequiredBytes non-null physical referenced-span byte extents in target order to
     * snapshot; these are separate from the custom program's compact materialized input extents
     * @param materializedValueInputs whether the native executable receives compact materialized
     * value inputs rather than ordinary feed buffers
     * @throws NullPointerException if a required reference is {@code null}
     */
    MetalProgramExecutableResource(
            MetalDeviceContext context,
            MetalNativeApi api,
            MetalNativeApi.Handle executable,
            MetalDeviceContext.ChildLease contextLease,
            long[] inputRequiredBytes,
            long[] outputRequiredBytes,
            boolean materializedValueInputs) {
        this.context = Objects.requireNonNull(context, "context");
        this.api = Objects.requireNonNull(api, "api");
        this.executable = Objects.requireNonNull(executable, "executable");
        this.contextLease = Objects.requireNonNull(contextLease, "contextLease");
        this.inputRequiredBytes = inputRequiredBytes.clone();
        this.outputRequiredBytes = outputRequiredBytes.clone();
        this.materializedValueInputs = materializedValueInputs;
    }

    /** @return the exact device context required by compatible bound buffers */
    MetalDeviceContext context() { return context; }

    /** @return a private copy of prepared input byte extents, in compact materialized-value order
     * for a custom program or feed order for MPSGraph; never {@code null} */
    long[] inputRequiredBytes() { return inputRequiredBytes.clone(); }

    /** @return a private copy of physical referenced-span target byte extents in target order;
     * never {@code null} */
    long[] outputRequiredBytes() { return outputRequiredBytes.clone(); }

    /**
     * Performs one synchronous native executable call through already ordered concrete buffers.
     *
     * @param inputCount exact number of input handles expected by the executable
     * @param inputHandles non-null native address array in prepared input order: compact
     * materialized-value order for a custom program or feed order for MPSGraph
     * @param outputCount exact number of output handles expected by the executable
     * @param outputHandles non-null native address array in prepared target order
     * @throws NullPointerException if a handle array is {@code null}
     * @throws IllegalStateException if this resource is closed or a handle array is not alive and
     * accessible on the calling thread
     * @throws IllegalArgumentException if non-null handle arrays or counts violate the prepared ABI
     * @throws RuntimeException if native validation or synchronous execution fails
     */
    synchronized void run(int inputCount, MemorySegment inputHandles,
            int outputCount, MemorySegment outputHandles) {
        if (closed) throw new IllegalStateException("Metal program executable is closed");
        MetalNativeApi.ProgramExecutableAbi.validateRun(
                inputRequiredBytes.length, inputCount, inputHandles,
                outputRequiredBytes.length, outputCount, outputHandles,
                materializedValueInputs);
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
