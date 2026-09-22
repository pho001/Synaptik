package io.github.pho001.synaptik.backend.metal.internal;

import io.github.pho001.synaptik.runtime.resource.PreparedResource;
import java.util.Objects;

/**
 * Owns one reusable native custom-NEG compute pipeline and its adopted context child lease.
 *
 * <p>The resource is distinct from the MPSGraph executable owner. Cold binding supplies exactly
 * one compatible input and assigned output buffer; each admitted invocation passes those two
 * handles to the matching native custom-pipeline function once. Native execution uses one
 * command buffer, one compute encoder, and one completion wait, and writes the assigned
 * {@code MTLBuffer} without an explicit host-staging or intermediate-copy step. Close consumes
 * the pipeline before ending the context lease and is synchronized, idempotent, and
 * attempt-all.</p>
 */
final class MetalNegKernelPipelineResource implements PreparedResource {
    private final MetalDeviceContext context;
    private final MetalNativeApi api;
    private final MetalNativeApi.Handle pipeline;
    private final MetalDeviceContext.ChildLease contextLease;
    private final long requiredBytes;
    private boolean closed;

    /**
     * Adopts one successfully created custom pipeline and its provisional context lease.
     *
     * @param context exact non-null device context required by both bound buffers
     * @param api non-null native seam retained for invocation and release
     * @param pipeline non-null opaque custom-pipeline handle owned by this wrapper
     * @param contextLease non-null provisional child lease adopted without another increment
     * @param requiredBytes positive logical byte extent required from each bound buffer
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if {@code requiredBytes} is not positive
     */
    MetalNegKernelPipelineResource(
            MetalDeviceContext context,
            MetalNativeApi api,
            MetalNativeApi.Handle pipeline,
            MetalDeviceContext.ChildLease contextLease,
            long requiredBytes) {
        this.context = Objects.requireNonNull(context, "context");
        this.api = Objects.requireNonNull(api, "api");
        this.pipeline = Objects.requireNonNull(pipeline, "pipeline");
        this.contextLease = Objects.requireNonNull(contextLease, "contextLease");
        if (requiredBytes <= 0L) {
            throw new IllegalArgumentException("requiredBytes must be positive");
        }
        this.requiredBytes = requiredBytes;
    }

    /** @return the exact device context required by compatible bound buffers */
    MetalDeviceContext context() {
        return context;
    }

    /** @return the positive logical byte extent required from input and output */
    long requiredBytes() {
        return requiredBytes;
    }

    /**
     * Performs one synchronous custom-pipeline invocation through direct concrete buffers.
     *
     * @param input non-null live input handle validated during cold binding
     * @param output non-null live output handle validated during cold binding
     * @throws NullPointerException if {@code input} or {@code output} is {@code null}
     * @throws IllegalStateException if this resource is closed
     * @throws RuntimeException if native validation or execution fails
     */
    synchronized void run(MetalNativeApi.Handle input, MetalNativeApi.Handle output) {
        if (closed) {
            throw new IllegalStateException("Metal NEG kernel pipeline is closed");
        }
        api.runNegKernelPipeline(pipeline, input, output);
    }

    /**
     * Consumes the native custom pipeline once, then ends the adopted context lease.
     *
     * @throws RuntimeException if pipeline or deferred context cleanup fails
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
            api.releaseNegKernelPipeline(pipeline);
        } catch (RuntimeException | Error cleanup) {
            failure = cleanup;
        }
        try {
            contextLease.closeAfter(failure);
        } catch (RuntimeException | Error cleanup) {
            if (failure == null) {
                failure = cleanup;
            } else if (cleanup != failure) {
                failure.addSuppressed(cleanup);
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
