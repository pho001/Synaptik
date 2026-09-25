package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.trace.TraceEvent;
import io.github.pho001.synaptik.trace.TracePayload;

/**
 * Receives typed diagnostic events from one explicitly traced Metal integration.
 *
 * <p>The caller retains ownership and must support concurrent callbacks. Metal never closes the
 * observer. A callback {@link RuntimeException} disables later tracing for that integration and is
 * contained without changing backend work or outcomes; an {@link Error} propagates normally.</p>
 */
@FunctionalInterface
public interface MetalTraceObserver {
    /**
     * Observes one immutable event from the integration's producer-defined trace stream.
     *
     * @param event non-null typed event; callbacks may occur concurrently
     * @throws RuntimeException to disable later tracing without changing the backend outcome
     * @throws Error to propagate a fatal callback failure
     */
    void onEvent(TraceEvent<? extends TracePayload> event);
}
