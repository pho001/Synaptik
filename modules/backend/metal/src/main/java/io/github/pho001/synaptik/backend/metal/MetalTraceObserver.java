package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.trace.TraceEvent;
import io.github.pho001.synaptik.trace.TracePayload;

/**
 * Receives typed diagnostic events from one explicitly traced Metal integration.
 *
 * <p>The caller retains ownership and must support concurrent callbacks. Metal never closes the
 * observer. Structural PREPARE and pre-run planning callback failures disable tracing and are
 * contained without aborting native finalization or execution. A callback
 * {@link RuntimeException} from outcome reporting also
 * disables later tracing without changing backend work or outcomes. A callback {@link Error}
 * propagates only from successful outcome reporting; during failure reporting, the backend failure
 * remains primary and receives a distinct acyclic callback error as a suppressed failure.</p>
 */
@FunctionalInterface
public interface MetalTraceObserver {
    /**
     * Observes one immutable event from the integration's producer-defined trace stream.
     *
     * @param event non-null typed event; callbacks may occur concurrently
     * @throws RuntimeException as a callback failure signal; Metal disables later tracing and
     *     contains it
     * @throws Error as a callback failure signal; Metal contains it for structural events,
     *     propagates it from successful outcome reporting, and suppresses a distinct acyclic error
     *     on the primary backend failure during failure reporting
     */
    void onEvent(TraceEvent<? extends TracePayload> event);
}
