package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.trace.TraceEvent;

/**
 * Receives passive terminal diagnostics from an explicitly observed ordinary tuning transaction.
 *
 * <p>The caller owns the observer, must make it thread-safe when sharing it across calls, and may
 * retain the immutable events. Engine never closes or retains the observer after the call returns.
 * Each observed {@link Engine#prepareTuned(CompiledGraph, ModelAutotuningRequest,
 * EngineTuningObserver)} call defines its own event stream: it emits at most one PREPARE event,
 * with event ID zero and a {@link System#nanoTime()} reading. There is no process-wide sequence or
 * run tracing. Successful tuning and lifecycle, argument, backend execution, cleanup, and final
 * publication failures do not emit these terminal fallback/rejection events.</p>
 *
 * <p>The callback is synchronous, outside Engine lifecycle locks, and still admitted to the Engine.
 * It must not wait for another thread to close that Engine; self-close rejects before shutdown.
 * A {@link RuntimeException} or {@link Error} thrown by the callback alone is contained and is not
 * added to the tuning failure. Observation cannot change the selected fallback policy, prepared
 * result, or original rejection failure.</p>
 */
@FunctionalInterface
public interface EngineTuningObserver {
    /**
     * Observes one immutable terminal PREPARE event.
     *
     * @param event non-null event whose payload contains only closed outcome, phase, and reason
     *     categories; a shared observer can receive concurrent callbacks from distinct calls
     * @throws RuntimeException as an observer failure signal, contained by Engine
     * @throws Error as an observer failure signal, contained by Engine only at this callback boundary
     */
    void onEvent(TraceEvent<EngineTuningDiagnostic> event);
}
