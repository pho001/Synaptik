package io.github.pho001.synaptik.backend.cpu;

import io.github.pho001.synaptik.trace.TraceEvent;
import io.github.pho001.synaptik.trace.TracePayload;

/**
 * Caller-owned passive sink for cold CPU preparation diagnostics.
 *
 * <p>Callbacks run synchronously on the preparing thread and may occur concurrently for independent
 * preparations. The caller must make the observer thread-safe. CPU retains but never closes it;
 * events contain detached facts and may be retained beyond integration closure. No callbacks occur
 * per invocation or tensor element. Callback {@link RuntimeException}s and {@link Error}s disable
 * further diagnostics and are contained without changing selection, finalization, or backend
 * failures. Already in-flight concurrent callbacks may still finish after disabling. Event IDs and
 * preparation IDs are local to one integration; timestamps use {@link System#nanoTime()}.</p>
 */
@FunctionalInterface
public interface CpuTraceObserver {
    /**
     * Receives one immutable, data-free CPU event in the common trace envelope.
     *
     * @param event non-null event; its payload is a {@link CpuPreparationDiagnostics}
     */
    void onEvent(TraceEvent<? extends TracePayload> event);
}
