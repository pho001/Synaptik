package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.trace.TraceEvent;
import io.github.pho001.synaptik.trace.TraceEventId;
import io.github.pho001.synaptik.trace.TraceLevel;
import io.github.pho001.synaptik.trace.TracePhase;
import java.util.Objects;

/**
 * Produces at most one terminal event for one synchronously observed tuning transaction.
 * Retains only the borrowed observer and bounded category/boolean state, never a failure or any
 * tuning-domain object. The ordinary unobserved path neither constructs nor invokes this producer.
 */
final class EngineTuningTraceProducer {
    private final EngineTuningObserver observer;
    private EngineTuningDiagnostic.TuningPhase phase =
            EngineTuningDiagnostic.TuningPhase.LOCAL_WORKLOAD;
    private EngineTuningDiagnostic.Reason reason =
            EngineTuningDiagnostic.Reason.OTHER_TUNING_FAILURE;
    private boolean trialIoFailure;
    private boolean emitted;

    /**
     * Borrows one observer for the synchronous transaction; no identity or clock is read here.
     *
     * @param observer non-null caller-owned observer
     * @throws NullPointerException if {@code observer} is null
     */
    EngineTuningTraceProducer(EngineTuningObserver observer) {
        this.observer = Objects.requireNonNull(observer, "observer");
    }

    /** Records the producer's transition to complete-plan work, before requesting its handoff. */
    void completePlanPhase() {
        phase = EngineTuningDiagnostic.TuningPhase.COMPLETE_PLAN;
        reason = EngineTuningDiagnostic.Reason.OTHER_TUNING_FAILURE;
        trialIoFailure = false;
    }

    /** Records an actually absent backend handoff without interpreting its rejection text. */
    void noEligibleHandoff() {
        reason = EngineTuningDiagnostic.Reason.NO_ELIGIBLE_HANDOFF;
    }

    /** Distinguishes checked trial-preparation I/O from cache I/O at the enclosing tool boundary. */
    void trialIoFailure() {
        trialIoFailure = true;
    }

    /** Records tool-reported cache I/O only when it did not originate in trial preparation. */
    void cacheIoFailure() {
        if (!trialIoFailure) reason = EngineTuningDiagnostic.Reason.CACHE_IO;
    }

    /** Records the tool's typed exact correctness mismatch, not a backend execution failure. */
    void correctnessMismatch() {
        reason = EngineTuningDiagnostic.Reason.CORRECTNESS_MISMATCH;
    }

    /** Reports fallback only after production preparation and final publication both succeeded. */
    void fallbackSucceeded() {
        emit(EngineTuningDiagnostic.Outcome.SAFE_HEURISTIC_FALLBACK, TraceLevel.WARN);
    }

    /** Reports policy rejection only after representative cleanup has succeeded. */
    void requiredTuningRejected() {
        emit(EngineTuningDiagnostic.Outcome.REJECTED, TraceLevel.ERROR);
    }

    /**
     * Allocates bounded immutable metadata, then contains only failures from the observer call.
     * Event ID zero belongs to this transaction's single-event stream; the clock domain is the
     * current JVM's {@link System#nanoTime()} domain. Allocation and producer failures propagate.
     *
     * @param outcome non-null terminal resolution
     * @param level non-null resolution severity
     */
    private void emit(EngineTuningDiagnostic.Outcome outcome, TraceLevel level) {
        if (emitted) return;
        emitted = true;
        var event = new TraceEvent<>(new TraceEventId(0), TracePhase.PREPARE, level,
                System.nanoTime(), new EngineTuningDiagnostic(outcome, phase, reason));
        try {
            observer.onEvent(event);
        } catch (RuntimeException | Error observerFailure) {
            // Passive observation alone must not alter the preparation or rejection result.
        }
    }
}
