package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.trace.TracePayload;
import java.util.Objects;

/**
 * Immutable terminal facts about a recoverable ordinary Engine autotuning failure.
 *
 * <p>The producer reports a successful safe heuristic fallback, or rejection because the request
 * requires a tuned result, only after representative cleanup succeeds. The diagnostic identifies
 * the failed tuning phase, not a candidate, graph, cache entry, or invocation. It contains no
 * throwable, tensor values, paths, identities, or opaque cache bytes. Equality, hashing, and text
 * use ordinary record component semantics. This passive value never selects a policy or route.</p>
 *
 * @param outcome non-null terminal result of resolving the recoverable tuning failure
 * @param tuningPhase non-null tuning phase in which the failure occurred
 * @param reason non-null producer-known failure category; unclassified failures use
 *     {@link Reason#OTHER_TUNING_FAILURE}
 */
public record EngineTuningDiagnostic(Outcome outcome, TuningPhase tuningPhase, Reason reason)
        implements TracePayload {
    /** Terminal resolution after successful representative cleanup. */
    public enum Outcome {
        /** One fresh ordinary preparation was successfully published without tuning evidence. */
        SAFE_HEURISTIC_FALLBACK,
        /** The request required a tuned result, so the original tuning failure is rethrown. */
        REJECTED
    }

    /** Closed set of tuning phases, distinct from the enclosing PREPARE trace phase. */
    public enum TuningPhase {
        /** Local-workload handoff, cache coordination, measurement, or authentication failed. */
        LOCAL_WORKLOAD,
        /** Complete-plan handoff, correctness, cache coordination, or selection failed. */
        COMPLETE_PLAN
    }

    /** Producer-assigned categories; diagnostic text is never used to classify a failure. */
    public enum Reason {
        /** The selected backend supplied no eligible handoff for this tuning phase. */
        NO_ELIGIBLE_HANDOFF,
        /** Tuning-tool cache loading, validation, encoding, or publication reported I/O failure. */
        CACHE_IO,
        /** Complete-plan candidates differed under the exact canonical-byte comparison. */
        CORRECTNESS_MISMATCH,
        /** A recoverable failure has no more specific producer-known category, including budgets. */
        OTHER_TUNING_FAILURE
    }

    /**
     * Creates passive terminal metadata without retaining any producer-domain objects.
     *
     * @param outcome non-null terminal resolution
     * @param tuningPhase non-null phase that failed
     * @param reason non-null category assigned at the failure boundary
     * @throws NullPointerException if any argument is null
     */
    public EngineTuningDiagnostic {
        Objects.requireNonNull(outcome, "outcome");
        Objects.requireNonNull(tuningPhase, "tuningPhase");
        Objects.requireNonNull(reason, "reason");
    }

    /**
     * Returns the terminal resolution, not the tuning failure itself.
     *
     * @return the non-null immutable resolution
     */
    public Outcome outcome() {
        return outcome;
    }

    /**
     * Returns the tuning phase whose recoverable failure was resolved.
     *
     * @return the non-null immutable tuning phase
     */
    public TuningPhase tuningPhase() {
        return tuningPhase;
    }

    /**
     * Returns the failure category known at the producer boundary.
     *
     * @return the non-null category; the general category makes no more specific cause claim
     */
    public Reason reason() {
        return reason;
    }
}
