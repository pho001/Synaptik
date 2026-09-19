package io.github.pho001.synaptik.tools.tuning;

import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;

/**
 * Caller-owned exact correctness collaboration for complete-plan candidates.
 *
 * <p>The first candidate captures an opaque reference. Later candidates compare against that
 * reference and reveal only a two-valued outcome. The caller owns reference representation,
 * aggregate-byte-limit enforcement, execution, publication copying, and cleanup. The tool never
 * inspects, compares, persists, or returns the reference.
 *
 * @param <C> immutable complete candidate-batch type
 * @param <K> immutable complete-plan candidate type
 * @param <R> opaque caller-owned same-transaction reference type
 */
public interface CompletePlanCorrectness<C extends BackendTuningCandidateBatch, K, R> {
    /** Exact comparison outcome after the caller has completed execution and cleanup. */
    enum Outcome {
        /** The candidate's canonical publication bytes exactly match the captured reference. */
        MATCH,
        /** The candidate's canonical publication bytes differ from the captured reference. */
        MISMATCH
    }

    /**
     * Executes the first candidate and captures its opaque reference.
     *
     * @param candidateBatch non-null immutable source batch
     * @param candidate non-null first candidate
     * @param maximumAggregateBytes non-negative aggregate correctness-byte ceiling enforced by
     *     the caller
     * @return non-null opaque reference retained only for this tuning transaction; ownership and
     *     disposal remain with the caller collaboration
     * @throws Exception if caller-owned execution, copying, validation, or cleanup fails
     */
    R capture(C candidateBatch, K candidate, long maximumAggregateBytes) throws Exception;

    /**
     * Executes and exactly compares one later candidate.
     *
     * @param reference non-null opaque reference returned by {@link #capture capture}
     * @param candidateBatch non-null immutable source batch
     * @param candidate non-null later candidate
     * @param maximumAggregateBytes non-negative aggregate correctness-byte ceiling enforced by
     *     the caller
     * @return non-null exact match or mismatch result after caller cleanup succeeds; a mismatch
     *     is converted by the tool to its typed mismatch failure before timing begins
     * @throws Exception if caller-owned execution, copying, validation, or cleanup fails
     */
    Outcome compare(R reference, C candidateBatch, K candidate, long maximumAggregateBytes)
            throws Exception;
}
