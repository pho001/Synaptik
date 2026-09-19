package io.github.pho001.synaptik.tools.tuning;

import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;

/**
 * Performs one complete fresh preparation, execution, publication lifecycle, and cleanup for a
 * complete-plan candidate.
 *
 * <p>The tuning tool invokes this action only after every candidate has passed the separate
 * correctness phase. Each invocation is one warmup or one timed sample; the caller must create
 * fresh execution state as required by its lifecycle and must finish cleanup before returning.
 *
 * @param <C> immutable complete candidate-batch type
 * @param <K> immutable complete-plan candidate type
 */
@FunctionalInterface
public interface CompleteCandidateMeasurement<C extends BackendTuningCandidateBatch, K> {
    /**
     * Performs exactly one complete execution and all caller-owned cleanup.
     *
     * @param candidateBatch non-null immutable source batch
     * @param candidate non-null candidate from that batch
     * @throws Exception if preparation, execution, publication handling, or cleanup fails; the
     *     transaction stops and propagates the failure without publishing a result
     */
    void execute(C candidateBatch, K candidate) throws Exception;
}
