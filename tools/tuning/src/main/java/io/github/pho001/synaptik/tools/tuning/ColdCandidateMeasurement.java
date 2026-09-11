package io.github.pho001.synaptik.tools.tuning;

import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;

/**
 * Executes one complete backend-owned candidate on equivalent representative inputs at the cold
 * tuning boundary. The caller owns input equivalence, correctness validation, execution resources,
 * and cleanup; the tuning tool owns only invocation count and elapsed-time measurement.
 *
 * @param <C> immutable complete candidate-batch type
 * @param <K> immutable complete-candidate type
 */
@FunctionalInterface
public interface ColdCandidateMeasurement<C extends BackendTuningCandidateBatch, K> {
    /**
     * Executes one complete comparable candidate once.
     *
     * @param candidateBatch non-null immutable batch that owns the candidate; ownership is retained
     *     by the caller
     * @param candidate non-null complete candidate from that batch; ownership is retained by the
     *     caller
     * @throws Exception when candidate execution cannot complete; the tuning transaction propagates
     *     the same failure and publishes no new cache state
     */
    void execute(C candidateBatch, K candidate) throws Exception;
}
