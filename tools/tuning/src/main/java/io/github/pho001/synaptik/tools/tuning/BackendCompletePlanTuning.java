package io.github.pho001.synaptik.tools.tuning;

import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.util.List;
import java.util.Optional;

/**
 * Typed collaboration for backend-owned complete-plan candidates and decisions.
 *
 * <p>The tuning tool treats every candidate and opaque identity as an indivisible value. The
 * producer remains responsible for candidate legality and completeness, stable encounter order,
 * compatibility meaning, reuse lifetime, decision construction, and codec compatibility. The
 * tool snapshots the returned candidate list and opaque byte values; it does not persist
 * candidate objects or infer backend meaning from them.
 *
 * @param <C> immutable complete candidate-batch type
 * @param <D> immutable selected-decision type
 * @param <K> immutable complete-plan candidate type
 */
public interface BackendCompletePlanTuning<
        C extends BackendTuningCandidateBatch,
        D extends BackendTuningDecision,
        K> {
    /**
     * Returns the complete stable candidate set.
     *
     * @param candidateBatch non-null immutable batch supplied in the request handoff
     * @return non-null, non-empty candidate list in stable encounter order; the tool snapshots
     *     the list structure but retains its candidate references
     * @throws RuntimeException if the producer cannot enumerate the complete legal set
     */
    List<K> candidates(C candidateBatch);

    /**
     * Returns the opaque compatibility and persistence identity for the batch.
     *
     * @param candidateBatch non-null immutable batch supplied in the request handoff
     * @return non-null immutable compatibility value, including producer, codec, and reuse-scope
     *     identity
     * @throws RuntimeException if compatibility cannot be derived
     */
    CompletePlanTuningRequest.PlanCompatibility compatibility(C candidateBatch);

    /**
     * Returns the stable opaque identity of one candidate.
     *
     * @param candidate non-null candidate returned by {@link #candidates candidates}
     * @return non-null immutable identity, stable across repeated calls and unique within the
     *     batch
     * @throws RuntimeException if the candidate does not belong to the producer's identity domain
     */
    CompletePlanTuningRequest.CandidateIdentity candidateIdentity(K candidate);

    /**
     * Constructs the decision selecting one candidate.
     *
     * @param candidateBatch non-null immutable source batch
     * @param candidate non-null winning candidate from that batch
     * @return non-null immutable selected decision associated with the supplied batch
     * @throws RuntimeException if the candidate is not selectable for the supplied batch
     */
    D selectedDecision(C candidateBatch, K candidate);

    /**
     * Encodes a decision for bounded persistent storage.
     *
     * @param decision non-null immutable decision produced by this collaboration
     * @return non-null, non-empty opaque bytes; the tool snapshots the array and rejects values
     *     above its persistence bound
     * @throws RuntimeException if the decision cannot be encoded
     */
    byte[] encodeDecision(D decision);

    /**
     * Decodes a decision only when it remains compatible with the current batch.
     *
     * @param candidateBatch non-null current immutable batch
     * @param encodedDecision non-null tool-owned opaque bytes that the producer must not retain or
     *     mutate
     * @return non-null optional containing an immutable compatible decision, or empty to treat the
     *     entry as a safe cache miss
     * @throws RuntimeException if decoding itself fails; incompatibility should be reported as
     *     an empty result
     */
    Optional<D> decodeCompatibleDecision(C candidateBatch, byte[] encodedDecision);
}
