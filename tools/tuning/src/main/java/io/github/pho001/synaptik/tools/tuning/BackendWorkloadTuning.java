package io.github.pho001.synaptik.tools.tuning;

import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.util.List;
import java.util.Optional;

/**
 * Supplies the backend-owned operations needed to tune one opaque complete candidate batch.
 * Implementations define compatibility and identity bytes and remain responsible for constructing,
 * encoding, and compatibly decoding their own decisions. The tuning tool invokes this collaboration
 * only at the caller-supplied cold boundary; it does not inspect candidate or decision fields.
 *
 * @param <C> immutable complete candidate-batch type
 * @param <D> immutable selected-decision type
 * @param <K> immutable complete-candidate type
 */
public interface BackendWorkloadTuning<
        C extends BackendTuningCandidateBatch,
        D extends BackendTuningDecision,
        K> {
    /**
     * Returns the complete ordered candidates for a cache miss.
     *
     * @param candidateBatch non-null backend-owned complete batch
     * @return the non-null, non-empty complete candidate list in deterministic encounter order;
     *     the caller retains ownership and the tuning transaction snapshots the list structure
     */
    List<K> candidates(C candidateBatch);

    /**
     * Returns the canonical compatibility identity for a batch.
     *
     * @param candidateBatch non-null backend-owned complete batch
     * @return a non-null immutable opaque compatibility identity whose exact value includes every
     *     backend-owned fact required for safe reuse
     */
    WorkloadTuningRequest.WorkloadCompatibility compatibility(C candidateBatch);

    /**
     * Returns one stable canonical candidate identity.
     *
     * @param candidate non-null backend-owned candidate
     * @return a non-null immutable opaque identity that is stable for repeated calls during the
     *     transaction and unique within the supplied batch
     */
    WorkloadTuningRequest.CandidateIdentity candidateIdentity(K candidate);

    /**
     * Constructs the decision for the selected member of the supplied batch.
     *
     * @param candidateBatch non-null backend-owned complete batch
     * @param candidate non-null exact selected candidate from that batch; the tool does not copy or
     *     mutate it
     * @return a non-null immutable backend-owned decision
     */
    D selectedDecision(C candidateBatch, K candidate);

    /**
     * Encodes one persistent decision without executable state.
     *
     * @param decision non-null backend-owned decision
     * @return a non-null, non-empty opaque encoding without executable state; callers snapshot the
     *     bytes and reject encodings larger than the cache bound
     */
    byte[] encodeDecision(D decision);

    /**
     * Decodes a decision only when it remains compatible with the current complete batch.
     *
     * @param candidateBatch non-null current backend-owned batch
     * @param encodedDecision non-null opaque bytes owned by the caller for the duration of the call;
     *     the implementation must not mutate them
     * @return a non-null optional containing an immutable compatible decision, or empty for an
     *     ordinary cache miss; the optional must not contain {@code null}
     */
    Optional<D> decodeCompatibleDecision(C candidateBatch, byte[] encodedDecision);
}
