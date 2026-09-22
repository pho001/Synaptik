package io.github.pho001.synaptik.backend.metal.internal;

import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable Metal-owned selection reference for one compatible NEG candidate.
 *
 * <p>The value contains compatibility and identity only. It contains no measurement, cache,
 * executable, native handle, physical resource, or Runtime state.</p>
 *
 * @param candidateSchemaVersion exact candidate schema used for selection
 * @param compatibility exact canonical workload and live-session target compatibility
 * @param selectedCandidate exact stable candidate identity
 */
record MetalNegTuningDecision(
        int candidateSchemaVersion,
        MetalNegTuningBatch.Compatibility compatibility,
        MetalNegTuningBatch.Candidate selectedCandidate)
        implements BackendTuningDecision {
    /**
     * Validates one structurally complete selection without establishing current compatibility.
     *
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if the candidate schema is not positive
     */
    MetalNegTuningDecision {
        if (candidateSchemaVersion <= 0) {
            throw new IllegalArgumentException("candidateSchemaVersion must be positive");
        }
        Objects.requireNonNull(compatibility, "compatibility");
        Objects.requireNonNull(selectedCandidate, "selectedCandidate");
    }

    /**
     * Matches this untrusted selection against a freshly generated Metal batch.
     *
     * @param freshBatch non-null current batch generated from live analysis facts
     * @return matching candidate, or empty when schema, workload, target session, identity, or
     *     current membership differs
     * @throws NullPointerException if {@code freshBatch} is {@code null}
     */
    Optional<MetalNegTuningBatch.Candidate> match(MetalNegTuningBatch freshBatch) {
        Objects.requireNonNull(freshBatch, "freshBatch");
        if (candidateSchemaVersion != MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION
                || candidateSchemaVersion
                        != freshBatch.compatibility().candidateSchemaVersion()
                || !compatibility.equals(freshBatch.compatibility())) {
            return Optional.empty();
        }
        return freshBatch.find(selectedCandidate);
    }
}
