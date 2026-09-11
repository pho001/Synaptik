package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.util.Objects;

/**
 * Immutable explicit selection reference for one CPU OpenBLAS tuning batch. It contains only the
 * exact schema, canonical workload signature, and deterministic candidate identity. It owns no
 * timing, objective, cache location, serialized data, executable, provider, or runtime state.
 * Its method-free {@link BackendTuningDecision} role permits shared Prepare code to transport the
 * exact decision opaquely; CPU remains its sole interpreter and validator.
 *
 * @param candidateSchemaVersion exact batch schema version
 * @param workload exact canonical workload signature used during selection
 * @param selectedCandidate exact deterministic selected identity
 */
public record CpuOpenBlasTuningDecision(int candidateSchemaVersion,
        CpuOpenBlasTuningBatch.WorkloadSignature workload,
        CpuOpenBlasTuningBatch.CandidateIdentity selectedCandidate)
        implements BackendTuningDecision {
    /**
     * Validates a structurally complete selection reference. Both immutable compatibility values
     * are retained by reference; construction owns no candidate plan or resource.
     *
     * @throws NullPointerException if {@code workload} or {@code selectedCandidate} is
     *     {@code null}
     * @throws IllegalArgumentException if {@code candidateSchemaVersion} is not positive
     */
    public CpuOpenBlasTuningDecision {
        Objects.requireNonNull(workload, "workload");
        Objects.requireNonNull(selectedCandidate, "selectedCandidate");
        if (candidateSchemaVersion <= 0) {
            throw new IllegalArgumentException("candidate schema version must be positive");
        }
    }

    /**
     * Tests exact schema, workload, and candidate membership against a fresh CPU batch.
     * @param batch non-null freshly generated CPU candidate batch
     * @return non-null optional containing the retained matching complete candidate, or empty
     *     when any compatibility fact differs; neither value is mutated
     * @throws NullPointerException if {@code batch} is {@code null}
     */
    public java.util.Optional<CpuOpenBlasTuningBatch.Candidate> match(
            CpuOpenBlasTuningBatch batch) {
        Objects.requireNonNull(batch, "batch");
        if (candidateSchemaVersion != batch.schemaVersion() || !workload.equals(batch.workload())) {
            return java.util.Optional.empty();
        }
        return batch.find(selectedCandidate);
    }
}
