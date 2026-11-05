package io.github.pho001.synaptik.backend.cpu.internal.route;

import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.util.Objects;

/**
 * Immutable explicit local selection bound to its exact CPU batch, workload and typed identity.
 * The originating batch retains the chosen cold realization so Phase 2 can authenticate that
 * realization without repeating local route search. Cold plans are immutable declarations, not
 * executables or native resources; this value owns no timing, objective, cache location, provider,
 * physical workspace, or Runtime state.
 * Its method-free {@link BackendTuningDecision} role permits shared Prepare code to transport the
 * exact decision opaquely; CPU remains its sole interpreter and validator.
 *
 * @param candidateSchemaVersion exact batch schema version
 * @param workload exact canonical workload signature used during selection
 * @param selectedCandidate exact deterministic selected identity
 * @param batch originating immutable complete candidate batch retaining the cold realization
 */
public record CpuLocalTuningDecision(int candidateSchemaVersion,
        CpuLocalTuningBatch.WorkloadSignature workload,
        CpuLocalTuningBatch.CandidateIdentity selectedCandidate,
        CpuLocalTuningBatch batch)
        implements BackendTuningDecision {
    /**
     * Validates exact schema, workload and membership against the originating complete batch. Values
     * are retained by reference; construction acquires no executable or native resource.
     *
     * @throws NullPointerException if {@code workload}, {@code selectedCandidate}, or {@code batch} is
     *     {@code null}
     * @throws IllegalArgumentException if schema, workload or selected membership disagrees with the originating batch
     */
    public CpuLocalTuningDecision {
        Objects.requireNonNull(workload, "workload");
        Objects.requireNonNull(selectedCandidate, "selectedCandidate");
        Objects.requireNonNull(batch, "batch");
        if (candidateSchemaVersion != batch.schemaVersion() || !workload.equals(batch.workload())
                || batch.find(selectedCandidate).isEmpty()) {
            throw new IllegalArgumentException("selected realization identity disagrees");
        }
        if (candidateSchemaVersion <= 0) {
            throw new IllegalArgumentException("candidate schema version must be positive");
        }
    }

    /**
     * Returns the immutable authenticated cold realization, without repeating route discovery.
     * @return exact retained member of the originating batch
     */
    public CpuLocalTuningBatch.Candidate realization() {
        return batch.find(selectedCandidate).orElseThrow();
    }
    /**
     * Tests exact schema, workload, and candidate membership against a fresh CPU batch.
     * @param batch non-null freshly generated CPU candidate batch
     * @return non-null optional containing the retained matching complete candidate, or empty
     *     when any compatibility fact differs; neither value is mutated
     * @throws NullPointerException if {@code batch} is {@code null}
     */
    public java.util.Optional<CpuLocalTuningBatch.Candidate> match(
            CpuLocalTuningBatch batch) {
        Objects.requireNonNull(batch, "batch");
        if (candidateSchemaVersion != batch.schemaVersion() || !workload.equals(batch.workload())) {
            return java.util.Optional.empty();
        }
        return batch.find(selectedCandidate);
    }
}
