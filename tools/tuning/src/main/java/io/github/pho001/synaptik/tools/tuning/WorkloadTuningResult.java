package io.github.pho001.synaptik.tools.tuning;

import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Immutable selected handoffs and separate rich evidence from one successful tuning transaction.
 * Collection structure and raw samples are snapshotted. Cache-hit workload evidence contains no
 * candidate evidence; measured workload evidence contains a non-empty list whose raw samples
 * correspond exactly to each summary.
 *
 * @param <C> immutable complete candidate-batch type
 * @param <D> immutable selected-decision type
 */
public final class WorkloadTuningResult<
        C extends BackendTuningCandidateBatch,
        D extends BackendTuningDecision> {
    /** Identifies whether a workload was decoded from the cache or measured now. */
    public enum Source {
        /** A compatible decision and compact summary were reused without candidate measurement. */
        CACHE_HIT,
        /** Every complete candidate was measured during this transaction. */
        MEASURED
    }

    /**
     * Compact bounded timing summary retained by both evidence and persistent entries.
     *
     * @param minimumNanos non-negative minimum elapsed sample in nanoseconds
     * @param medianNanos non-negative integer median elapsed sample in nanoseconds
     * @param maximumNanos non-negative maximum elapsed sample in nanoseconds
     * @param sampleCount positive number of summarized timed samples
     */
    public record SampleSummary(long minimumNanos, long medianNanos, long maximumNanos, int sampleCount) {
        /**
         * Validates non-negative ordered values and a positive sample count.
         *
         * @throws IllegalArgumentException if the values are negative or out of order, or if the
         *     sample count is not positive
         */
        public SampleSummary {
            if (minimumNanos < 0 || minimumNanos > medianNanos || medianNanos > maximumNanos) {
                throw new IllegalArgumentException("sample summary values must be non-negative and ordered");
            }
            if (sampleCount <= 0) {
                throw new IllegalArgumentException("sampleCount must be positive");
            }
        }
    }

    /** Immutable evidence for one original occurrence. */
    public static final class OccurrenceEvidence {
        private final byte[] contextFingerprint;
        private final long weight;

        /**
         * Creates snapshotted occurrence evidence.
         *
         * @param contextFingerprint non-null, non-empty opaque context bytes; the array is
         *     snapshotted
         * @param weight positive occurrence weight
         * @throws NullPointerException if {@code contextFingerprint} is {@code null}
         * @throws IllegalArgumentException if the fingerprint is empty or the weight is not
         *     positive
         */
        public OccurrenceEvidence(byte[] contextFingerprint, long weight) {
            Objects.requireNonNull(contextFingerprint, "contextFingerprint");
            if (contextFingerprint.length == 0) {
                throw new IllegalArgumentException("contextFingerprint must not be empty");
            }
            if (weight <= 0) {
                throw new IllegalArgumentException("weight must be positive");
            }
            this.contextFingerprint = contextFingerprint.clone();
            this.weight = weight;
        }

        /**
         * Returns the occurrence-specific evidence context.
         *
         * @return a fresh caller-owned copy of the non-empty context fingerprint
         */
        public byte[] contextFingerprint() { return contextFingerprint.clone(); }
        /**
         * Returns this occurrence's caller-defined multiplicity.
         *
         * @return the positive occurrence weight
         */
        public long weight() { return weight; }
    }

    /**
     * Rich evidence for one measured candidate.
     *
     * @param identity non-null opaque backend-owned identity
     * @param elapsedSamplesNanos non-null, non-empty raw elapsed samples in encounter order; every
     *     element must be non-null and non-negative and the list is snapshotted without reordering
     * @param summary non-null compact summary whose count, minimum, integer-middle median of the
     *     sorted samples, and maximum correspond exactly to the raw samples
     */
    public record CandidateEvidence(
            WorkloadTuningRequest.CandidateIdentity identity,
            List<Long> elapsedSamplesNanos,
            SampleSummary summary) {
        /**
         * Snapshots the raw encounter-order sample list and validates all components.
         *
         * @throws NullPointerException if {@code identity}, {@code elapsedSamplesNanos}, or
         *     {@code summary} is {@code null}
         * @throws IllegalArgumentException if the sample list is empty, a sample element is
         *     {@code null} or negative, or its exact count/minimum/integer-middle-median/maximum
         *     does not match {@code summary}
         */
        public CandidateEvidence {
            Objects.requireNonNull(identity, "identity");
            Objects.requireNonNull(summary, "summary");
            var samples = new ArrayList<>(
                    Objects.requireNonNull(elapsedSamplesNanos, "elapsedSamplesNanos"));
            if (samples.isEmpty()) {
                throw new IllegalArgumentException("elapsed samples must not be empty");
            }
            for (Long sample : samples) {
                if (sample == null || sample < 0) {
                    throw new IllegalArgumentException(
                            "elapsed samples must be non-null and non-negative");
                }
            }
            elapsedSamplesNanos = List.copyOf(samples);
            if (samples.size() != summary.sampleCount()) {
                throw new IllegalArgumentException("raw sample count does not match summary");
            }
            samples.sort(Long::compare);
            int middle = samples.size() / 2;
            if (summary.minimumNanos() != samples.getFirst()
                    || summary.medianNanos() != samples.get(middle)
                    || summary.maximumNanos() != samples.getLast()) {
                throw new IllegalArgumentException("raw samples do not match summary values");
            }
        }
    }

    /**
     * Evidence for one deduplicated compatible workload.
     *
     * @param compatibility non-null canonical backend-owned compatibility identity
     * @param totalWeight positive checked sum of occurrence weights
     * @param occurrences immutable non-empty occurrence evidence in request order
     * @param source non-null cache-hit or measured source
     * @param candidates immutable measured-candidate evidence, or empty for a cache hit
     * @param winnerIdentity non-null opaque winning-candidate identity
     * @param winnerSummary non-null compact summary for the winner
     */
    public record WorkloadEvidence(
            WorkloadTuningRequest.WorkloadCompatibility compatibility,
            long totalWeight,
            List<OccurrenceEvidence> occurrences,
            Source source,
            List<CandidateEvidence> candidates,
            WorkloadTuningRequest.CandidateIdentity winnerIdentity,
            SampleSummary winnerSummary) {
        /**
         * Snapshots collections and validates the complete evidence relationship.
         *
         * @throws NullPointerException if a component or collection element is {@code null}
         * @throws IllegalArgumentException if total weight is not positive, occurrences are empty,
         *     measured evidence has no candidates, or cache-hit evidence has candidate entries
         */
        public WorkloadEvidence {
            Objects.requireNonNull(compatibility, "compatibility");
            if (totalWeight <= 0) {
                throw new IllegalArgumentException("totalWeight must be positive");
            }
            occurrences = List.copyOf(Objects.requireNonNull(occurrences, "occurrences"));
            if (occurrences.isEmpty()) {
                throw new IllegalArgumentException("occurrences must not be empty");
            }
            Objects.requireNonNull(source, "source");
            candidates = List.copyOf(Objects.requireNonNull(candidates, "candidates"));
            if ((source == Source.MEASURED) != !candidates.isEmpty()) {
                throw new IllegalArgumentException(
                        "measured evidence must contain candidates and cache hits must not");
            }
            Objects.requireNonNull(winnerIdentity, "winnerIdentity");
            Objects.requireNonNull(winnerSummary, "winnerSummary");
        }
    }

    /**
     * Evidence identity and policy plus ordered deduplicated workload results.
     *
     * @param modelFingerprint non-null immutable caller-defined model evidence identity
     * @param profileFingerprint non-null immutable representative-profile evidence identity
     * @param objective non-null selection objective
     * @param budget non-null validated sampling budget
     * @param workloads immutable non-empty deduplicated workload evidence in encounter order
     */
    public record Evidence(
            WorkloadTuningRequest.ModelFingerprint modelFingerprint,
            WorkloadTuningRequest.ProfileFingerprint profileFingerprint,
            WorkloadTuningRequest.Objective objective,
            WorkloadTuningRequest.Budget budget,
            List<WorkloadEvidence> workloads) {
        /**
         * Snapshots workload evidence and validates all non-null components.
         *
         * @throws NullPointerException if a component or workload element is {@code null}
         * @throws IllegalArgumentException if {@code workloads} is empty
         */
        public Evidence {
            Objects.requireNonNull(modelFingerprint, "modelFingerprint");
            Objects.requireNonNull(profileFingerprint, "profileFingerprint");
            Objects.requireNonNull(objective, "objective");
            Objects.requireNonNull(budget, "budget");
            workloads = List.copyOf(Objects.requireNonNull(workloads, "workloads"));
            if (workloads.isEmpty()) {
                throw new IllegalArgumentException("workloads must not be empty");
            }
        }
    }

    private final List<BackendPartitionTuningHandoff<C, D>> selectedHandoffs;
    private final Evidence evidence;

    /**
     * Creates a successful immutable result.
     *
     * @param selectedHandoffs non-null, non-empty ordered output handoffs
     * @param evidence non-null rich evidence kept separate from persistent cache state
     * @throws NullPointerException if an argument or selected-handoff element is {@code null}
     * @throws IllegalArgumentException if {@code selectedHandoffs} is empty
     */
    public WorkloadTuningResult(
            List<BackendPartitionTuningHandoff<C, D>> selectedHandoffs,
            Evidence evidence) {
        this.selectedHandoffs = List.copyOf(
                Objects.requireNonNull(selectedHandoffs, "selectedHandoffs"));
        if (this.selectedHandoffs.isEmpty()) {
            throw new IllegalArgumentException("selectedHandoffs must not be empty");
        }
        this.evidence = Objects.requireNonNull(evidence, "evidence");
    }

    /**
     * Returns the selected decisions associated with the original Prepare handoffs.
     *
     * @return the immutable non-empty selected handoffs in original occurrence order
     */
    public List<BackendPartitionTuningHandoff<C, D>> selectedHandoffs() { return selectedHandoffs; }
    /**
     * Returns the evidence identity, policy, and deduplicated workload outcomes.
     *
     * @return the non-null immutable rich evidence
     */
    public Evidence evidence() { return evidence; }
}
