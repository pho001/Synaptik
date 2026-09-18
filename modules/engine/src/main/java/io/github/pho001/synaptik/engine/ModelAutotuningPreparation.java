package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable public result of one complete tuned or safe-heuristic preparation.
 *
 * <p>The contained handle is freshly prepared for the exact compiled graph and Engine. Evidence
 * is present exactly for {@link Outcome#TUNED}; fallback is always explicit and carries none.
 * Metadata remains readable after Engine closure, but the owner rejects later execution.</p>
 */
public final class ModelAutotuningPreparation {
    private final PreparedExecution preparedExecution;
    private final Outcome outcome;
    private final Optional<Evidence> evidence;

    /**
     * Creates the package-owned result after preparation and evidence authentication complete.
     *
     * @param preparedExecution non-null fresh owner-bound production handle, retained exactly
     * @param outcome non-null outcome governing evidence presence
     * @param evidence non-null optional immutable evidence; present exactly for {@code TUNED}
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if outcome and evidence presence disagree
     */
    ModelAutotuningPreparation(
            PreparedExecution preparedExecution, Outcome outcome, Optional<Evidence> evidence) {
        this.preparedExecution = Objects.requireNonNull(preparedExecution, "preparedExecution");
        this.outcome = Objects.requireNonNull(outcome, "outcome");
        this.evidence = Objects.requireNonNull(evidence, "evidence");
        if ((outcome == Outcome.TUNED) != evidence.isPresent()) {
            throw new IllegalArgumentException("tuned outcome requires evidence and fallback forbids it");
        }
    }

    /** Returns the production handle.
     * @return the exact fresh non-null owner-bound handle
     */
    public PreparedExecution preparedExecution() { return preparedExecution; }
    /** Identifies how the handle was selected.
     * @return the non-null preparation outcome
     */
    public Outcome outcome() { return outcome; }
    /** Returns authenticated tuning evidence.
     * @return evidence present exactly for {@code TUNED}
     */
    public Optional<Evidence> evidence() { return evidence; }

    /** Identifies whether the handle uses an authenticated tuning decision or safe heuristics. */
    public enum Outcome {
        /** An authenticated tuning decision selected the fresh production preparation. */ TUNED,
        /** Allowed recovery used a fresh ordinary safe-heuristic preparation. */ SAFE_HEURISTIC_FALLBACK
    }
    /** Identifies whether winner evidence was reused from cache or measured in this request. */
    public enum Source {
        /** A compatible persistent entry supplied the winner without trial execution. */ CACHE_HIT,
        /** This request measured the reported candidates. */ MEASURED
    }
    /** Describes the backend-declared lifetime over which compatibility may be reused. */
    public enum ReuseScope {
        /** Compatibility is confined to the current backend session. */ SESSION,
        /** Compatibility may be persisted when all backend checks pass. */ PERSISTENT
    }

    /**
     * Immutable authenticated evidence for the request and its ordered tuned workloads.
     *
     * @param modelIdentity exact caller-defined model identity from the request
     * @param representativeProfile exact caller-defined profile identity from configuration
     * @param objective exact requested selection objective
     * @param budget exact requested bounded measurement budget
     * @param workloads non-empty immutable ordered workload evidence snapshot
     */
    public record Evidence(
            ModelAutotuningRequest.ModelIdentity modelIdentity,
            ModelAutotuningConfig.RepresentativeProfileIdentity representativeProfile,
            ModelAutotuningConfig.Objective objective,
            ModelAutotuningConfig.Budget budget,
            List<WorkloadEvidence> workloads) {
        /** Validates and snapshots one complete evidence value.
         * @throws NullPointerException if any component or workload element is {@code null}
         * @throws IllegalArgumentException if {@code workloads} is empty
         */
        public Evidence {
            Objects.requireNonNull(modelIdentity, "modelIdentity");
            Objects.requireNonNull(representativeProfile, "representativeProfile");
            Objects.requireNonNull(objective, "objective");
            Objects.requireNonNull(budget, "budget");
            workloads = List.copyOf(Objects.requireNonNull(workloads, "workloads"));
            if (workloads.isEmpty()) throw new IllegalArgumentException("workloads must not be empty");
        }
    }

    /**
     * Evidence for one compatible local workload and its selected winner.
     *
     * @param compatibility opaque backend compatibility identity; not selection authority
     * @param totalWeight positive sum of occurrence weights
     * @param occurrences non-empty immutable occurrence snapshot
     * @param source whether the winner came from cache or current measurement
     * @param candidates immutable encounter-ordered measurements; non-empty only for measured data
     * @param winnerIdentity non-null opaque selected-candidate identity
     * @param winnerSummary non-null winner timing summary in nanoseconds
     */
    public record WorkloadEvidence(
            CompatibilityIdentity compatibility,
            long totalWeight,
            List<OccurrenceEvidence> occurrences,
            Source source,
            List<CandidateEvidence> candidates,
            CandidateIdentity winnerIdentity,
            SampleSummary winnerSummary) {
        /** Validates and snapshots one complete workload-evidence value.
         * @throws NullPointerException if any reference component or list element is {@code null}
         * @throws IllegalArgumentException if weight is non-positive, occurrences are empty, or
         *     candidate presence disagrees with the source
         */
        public WorkloadEvidence {
            Objects.requireNonNull(compatibility, "compatibility");
            if (totalWeight <= 0) throw new IllegalArgumentException("totalWeight must be positive");
            occurrences = List.copyOf(Objects.requireNonNull(occurrences, "occurrences"));
            if (occurrences.isEmpty()) throw new IllegalArgumentException("occurrences must not be empty");
            Objects.requireNonNull(source, "source");
            candidates = List.copyOf(Objects.requireNonNull(candidates, "candidates"));
            if ((source == Source.MEASURED) != !candidates.isEmpty()) {
                throw new IllegalArgumentException("measured evidence requires candidates and cache hits forbid them");
            }
            Objects.requireNonNull(winnerIdentity, "winnerIdentity");
            Objects.requireNonNull(winnerSummary, "winnerSummary");
        }
    }

    /**
     * Identifies one weighted occurrence contributing to a local workload.
     *
     * @param occurrenceIndex non-negative model-local occurrence index
     * @param partitionIndex non-negative planned-partition index
     * @param weight positive occurrence weight
     * @param contextIdentity non-null immutable context identity
     */
    public record OccurrenceEvidence(
            int occurrenceIndex, int partitionIndex, long weight, ContextIdentity contextIdentity) {
        /** Validates the non-negative indexes, positive weight, and context identity.
         * @throws NullPointerException if {@code contextIdentity} is {@code null}
         * @throws IllegalArgumentException if an index is negative or weight is non-positive
         */
        public OccurrenceEvidence {
            if (occurrenceIndex < 0) throw new IllegalArgumentException("occurrenceIndex must be non-negative");
            if (partitionIndex < 0) throw new IllegalArgumentException("partitionIndex must be non-negative");
            if (weight <= 0) throw new IllegalArgumentException("weight must be positive");
            Objects.requireNonNull(contextIdentity, "contextIdentity");
        }
    }

    /**
     * Encounter-ordered raw timed samples and verified summary for one measured candidate.
     *
     * @param identity non-null opaque candidate identity
     * @param elapsedSamplesNanos non-empty immutable list of non-negative nanosecond samples
     * @param summary summary whose count, minimum, integer-middle median, and maximum match samples
     */
    public record CandidateEvidence(
            CandidateIdentity identity, List<Long> elapsedSamplesNanos, SampleSummary summary) {
        /** Validates, snapshots, and cross-checks raw samples against their summary.
         * @throws NullPointerException if a reference component is {@code null}
         * @throws IllegalArgumentException if samples are empty, contain a null or negative value,
         *     or do not exactly match the supplied summary
         */
        public CandidateEvidence {
            Objects.requireNonNull(identity, "identity");
            Objects.requireNonNull(summary, "summary");
            var copy = new ArrayList<>(Objects.requireNonNull(elapsedSamplesNanos, "elapsedSamplesNanos"));
            if (copy.isEmpty()) throw new IllegalArgumentException("elapsed samples must not be empty");
            for (Long sample : copy) if (sample == null || sample < 0) {
                throw new IllegalArgumentException("elapsed samples must be non-null and non-negative");
            }
            elapsedSamplesNanos = List.copyOf(copy);
            if (copy.size() != summary.sampleCount()) throw new IllegalArgumentException("sample count mismatch");
            copy.sort(Long::compare);
            if (summary.minimumNanos() != copy.getFirst()
                    || summary.medianNanos() != copy.get(copy.size() / 2)
                    || summary.maximumNanos() != copy.getLast()) {
                throw new IllegalArgumentException("samples do not match summary");
            }
        }
    }

    /**
     * Immutable nanosecond timing summary.
     *
     * @param minimumNanos non-negative minimum
     * @param medianNanos non-negative integer-middle median between minimum and maximum
     * @param maximumNanos non-negative maximum
     * @param sampleCount positive timed-sample count
     */
    public record SampleSummary(long minimumNanos, long medianNanos, long maximumNanos, int sampleCount) {
        /** Validates ordered non-negative statistics and a positive sample count.
         * @throws IllegalArgumentException if values are negative or unordered, or count is not
         *     positive
         */
        public SampleSummary {
            if (minimumNanos < 0 || minimumNanos > medianNanos || medianNanos > maximumNanos) {
                throw new IllegalArgumentException("sample values must be non-negative and ordered");
            }
            if (sampleCount <= 0) throw new IllegalArgumentException("sampleCount must be positive");
        }
    }

    /** Immutable opaque backend compatibility identity; it records but does not grant reuse. */
    public static final class CompatibilityIdentity {
        private final int schemaVersion; private final byte[] bytes; private final ReuseScope reuseScope;
        /** Creates an opaque compatibility label.
         *
         * @param schemaVersion positive backend-owned schema version
         * @param bytes non-null non-empty opaque bytes, defensively copied
         * @param reuseScope non-null backend-declared reuse scope
         * @throws NullPointerException if bytes or reuse scope is null
         * @throws IllegalArgumentException if version is non-positive or bytes are empty
         */
        public CompatibilityIdentity(int schemaVersion, byte[] bytes, ReuseScope reuseScope) {
            if (schemaVersion <= 0) throw new IllegalArgumentException("schemaVersion must be positive");
            Objects.requireNonNull(bytes, "bytes");
            if (bytes.length == 0) throw new IllegalArgumentException("bytes must not be empty");
            this.schemaVersion = schemaVersion; this.bytes = bytes.clone();
            this.reuseScope = Objects.requireNonNull(reuseScope, "reuseScope");
        }
        /** Returns the compatibility schema.
         * @return the positive backend-owned version
         */
        public int schemaVersion() { return schemaVersion; }
        /** Copies the compatibility payload.
         * @return a fresh non-empty byte array
         */
        public byte[] bytes() { return bytes.clone(); }
        /** Returns the declared compatibility lifetime.
         * @return the non-null reuse scope
         */
        public ReuseScope reuseScope() { return reuseScope; }
        @Override public boolean equals(Object other) { return this == other || other instanceof CompatibilityIdentity that && schemaVersion == that.schemaVersion && reuseScope == that.reuseScope && Arrays.equals(bytes, that.bytes); }
        @Override public int hashCode() { return 31 * (31 * schemaVersion + Arrays.hashCode(bytes)) + reuseScope.hashCode(); }
        @Override public String toString() { return "CompatibilityIdentity[schemaVersion=" + schemaVersion + ", byteLength=" + bytes.length + ", reuseScope=" + reuseScope + "]"; }
    }

    /** Immutable versioned identity for one occurrence's model-local context. */
    public static final class ContextIdentity {
        private final int schemaVersion; private final byte[] bytes;
        /** Creates an occurrence-context identity.
         *
         * @param schemaVersion positive Engine-owned context schema version
         * @param bytes non-null non-empty opaque bytes, defensively copied
         * @throws NullPointerException if bytes are null
         * @throws IllegalArgumentException if version is non-positive or bytes are empty
         */
        public ContextIdentity(int schemaVersion, byte[] bytes) {
            if (schemaVersion <= 0) throw new IllegalArgumentException("schemaVersion must be positive");
            Objects.requireNonNull(bytes, "bytes");
            if (bytes.length == 0) throw new IllegalArgumentException("bytes must not be empty");
            this.schemaVersion = schemaVersion; this.bytes = bytes.clone();
        }
        /** Returns the context schema.
         * @return the positive Engine-owned version
         */
        public int schemaVersion() { return schemaVersion; }
        /** Copies the context payload.
         * @return a fresh non-empty byte array
         */
        public byte[] bytes() { return bytes.clone(); }
        @Override public boolean equals(Object other) { return this == other || other instanceof ContextIdentity that && schemaVersion == that.schemaVersion && Arrays.equals(bytes, that.bytes); }
        @Override public int hashCode() { return 31 * schemaVersion + Arrays.hashCode(bytes); }
        @Override public String toString() { return "ContextIdentity[schemaVersion=" + schemaVersion + ", byteLength=" + bytes.length + "]"; }
    }

    /** Immutable opaque identity for one backend-owned tuning candidate. */
    public static final class CandidateIdentity {
        private final byte[] bytes;
        /** Creates an opaque candidate identity.
         *
         * @param bytes non-null non-empty opaque bytes, defensively copied
         * @throws NullPointerException if bytes are null
         * @throws IllegalArgumentException if bytes are empty
         */
        public CandidateIdentity(byte[] bytes) {
            Objects.requireNonNull(bytes, "bytes");
            if (bytes.length == 0) throw new IllegalArgumentException("bytes must not be empty");
            this.bytes = bytes.clone();
        }
        /** Copies the candidate payload.
         * @return a fresh non-empty byte array
         */
        public byte[] bytes() { return bytes.clone(); }
        @Override public boolean equals(Object other) { return this == other || other instanceof CandidateIdentity that && Arrays.equals(bytes, that.bytes); }
        @Override public int hashCode() { return Arrays.hashCode(bytes); }
        @Override public String toString() { return "CandidateIdentity[byteLength=" + bytes.length + "]"; }
    }
}
