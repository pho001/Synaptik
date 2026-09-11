package io.github.pho001.synaptik.tools.tuning;

import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Immutable validated inputs for one exact/default cold workload-tuning transaction.
 * Caller-owned byte sequences and collection structure are snapshotted; exact immutable Prepare
 * handoff references and the explicit cache path are retained. Model and profile fingerprints
 * identify returned evidence and do not participate in workload-cache matching.
 *
 * @param <C> immutable complete candidate-batch type
 * @param <D> immutable selected-decision type
 */
public final class WorkloadTuningRequest<
        C extends BackendTuningCandidateBatch,
        D extends BackendTuningDecision> {
    /** The sole objective supported by the initial workflow. */
    public enum Objective {
        /** Selects the first encountered candidate having the lowest integer-middle median. */
        MIN_MEDIAN_ELAPSED_NANOS
    }

    /** Controls whether an identical result may be reused only now or through the cache file. */
    public enum ReuseScope {
        /** Reuse identical occurrences only within this invocation. */
        SESSION,
        /** Permit reuse from and publication to the explicit persistent workload cache. */
        PERSISTENT
    }

    /** Stable caller-defined model evidence identity. */
    public static final class ModelFingerprint extends OpaqueFingerprint {
        /**
         * Creates a snapshotted model fingerprint.
         *
         * @param schemaVersion positive caller-owned schema version
         * @param bytes non-null, non-empty stable opaque bytes; the array is snapshotted
         * @throws NullPointerException if {@code bytes} is {@code null}
         * @throws IllegalArgumentException if the schema version is not positive or the array is
         *     empty
         */
        public ModelFingerprint(int schemaVersion, byte[] bytes) {
            super(schemaVersion, bytes, "model fingerprint");
        }
    }

    /** Stable caller-defined representative-profile evidence identity. */
    public static final class ProfileFingerprint extends OpaqueFingerprint {
        /**
         * Creates a snapshotted representative-profile fingerprint.
         *
         * @param schemaVersion positive caller-owned schema version
         * @param bytes non-null, non-empty stable opaque bytes; the array is snapshotted
         * @throws NullPointerException if {@code bytes} is {@code null}
         * @throws IllegalArgumentException if the schema version is not positive or the array is
         *     empty
         */
        public ProfileFingerprint(int schemaVersion, byte[] bytes) {
            super(schemaVersion, bytes, "profile fingerprint");
        }
    }

    /** Backend-owned canonical compatibility identity compared only by exact value. */
    public static final class WorkloadCompatibility extends OpaqueFingerprint {
        private final ReuseScope reuseScope;

        /**
         * Creates a snapshotted compatibility value.
         *
         * @param keySchemaVersion positive backend-owned key-schema version
         * @param bytes non-null, non-empty canonical opaque bytes; the array is snapshotted
         * @param reuseScope non-null reuse scope
         * @throws NullPointerException if {@code bytes} or {@code reuseScope} is {@code null}
         * @throws IllegalArgumentException if the key schema version is not positive or the byte
         *     array is empty
         */
        public WorkloadCompatibility(
                int keySchemaVersion, byte[] bytes, ReuseScope reuseScope) {
            super(keySchemaVersion, bytes, "workload compatibility");
            this.reuseScope = Objects.requireNonNull(reuseScope, "reuseScope");
        }

        /**
         * Returns whether this compatibility value permits persistent reuse.
         *
         * @return the non-null retained reuse scope
         */
        public ReuseScope reuseScope() {
            return reuseScope;
        }

        @Override
        public boolean equals(java.lang.Object other) {
            return this == other || other instanceof WorkloadCompatibility that
                    && sameFingerprint(that) && reuseScope == that.reuseScope;
        }

        @Override
        public int hashCode() {
            return 31 * fingerprintHashCode() + reuseScope.hashCode();
        }

        @Override
        public String toString() {
            return "WorkloadCompatibility[keySchemaVersion=" + schemaVersion()
                    + ", byteLength=" + byteLength() + ", reuseScope=" + reuseScope + "]";
        }
    }

    /** Backend-owned canonical candidate identity compared only by exact value. */
    public static final class CandidateIdentity extends OpaqueBytes {
        /**
         * Creates a snapshotted candidate identity.
         *
         * @param bytes non-null, non-empty canonical opaque bytes; the array is snapshotted
         * @throws NullPointerException if {@code bytes} is {@code null}
         * @throws IllegalArgumentException if {@code bytes} is empty
         */
        public CandidateIdentity(byte[] bytes) {
            super(bytes, "candidate identity");
        }

        @Override
        public String toString() {
            return "CandidateIdentity[byteLength=" + byteLength() + "]";
        }
    }

    /**
     * Complete bounded sampling policy validated before any candidate execution.
     *
     * @param maximumDistinctCacheMisses positive maximum deduplicated misses
     * @param maximumCandidatesPerMiss positive maximum complete candidates for each miss
     * @param warmupCount non-negative complete executions before timed sampling
     * @param timedSampleCount positive odd number of timed complete executions
     */
    public record Budget(
            int maximumDistinctCacheMisses,
            int maximumCandidatesPerMiss,
            int warmupCount,
            int timedSampleCount) {
        /**
         * Validates positive bounds, non-negative warmups, and a positive odd sample count.
         *
         * @throws IllegalArgumentException if either maximum is not positive, the warmup count is
         *     negative, or the timed sample count is not positive and odd
         */
        public Budget {
            if (maximumDistinctCacheMisses <= 0) {
                throw new IllegalArgumentException("maximumDistinctCacheMisses must be positive");
            }
            if (maximumCandidatesPerMiss <= 0) {
                throw new IllegalArgumentException("maximumCandidatesPerMiss must be positive");
            }
            if (warmupCount < 0) {
                throw new IllegalArgumentException("warmupCount must be non-negative");
            }
            if (timedSampleCount <= 0 || (timedSampleCount & 1) == 0) {
                throw new IllegalArgumentException("timedSampleCount must be positive and odd");
            }
        }
    }

    /**
     * One original occurrence and its evidence context.
     *
     * @param <C> immutable complete candidate-batch type
     * @param <D> immutable selected-decision type
     */
    public static final class Occurrence<
            C extends BackendTuningCandidateBatch,
            D extends BackendTuningDecision> {
        private final BackendPartitionTuningHandoff<C, D> handoff;
        private final long weight;
        private final byte[] contextFingerprint;

        /**
         * Creates an occurrence retaining the exact handoff while snapshotting its context bytes.
         *
         * @param handoff non-null exact handoff whose input decision must be empty
         * @param weight positive occurrence weight
         * @param contextFingerprint non-null, non-empty opaque context bytes; the array is
         *     snapshotted
         * @throws NullPointerException if {@code handoff} or {@code contextFingerprint} is
         *     {@code null}
         * @throws IllegalArgumentException if the handoff already contains a decision, the weight
         *     is not positive, or the context fingerprint is empty
         */
        public Occurrence(
                BackendPartitionTuningHandoff<C, D> handoff,
                long weight,
                byte[] contextFingerprint) {
            this.handoff = Objects.requireNonNull(handoff, "handoff");
            if (handoff.selectedDecision().isPresent()) {
                throw new IllegalArgumentException("input handoff selected decision must be empty");
            }
            if (weight <= 0) {
                throw new IllegalArgumentException("weight must be positive");
            }
            this.weight = weight;
            this.contextFingerprint = snapshot(contextFingerprint, "context fingerprint");
        }

        /**
         * Returns the occurrence's opaque Prepare handoff.
         *
         * @return the non-null exact supplied immutable handoff; ownership remains with the caller
         */
        public BackendPartitionTuningHandoff<C, D> handoff() {
            return handoff;
        }

        /**
         * Returns this occurrence's caller-defined multiplicity.
         *
         * @return the positive occurrence weight
         */
        public long weight() {
            return weight;
        }

        /**
         * Returns the occurrence-specific evidence context.
         *
         * @return a fresh caller-owned copy of the non-empty context fingerprint
         */
        public byte[] contextFingerprint() {
            return contextFingerprint.clone();
        }
    }

    private final ModelFingerprint modelFingerprint;
    private final ProfileFingerprint profileFingerprint;
    private final List<Occurrence<C, D>> occurrences;
    private final Objective objective;
    private final Budget budget;
    private final Path workloadCache;

    /**
     * Creates one immutable validated request.
     *
     * @param modelFingerprint non-null stable caller-defined model evidence identity
     * @param profileFingerprint non-null stable caller-defined representative-profile identity
     * @param occurrences non-null, non-empty ordered occurrences; list structure is snapshotted
     *     and elements must be non-null
     * @param objective non-null and currently {@link Objective#MIN_MEDIAN_ELAPSED_NANOS}
     * @param budget non-null complete-work sampling budget
     * @param workloadCache non-null explicit workload-cache file path retained by reference; its
     *     parent directory must exist when publication is required
     * @throws NullPointerException if any argument or occurrence is {@code null}
     * @throws IllegalArgumentException if {@code occurrences} is empty or the objective is not the
     *     supported initial objective
     */
    public WorkloadTuningRequest(
            ModelFingerprint modelFingerprint,
            ProfileFingerprint profileFingerprint,
            List<Occurrence<C, D>> occurrences,
            Objective objective,
            Budget budget,
            Path workloadCache) {
        this.modelFingerprint = Objects.requireNonNull(modelFingerprint, "modelFingerprint");
        this.profileFingerprint = Objects.requireNonNull(profileFingerprint, "profileFingerprint");
        this.occurrences = List.copyOf(Objects.requireNonNull(occurrences, "occurrences"));
        if (this.occurrences.isEmpty()) {
            throw new IllegalArgumentException("occurrences must not be empty");
        }
        this.objective = Objects.requireNonNull(objective, "objective");
        if (objective != Objective.MIN_MEDIAN_ELAPSED_NANOS) {
            throw new IllegalArgumentException("unsupported objective");
        }
        this.budget = Objects.requireNonNull(budget, "budget");
        this.workloadCache = Objects.requireNonNull(workloadCache, "workloadCache");
    }

    /**
     * Returns the caller-defined identity attached to result evidence.
     *
     * @return the non-null immutable model fingerprint
     */
    public ModelFingerprint modelFingerprint() { return modelFingerprint; }
    /**
     * Returns the caller-defined representative-profile identity attached to result evidence.
     *
     * @return the non-null immutable representative-profile fingerprint
     */
    public ProfileFingerprint profileFingerprint() { return profileFingerprint; }
    /**
     * Returns the original work occurrences.
     *
     * @return the immutable non-empty ordered occurrence snapshot
     */
    public List<Occurrence<C, D>> occurrences() { return occurrences; }
    /**
     * Returns the selection objective.
     *
     * @return the non-null exact initial objective
     */
    public Objective objective() { return objective; }
    /**
     * Returns the complete-work and sampling bounds.
     *
     * @return the non-null immutable sampling budget
     */
    public Budget budget() { return budget; }
    /**
     * Returns the cache file used by this transaction.
     *
     * @return the non-null explicit workload-cache file path retained at construction
     */
    public Path workloadCache() { return workloadCache; }

    private abstract static class OpaqueFingerprint extends OpaqueBytes {
        private final int schemaVersion;

        OpaqueFingerprint(int schemaVersion, byte[] bytes, String name) {
            super(bytes, name);
            if (schemaVersion <= 0) {
                throw new IllegalArgumentException(name + " schema version must be positive");
            }
            this.schemaVersion = schemaVersion;
        }

        /**
         * Returns the owner-defined schema for interpreting the opaque bytes.
         *
         * @return the positive schema version
         */
        public final int schemaVersion() { return schemaVersion; }

        final boolean sameFingerprint(OpaqueFingerprint other) {
            return schemaVersion == other.schemaVersion && sameBytes(other);
        }

        final int fingerprintHashCode() {
            return 31 * schemaVersion + bytesHashCode();
        }

        @Override
        public boolean equals(java.lang.Object other) {
            return this == other || other != null && getClass() == other.getClass()
                    && sameFingerprint((OpaqueFingerprint) other);
        }

        @Override
        public int hashCode() {
            return fingerprintHashCode();
        }
    }

    private abstract static class OpaqueBytes {
        private final byte[] bytes;

        OpaqueBytes(byte[] bytes, String name) {
            this.bytes = snapshot(bytes, name);
        }

        /**
         * Returns the opaque identity bytes without exposing internal mutable state.
         *
         * @return a fresh caller-owned copy of the non-empty byte sequence
         */
        public final byte[] bytes() { return bytes.clone(); }
        final int byteLength() { return bytes.length; }
        final boolean sameBytes(OpaqueBytes other) { return Arrays.equals(bytes, other.bytes); }
        final int bytesHashCode() { return Arrays.hashCode(bytes); }

        @Override
        public boolean equals(java.lang.Object other) {
            return this == other || other != null && getClass() == other.getClass()
                    && sameBytes((OpaqueBytes) other);
        }

        @Override
        public int hashCode() { return bytesHashCode(); }
    }

    private static byte[] snapshot(byte[] bytes, String name) {
        Objects.requireNonNull(bytes, name);
        if (bytes.length == 0) {
            throw new IllegalArgumentException(name + " must not be empty");
        }
        return bytes.clone();
    }
}
