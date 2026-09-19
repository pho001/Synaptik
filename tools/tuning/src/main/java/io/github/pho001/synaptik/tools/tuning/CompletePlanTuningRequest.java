package io.github.pho001.synaptik.tools.tuning;

import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable identities, policies, bounds, cache location, and opaque Prepare handoff for one
 * complete-plan tuning transaction.
 *
 * <p>Construction performs no filesystem access. Opaque byte identities are snapshotted and
 * bounded to 1 MiB each; their accessors return fresh arrays. The explicit cache path and exact
 * decision-empty handoff are retained by reference. The producer later declares whether the
 * transaction is session-only or eligible for persistent reuse.
 *
 * @param <C> immutable complete candidate-batch type transported by the handoff
 * @param <D> immutable selected-decision type transported by the handoff
 */
public final class CompletePlanTuningRequest<
        C extends BackendTuningCandidateBatch,
        D extends BackendTuningDecision> {
    /** Supported deterministic timing objective. */
    public enum Objective {
        /** Select the first candidate having the smallest integer-middle elapsed-time median. */
        MIN_MEDIAN_ELAPSED_NANOS
    }

    /** Supported correctness policy, executed before any timing. */
    public enum CorrectnessPolicy {
        /** Require exact equality of caller-produced canonical publication bytes. */
        EXACT_CANONICAL_BYTES
    }

    /** Backend-declared lifetime within which a selected decision may be reused. */
    public enum ReuseScope {
        /** Reuse is limited to the live session; the tuning tool performs no cache I/O. */
        SESSION,
        /** Reuse may cross sessions through the explicit authenticated model-plan cache. */
        PERSISTENT
    }

    private abstract static class OpaqueFingerprint {
        private final int schemaVersion;
        private final byte[] bytes;

        /**
         * Creates a versioned opaque identity.
         *
         * @param schemaVersion positive caller-owned schema version
         * @param bytes non-null, non-empty opaque identity bytes of at most 1 MiB; snapshotted
         * @throws NullPointerException if {@code bytes} is null
         * @throws IllegalArgumentException if the schema is not positive or the byte length is
         *     zero or greater than 1 MiB
         */
        OpaqueFingerprint(int schemaVersion, byte[] bytes) {
            if (schemaVersion <= 0) {
                throw new IllegalArgumentException("schemaVersion must be positive");
            }
            this.schemaVersion = schemaVersion;
            this.bytes = snapshot(bytes, "fingerprint");
        }

        /**
         * Returns the identity schema version.
         *
         * @return the positive schema version
         */
        public final int schemaVersion() {
            return schemaVersion;
        }

        /**
         * Returns the opaque identity payload.
         *
         * @return a fresh copy of the non-empty opaque bytes
         */
        public final byte[] bytes() {
            return bytes.clone();
        }

        @Override
        public final boolean equals(java.lang.Object other) {
            return this == other || other != null && getClass() == other.getClass()
                    && schemaVersion == ((OpaqueFingerprint) other).schemaVersion
                    && Arrays.equals(bytes, ((OpaqueFingerprint) other).bytes);
        }

        @Override
        public final int hashCode() {
            return 31 * schemaVersion + Arrays.hashCode(bytes);
        }
    }

    /** Versioned opaque identity for the model whose complete plans are being compared. */
    public static final class ModelFingerprint extends OpaqueFingerprint {
        /**
         * Creates a model fingerprint.
         *
         * @param schemaVersion positive model-fingerprint schema
         * @param bytes non-empty model-fingerprint bytes; snapshotted
         * @throws NullPointerException if {@code bytes} is null
         * @throws IllegalArgumentException if the schema is not positive or the byte length is
         *     zero or greater than 1 MiB
         */
        public ModelFingerprint(int schemaVersion, byte[] bytes) {
            super(schemaVersion, bytes);
        }
    }
    /** Versioned opaque identity for the representative input profile. */
    public static final class ProfileFingerprint extends OpaqueFingerprint {
        /**
         * Creates a representative-profile fingerprint.
         *
         * @param schemaVersion positive profile-fingerprint schema
         * @param bytes non-empty profile-fingerprint bytes; snapshotted
         * @throws NullPointerException if {@code bytes} is null
         * @throws IllegalArgumentException if the schema is not positive or the byte length is
         *     zero or greater than 1 MiB
         */
        public ProfileFingerprint(int schemaVersion, byte[] bytes) {
            super(schemaVersion, bytes);
        }
    }
    /** Versioned opaque identity for the execution target and relevant environment. */
    public static final class TargetFingerprint extends OpaqueFingerprint {
        /**
         * Creates a target fingerprint.
         *
         * @param schemaVersion positive target-fingerprint schema
         * @param bytes non-empty target-fingerprint bytes; snapshotted
         * @throws NullPointerException if {@code bytes} is null
         * @throws IllegalArgumentException if the schema is not positive or the byte length is
         *     zero or greater than 1 MiB
         */
        public TargetFingerprint(int schemaVersion, byte[] bytes) {
            super(schemaVersion, bytes);
        }
    }
    /** Versioned opaque identity for candidate constraints and tuning policy. */
    public static final class PolicyIdentity extends OpaqueFingerprint {
        /**
         * Creates a policy identity.
         *
         * @param schemaVersion positive policy schema
         * @param bytes non-empty policy bytes; snapshotted
         * @throws NullPointerException if {@code bytes} is null
         * @throws IllegalArgumentException if the schema is not positive or the byte length is
         *     zero or greater than 1 MiB
         */
        public PolicyIdentity(int schemaVersion, byte[] bytes) {
            super(schemaVersion, bytes);
        }
    }
    /** Versioned opaque identity for the candidate/decision producer implementation. */
    public static final class ProducerIdentity extends OpaqueFingerprint {
        /**
         * Creates a producer identity.
         *
         * @param schemaVersion positive producer schema
         * @param bytes non-empty producer bytes; snapshotted
         * @throws NullPointerException if {@code bytes} is null
         * @throws IllegalArgumentException if the schema is not positive or the byte length is
         *     zero or greater than 1 MiB
         */
        public ProducerIdentity(int schemaVersion, byte[] bytes) {
            super(schemaVersion, bytes);
        }
    }
    /** Versioned opaque identity for the selected-decision codec. */
    public static final class DecisionCodecIdentity extends OpaqueFingerprint {
        /**
         * Creates a decision-codec identity.
         *
         * @param schemaVersion positive codec schema
         * @param bytes non-empty codec bytes; snapshotted
         * @throws NullPointerException if {@code bytes} is null
         * @throws IllegalArgumentException if the schema is not positive or the byte length is
         *     zero or greater than 1 MiB
         */
        public DecisionCodecIdentity(int schemaVersion, byte[] bytes) {
            super(schemaVersion, bytes);
        }
    }
    /** Opaque stable identity for one complete candidate within the supplied batch. */
    public static final class CandidateIdentity extends OpaqueFingerprint {
        /**
         * Creates a schema-1 candidate identity by snapshotting its bytes.
         *
         * @param bytes non-null, non-empty opaque candidate identity bytes of at most 1 MiB
         * @throws NullPointerException if {@code bytes} is null
         * @throws IllegalArgumentException if the byte length is zero or greater than 1 MiB
         */
        public CandidateIdentity(byte[] bytes) {
            super(1, bytes);
        }
    }

    /**
     * Opaque backend compatibility plus producer, codec, and reuse-scope identities needed to
     * authenticate selected-decision reuse.
     *
     * <p>Equality includes all identity fields, compatibility bytes, and reuse scope. A session
     * value is still meaningful in the in-memory selected-plan record but is never read from or
     * written to the model-plan cache.
     */
    public static final class PlanCompatibility {
        private final ProducerIdentity producerIdentity;
        private final DecisionCodecIdentity decisionCodecIdentity;
        private final int schemaVersion;
        private final byte[] bytes;
        private final ReuseScope reuseScope;

        /**
         * Creates an opaque compatibility value.
         * @param producerIdentity non-null versioned producer identity
         * @param decisionCodecIdentity non-null versioned codec identity
         * @param schemaVersion positive compatibility schema
         * @param bytes non-empty compatibility bytes, snapshotted
         * @param reuseScope non-null session or persistent reuse scope
         * @throws NullPointerException if an identity, byte array, or reuse scope is null
         * @throws IllegalArgumentException if the schema is not positive or the byte length is
         *     zero or greater than 1 MiB
         */
        public PlanCompatibility(ProducerIdentity producerIdentity,
                DecisionCodecIdentity decisionCodecIdentity, int schemaVersion, byte[] bytes,
                ReuseScope reuseScope) {
            this.producerIdentity = Objects.requireNonNull(producerIdentity, "producerIdentity");
            this.decisionCodecIdentity =
                    Objects.requireNonNull(decisionCodecIdentity, "decisionCodecIdentity");
            if (schemaVersion <= 0) {
                throw new IllegalArgumentException("schemaVersion must be positive");
            }
            this.schemaVersion = schemaVersion;
            this.bytes = snapshot(bytes, "compatibility");
            this.reuseScope = Objects.requireNonNull(reuseScope, "reuseScope");
        }
        /**
         * Returns the producer identity.
         *
         * @return the non-null immutable producer identity retained by reference
         */
        public ProducerIdentity producerIdentity() {
            return producerIdentity;
        }
        /**
         * Returns the decision-codec identity.
         *
         * @return the non-null immutable decision-codec identity retained by reference
         */
        public DecisionCodecIdentity decisionCodecIdentity() {
            return decisionCodecIdentity;
        }
        /**
         * Returns the compatibility schema version.
         *
         * @return the positive compatibility schema
         */
        public int schemaVersion() {
            return schemaVersion;
        }
        /**
         * Returns the opaque compatibility payload.
         *
         * @return a fresh copy of the non-empty, bounded compatibility bytes
         */
        public byte[] bytes() {
            return bytes.clone();
        }
        /**
         * Returns the producer-declared reuse lifetime.
         *
         * @return the non-null backend-declared reuse scope
         */
        public ReuseScope reuseScope() {
            return reuseScope;
        }

        @Override
        public boolean equals(java.lang.Object other) {
            return this == other || other instanceof PlanCompatibility that
                    && producerIdentity.equals(that.producerIdentity)
                    && decisionCodecIdentity.equals(that.decisionCodecIdentity)
                    && schemaVersion == that.schemaVersion && Arrays.equals(bytes, that.bytes)
                    && reuseScope == that.reuseScope;
        }
        @Override
        public int hashCode() {
            int result =
                    Objects.hash(
                            producerIdentity,
                            decisionCodecIdentity,
                            schemaVersion,
                            reuseScope);
            return 31 * result + Arrays.hashCode(bytes);
        }
    }

    private final ModelFingerprint modelFingerprint;
    private final ProfileFingerprint profileFingerprint;
    private final TargetFingerprint targetFingerprint;
    private final Objective objective;
    private final CorrectnessPolicy correctnessPolicy;
    private final PolicyIdentity policyIdentity;
    private final int maximumPlanCandidates;
    private final int warmupCount;
    private final int timedSampleCount;
    private final long maximumTotalPlanExecutions;
    private final long maximumAggregateCorrectnessBytes;
    private final Path modelPlanCache;
    private final BackendPartitionTuningHandoff<C, D> handoff;

    /**
     * Creates one immutable request without performing filesystem access.
     *
     * <p>The total execution ceiling includes one correctness action, every warmup, and every
     * timed sample for every candidate. The tuning transaction later verifies the checked formula
     * {@code N * (1 + warmupCount + timedSampleCount)} against this ceiling before the first
     * correctness callback.
     * @param modelFingerprint non-null versioned model identity
     * @param profileFingerprint non-null versioned representative-profile identity
     * @param targetFingerprint non-null versioned target identity
     * @param objective non-null minimum-median objective
     * @param correctnessPolicy non-null exact-byte correctness policy
     * @param policyIdentity non-null versioned constraint/policy identity
     * @param maximumPlanCandidates positive complete-candidate ceiling
     * @param warmupCount non-negative warmups per candidate
     * @param timedSampleCount positive odd timed samples per candidate
     * @param maximumTotalPlanExecutions positive ceiling including correctness, warmup, and samples
     * @param maximumAggregateCorrectnessBytes non-negative caller-enforced byte ceiling
     * @param modelPlanCache non-null explicit cache path; no access occurs during construction
     * @param handoff non-null exact handoff whose decision must be absent
     * @throws NullPointerException if an identity, policy, path, or handoff is null
     * @throws IllegalArgumentException if a numeric bound is outside its documented range, the
     *     cache path has an empty string form, or the input handoff already contains a decision
     */
    public CompletePlanTuningRequest(ModelFingerprint modelFingerprint,
            ProfileFingerprint profileFingerprint, TargetFingerprint targetFingerprint,
            Objective objective, CorrectnessPolicy correctnessPolicy, PolicyIdentity policyIdentity,
            int maximumPlanCandidates, int warmupCount, int timedSampleCount,
            long maximumTotalPlanExecutions, long maximumAggregateCorrectnessBytes,
            Path modelPlanCache, BackendPartitionTuningHandoff<C, D> handoff) {
        this.modelFingerprint = Objects.requireNonNull(modelFingerprint, "modelFingerprint");
        this.profileFingerprint = Objects.requireNonNull(profileFingerprint, "profileFingerprint");
        this.targetFingerprint = Objects.requireNonNull(targetFingerprint, "targetFingerprint");
        this.objective = Objects.requireNonNull(objective, "objective");
        this.correctnessPolicy = Objects.requireNonNull(correctnessPolicy, "correctnessPolicy");
        this.policyIdentity = Objects.requireNonNull(policyIdentity, "policyIdentity");
        if (maximumPlanCandidates <= 0) {
            throw new IllegalArgumentException("maximumPlanCandidates must be positive");
        }
        if (warmupCount < 0) {
            throw new IllegalArgumentException("warmupCount must be non-negative");
        }
        if (timedSampleCount <= 0 || (timedSampleCount & 1) == 0) {
            throw new IllegalArgumentException("timedSampleCount must be positive and odd");
        }
        if (maximumTotalPlanExecutions <= 0) {
            throw new IllegalArgumentException("maximumTotalPlanExecutions must be positive");
        }
        if (maximumAggregateCorrectnessBytes < 0) {
            throw new IllegalArgumentException(
                    "maximumAggregateCorrectnessBytes must be non-negative");
        }
        this.maximumPlanCandidates = maximumPlanCandidates;
        this.warmupCount = warmupCount;
        this.timedSampleCount = timedSampleCount;
        this.maximumTotalPlanExecutions = maximumTotalPlanExecutions;
        this.maximumAggregateCorrectnessBytes = maximumAggregateCorrectnessBytes;
        this.modelPlanCache = Objects.requireNonNull(modelPlanCache, "modelPlanCache");
        if (modelPlanCache.toString().isEmpty()) {
            throw new IllegalArgumentException("modelPlanCache must be an explicit path");
        }
        this.handoff = Objects.requireNonNull(handoff, "handoff");
        if (handoff.selectedDecision().isPresent()) {
            throw new IllegalArgumentException("input handoff decision must be absent");
        }
    }

    /**
     * Returns the model identity.
     *
     * @return the non-null immutable model fingerprint retained by reference
     */
    public ModelFingerprint modelFingerprint() {
        return modelFingerprint;
    }

    /**
     * Returns the representative-profile identity.
     *
     * @return the non-null immutable representative-profile fingerprint retained by reference
     */
    public ProfileFingerprint profileFingerprint() {
        return profileFingerprint;
    }

    /**
     * Returns the execution-target identity.
     *
     * @return the non-null immutable target fingerprint retained by reference
     */
    public TargetFingerprint targetFingerprint() {
        return targetFingerprint;
    }

    /**
     * Returns the selection objective.
     *
     * @return the non-null exact supported objective
     */
    public Objective objective() {
        return objective;
    }

    /**
     * Returns the correctness policy.
     *
     * @return the non-null exact supported correctness policy
     */
    public CorrectnessPolicy correctnessPolicy() {
        return correctnessPolicy;
    }

    /**
     * Returns the constraint/policy identity.
     *
     * @return the non-null immutable constraint/policy identity retained by reference
     */
    public PolicyIdentity policyIdentity() {
        return policyIdentity;
    }

    /**
     * Returns the candidate-count bound.
     *
     * @return positive complete-candidate ceiling
     */
    public int maximumPlanCandidates() {
        return maximumPlanCandidates;
    }

    /**
     * Returns the configured warmup count.
     *
     * @return non-negative warmups per candidate
     */
    public int warmupCount() {
        return warmupCount;
    }

    /**
     * Returns the configured timed-sample count.
     *
     * @return positive odd timed samples per candidate
     */
    public int timedSampleCount() {
        return timedSampleCount;
    }

    /**
     * Returns the whole-transaction execution bound.
     *
     * @return the positive whole-transaction execution ceiling
     */
    public long maximumTotalPlanExecutions() {
        return maximumTotalPlanExecutions;
    }

    /**
     * Returns the caller-enforced correctness payload bound.
     *
     * @return the non-negative aggregate correctness-payload ceiling in bytes
     */
    public long maximumAggregateCorrectnessBytes() {
        return maximumAggregateCorrectnessBytes;
    }

    /**
     * Returns the explicit model-plan-cache location.
     *
     * @return the non-null explicit model-plan-cache path retained by reference; a session-scoped
     *     transaction does not access it
     */
    public Path modelPlanCache() {
        return modelPlanCache;
    }

    /**
     * Returns the input handoff.
     *
     * @return the non-null exact decision-empty input handoff retained by reference
     */
    public BackendPartitionTuningHandoff<C, D> handoff() {
        return handoff;
    }

    private static byte[] snapshot(byte[] bytes, String name) {
        Objects.requireNonNull(bytes, name);
        if (bytes.length == 0 || bytes.length > ModelPlanCacheFile.MAX_OPAQUE_BYTES) {
            throw new IllegalArgumentException(name + " length is invalid");
        }
        return bytes.clone();
    }
}
