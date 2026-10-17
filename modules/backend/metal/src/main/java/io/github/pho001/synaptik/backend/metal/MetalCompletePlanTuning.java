package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.zip.CRC32;

/**
 * Supported Metal-owned collaboration for fixed complete-plan tuning.
 *
 * <p>Phase one has already selected the exact local singleton-NEG route. Consequently this
 * collaboration exposes exactly one authenticated complete-plan candidate: the complete graph
 * prepared with that fixed route. The candidate still participates in correctness capture,
 * measurement, evidence, and final preparation, but it does not invent a second execution plan.
 * Compatibility and decision reuse are limited to the current live integration session.</p>
 *
 * <p>This collaboration is retained by the same {@link MetalBackendIntegration} as its
 * {@link MetalLocalWorkloadTuning}. It performs no execution, measurement, cache input/output,
 * winner selection, or fallback-policy work.</p>
 */
public final class MetalCompletePlanTuning {
    private static final int VALUE_SCHEMA_VERSION = 1;
    private static final int COMPATIBILITY_MAGIC = 0x4d435031;
    private static final int CANDIDATE_MAGIC = 0x4d434331;
    private static final int DECISION_MAGIC = 0x4d434431;
    private static final int FIXED_PLAN_CANDIDATE = 1;
    private static final int MAX_DECISION_BYTES = 4096;

    private final MetalLocalWorkloadTuning local;

    MetalCompletePlanTuning(MetalLocalWorkloadTuning local) {
        this.local = Objects.requireNonNull(local, "local");
    }

    /**
     * Produces the one authenticated fixed-route complete-plan candidate.
     *
     * @param projectedContext non-null exact stable Metal projection
     * @param phaseOneDecision non-null selected decision from the exact local batch and projection
     * @return non-empty decision-empty handoff containing exactly one candidate
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the decision is foreign, stale, incompatible, or not
     *     associated with the exact projection
     * @throws IllegalStateException if the owning integration is closed
     */
    public BackendPartitionTuningHandoff<CandidateBatch, SelectedDecision> candidateHandoff(
            PrepareContext<?> projectedContext,
            MetalLocalWorkloadTuning.SelectedDecision phaseOneDecision) {
        Objects.requireNonNull(projectedContext, "projectedContext");
        local.authenticateCompletePlanPhaseOne(projectedContext, phaseOneDecision);
        byte[] phaseOneBytes = local.encodeDecision(phaseOneDecision);
        byte[] localCandidateIdentity = local.completePlanCandidateIdentity(phaseOneDecision);
        Association association = new Association(
                this,
                projectedContext,
                projectedContext.partition(),
                phaseOneDecision,
                encodeCompatibility(phaseOneBytes),
                encodeCandidate(localCandidateIdentity));
        Candidate candidate = new Candidate(association);
        CandidateBatch batch = new CandidateBatch(association, List.of(candidate));
        return new BackendPartitionTuningHandoff<>(
                association.partition, batch, Optional.empty());
    }

    /**
     * Returns the sole authenticated fixed-route complete-plan candidate.
     *
     * @param batch non-null batch issued by this collaboration
     * @return retained immutable singleton candidate list
     * @throws NullPointerException if {@code batch} is {@code null}
     * @throws IllegalArgumentException if the batch belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public List<Candidate> candidates(CandidateBatch batch) {
        return requireBatch(batch).candidates;
    }

    /**
     * Returns defensive canonical session compatibility for one complete-plan batch.
     *
     * @param batch non-null batch issued by this collaboration
     * @return immutable session-scoped compatibility
     * @throws NullPointerException if {@code batch} is {@code null}
     * @throws IllegalArgumentException if the batch belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public Compatibility compatibility(CandidateBatch batch) {
        Association association = requireBatch(batch).association;
        return new Compatibility(
                VALUE_SCHEMA_VERSION, ReuseScope.SESSION, association.compatibilityBytes);
    }

    /**
     * Returns defensive canonical identity for the sole fixed-plan candidate.
     *
     * @param candidate non-null candidate issued by this collaboration
     * @return immutable opaque identity
     * @throws NullPointerException if {@code candidate} is {@code null}
     * @throws IllegalArgumentException if the candidate belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public CandidateIdentity candidateIdentity(Candidate candidate) {
        return new CandidateIdentity(requireCandidate(candidate).candidateBytes);
    }

    /**
     * Constructs the selected decision for the sole candidate.
     *
     * @param batch non-null originating batch
     * @param candidate non-null exact member of {@code batch}
     * @return immutable exact-batch decision
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if either value has another owner or batch association
     * @throws IllegalStateException if the owning integration is closed
     */
    public SelectedDecision selectedDecision(CandidateBatch batch, Candidate candidate) {
        Association association = requireBatch(batch).association;
        Association candidateAssociation = requireCandidate(candidate);
        if (association != candidateAssociation) {
            throw new IllegalArgumentException("candidate does not belong to batch");
        }
        return new SelectedDecision(association);
    }

    /**
     * Encodes one current complete-plan decision with bounds and checksum protection.
     *
     * @param decision non-null decision issued by this collaboration
     * @return fresh canonical caller-owned bytes
     * @throws NullPointerException if {@code decision} is {@code null}
     * @throws IllegalArgumentException if it belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public byte[] encodeDecision(SelectedDecision decision) {
        Association association = requireDecision(decision);
        int bodyLength = Math.addExact(
                Integer.BYTES * 5,
                Math.addExact(association.compatibilityBytes.length,
                        association.candidateBytes.length));
        ByteBuffer buffer = ByteBuffer.allocate(Math.addExact(bodyLength, Long.BYTES));
        buffer.putInt(DECISION_MAGIC);
        buffer.putInt(VALUE_SCHEMA_VERSION);
        buffer.putInt(ReuseScope.SESSION.ordinal());
        buffer.putInt(association.compatibilityBytes.length);
        buffer.put(association.compatibilityBytes);
        buffer.putInt(association.candidateBytes.length);
        buffer.put(association.candidateBytes);
        CRC32 checksum = new CRC32();
        checksum.update(buffer.array(), 0, bodyLength);
        buffer.putLong(checksum.getValue());
        return buffer.array();
    }

    /**
     * Decodes only a decision compatible with the exact supplied current batch.
     *
     * @param batch non-null current batch issued by this collaboration
     * @param encodedDecision non-null caller-owned bytes, snapshotted before parsing
     * @return exact-batch decision, or empty for malformed, corrupt, stale, wrong-session, or
     *     unknown input
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if {@code batch} belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public Optional<SelectedDecision> decodeCompatibleDecision(
            CandidateBatch batch, byte[] encodedDecision) {
        Association association = requireBatch(batch).association;
        byte[] input = Objects.requireNonNull(encodedDecision, "encodedDecision").clone();
        if (input.length < Integer.BYTES * 5 + Long.BYTES
                || input.length > MAX_DECISION_BYTES) {
            return Optional.empty();
        }
        try {
            ByteBuffer buffer = ByteBuffer.wrap(input);
            if (buffer.getInt() != DECISION_MAGIC
                    || buffer.getInt() != VALUE_SCHEMA_VERSION
                    || buffer.getInt() != ReuseScope.SESSION.ordinal()) {
                return Optional.empty();
            }
            int compatibilityLength = buffer.getInt();
            if (compatibilityLength < 0
                    || compatibilityLength > buffer.remaining() - Integer.BYTES - Long.BYTES) {
                return Optional.empty();
            }
            byte[] compatibility = new byte[compatibilityLength];
            buffer.get(compatibility);
            int candidateLength = buffer.getInt();
            if (candidateLength < 0 || candidateLength != buffer.remaining() - Long.BYTES) {
                return Optional.empty();
            }
            byte[] candidate = new byte[candidateLength];
            buffer.get(candidate);
            long encodedChecksum = buffer.getLong();
            CRC32 checksum = new CRC32();
            checksum.update(input, 0, input.length - Long.BYTES);
            if (encodedChecksum != checksum.getValue()
                    || !Arrays.equals(association.compatibilityBytes, compatibility)
                    || !Arrays.equals(association.candidateBytes, candidate)) {
                return Optional.empty();
            }
            return Optional.of(new SelectedDecision(association));
        } catch (RuntimeException malformed) {
            return Optional.empty();
        }
    }

    /**
     * Freshly prepares the sole complete-plan candidate without executing or measuring it.
     *
     * @param batch non-null originating batch
     * @param candidate non-null exact member of {@code batch}
     * @return Metal-owned partition preparation for the authenticated fixed route
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if exact association or fresh authentication fails
     * @throws IllegalStateException if the owning integration is closed
     */
    public PartitionPreparation<?, ?> trialPreparation(
            CandidateBatch batch, Candidate candidate) {
        return selectedPreparation(batch, selectedDecision(batch, candidate));
    }

    /**
     * Freshly prepares an authenticated complete-plan decision without heuristic fallback.
     *
     * @param batch non-null originating batch
     * @param decision non-null decision associated with exactly {@code batch}
     * @return Metal-owned partition preparation for the authenticated fixed route
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if exact association or fresh authentication fails
     * @throws IllegalStateException if the owning integration is closed
     */
    public PartitionPreparation<?, ?> selectedPreparation(
            CandidateBatch batch, SelectedDecision decision) {
        Association association = requireBatch(batch).association;
        if (requireDecision(decision) != association) {
            throw new IllegalArgumentException("decision does not belong to batch");
        }
        return local.completePlanSelectedPreparation(
                association.context, association.phaseOneDecision);
    }

    private CandidateBatch requireBatch(CandidateBatch batch) {
        Objects.requireNonNull(batch, "batch");
        if (batch.association.owner != this) {
            throw new IllegalArgumentException(
                    "batch belongs to another Metal complete-plan collaboration");
        }
        local.requireLive();
        return batch;
    }

    private Association requireCandidate(Candidate candidate) {
        Objects.requireNonNull(candidate, "candidate");
        if (candidate.association.owner != this) {
            throw new IllegalArgumentException(
                    "candidate belongs to another Metal complete-plan collaboration");
        }
        local.requireLive();
        return candidate.association;
    }

    private Association requireDecision(SelectedDecision decision) {
        Objects.requireNonNull(decision, "decision");
        if (decision.association.owner != this) {
            throw new IllegalArgumentException(
                    "decision belongs to another Metal complete-plan collaboration");
        }
        local.requireLive();
        return decision.association;
    }

    private static byte[] encodeCompatibility(byte[] phaseOneDecision) {
        ByteBuffer buffer = ByteBuffer.allocate(
                Integer.BYTES * 4 + phaseOneDecision.length);
        buffer.putInt(COMPATIBILITY_MAGIC);
        buffer.putInt(VALUE_SCHEMA_VERSION);
        buffer.putInt(ReuseScope.SESSION.ordinal());
        buffer.putInt(phaseOneDecision.length);
        buffer.put(phaseOneDecision);
        return buffer.array();
    }

    private static byte[] encodeCandidate(byte[] localCandidateIdentity) {
        ByteBuffer buffer = ByteBuffer.allocate(
                Integer.BYTES * 4 + localCandidateIdentity.length);
        buffer.putInt(CANDIDATE_MAGIC);
        buffer.putInt(VALUE_SCHEMA_VERSION);
        buffer.putInt(FIXED_PLAN_CANDIDATE);
        buffer.putInt(localCandidateIdentity.length);
        buffer.put(localCandidateIdentity);
        return buffer.array();
    }

    /** Scope within which complete-plan compatibility and decisions may be reused. */
    public enum ReuseScope {
        /** Reuse is limited to the exact live {@link MetalBackendIntegration}. */
        SESSION
    }

    /** Opaque exact-association complete-plan candidate batch. */
    public static final class CandidateBatch implements BackendTuningCandidateBatch {
        private final Association association;
        private final List<Candidate> candidates;

        private CandidateBatch(Association association, List<Candidate> candidates) {
            this.association = association;
            this.candidates = candidates;
        }

        @Override
        public boolean equals(Object other) {
            return this == other || other instanceof CandidateBatch value
                    && association == value.association;
        }

        @Override
        public int hashCode() { return System.identityHashCode(association); }

        @Override
        public String toString() { return "MetalCompletePlanCandidateBatch[opaque]"; }
    }

    /** Opaque exact-association selected complete-plan decision. */
    public static final class SelectedDecision implements BackendTuningDecision {
        private final Association association;

        private SelectedDecision(Association association) { this.association = association; }

        @Override
        public boolean equals(Object other) {
            return this == other || other instanceof SelectedDecision value
                    && association == value.association;
        }

        @Override
        public int hashCode() { return System.identityHashCode(association); }

        @Override
        public String toString() { return "MetalCompletePlanSelectedDecision[opaque]"; }
    }

    /** Opaque sole authenticated fixed-route complete-plan candidate. */
    public static final class Candidate {
        private final Association association;

        private Candidate(Association association) { this.association = association; }

        @Override
        public boolean equals(Object other) {
            return this == other || other instanceof Candidate value
                    && association == value.association;
        }

        @Override
        public int hashCode() { return System.identityHashCode(association); }

        @Override
        public String toString() { return "MetalCompletePlanCandidate[opaque]"; }
    }

    /** Immutable canonical complete-plan compatibility bytes and reuse scope. */
    public static final class Compatibility {
        private final int schemaVersion;
        private final ReuseScope reuseScope;
        private final byte[] encoded;

        private Compatibility(int schemaVersion, ReuseScope reuseScope, byte[] encoded) {
            this.schemaVersion = schemaVersion;
            this.reuseScope = reuseScope;
            this.encoded = encoded.clone();
        }

        /**
         * Returns the compatibility schema.
         * @return exact positive complete-plan compatibility schema version
         */
        public int schemaVersion() { return schemaVersion; }

        /**
         * Copies the compatibility payload.
         * @return fresh caller-owned canonical compatibility bytes
         */
        public byte[] bytes() { return encoded.clone(); }

        /**
         * Returns the declared reuse scope.
         * @return exact session-only reuse scope
         */
        public ReuseScope reuseScope() { return reuseScope; }

        @Override
        public boolean equals(Object other) {
            return this == other || other instanceof Compatibility value
                    && schemaVersion == value.schemaVersion
                    && reuseScope == value.reuseScope
                    && Arrays.equals(encoded, value.encoded);
        }

        @Override
        public int hashCode() {
            return 31 * (31 * schemaVersion + reuseScope.hashCode())
                    + Arrays.hashCode(encoded);
        }

        @Override
        public String toString() {
            return "MetalCompletePlanCompatibility[schema=" + schemaVersion
                    + ", scope=" + reuseScope + ", bytes=" + encoded.length + ']';
        }
    }

    /** Immutable canonical complete-plan candidate identity bytes. */
    public static final class CandidateIdentity {
        private final byte[] encoded;

        private CandidateIdentity(byte[] encoded) { this.encoded = encoded.clone(); }

        /**
         * Copies the candidate identity.
         * @return fresh caller-owned canonical candidate identity bytes
         */
        public byte[] bytes() { return encoded.clone(); }

        @Override
        public boolean equals(Object other) {
            return this == other || other instanceof CandidateIdentity value
                    && Arrays.equals(encoded, value.encoded);
        }

        @Override
        public int hashCode() { return Arrays.hashCode(encoded); }

        @Override
        public String toString() {
            return "MetalCompletePlanCandidateIdentity[bytes=" + encoded.length + ']';
        }
    }

    private static final class Association {
        private final MetalCompletePlanTuning owner;
        private final PrepareContext<?> context;
        private final PlannedPartition partition;
        private final MetalLocalWorkloadTuning.SelectedDecision phaseOneDecision;
        private final byte[] compatibilityBytes;
        private final byte[] candidateBytes;

        private Association(
                MetalCompletePlanTuning owner,
                PrepareContext<?> context,
                PlannedPartition partition,
                MetalLocalWorkloadTuning.SelectedDecision phaseOneDecision,
                byte[] compatibilityBytes,
                byte[] candidateBytes) {
            this.owner = owner;
            this.context = context;
            this.partition = partition;
            this.phaseOneDecision = phaseOneDecision;
            this.compatibilityBytes = compatibilityBytes;
            this.candidateBytes = candidateBytes;
        }
    }
}
