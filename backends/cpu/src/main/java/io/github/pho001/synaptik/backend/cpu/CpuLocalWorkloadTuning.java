package io.github.pho001.synaptik.backend.cpu;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuBackendComposition;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasQualification;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasTuningBatch;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasTuningDecision;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.UUID;

/**
 * Supported CPU-owned collaboration for cold local-workload candidate preparation.
 *
 * <p>Instances are retained by one {@link CpuBackendIntegration}; callers acquire them through
 * {@link CpuBackendIntegration#localWorkloadTuning()} and never close them independently. The
 * collaboration exposes only opaque owner-associated values and defensive canonical bytes. It
 * performs no measurement, execution, representative-input binding, cache or filesystem access,
 * winner selection, or fallback-policy choice. Every trial and final preparation repeats CPU
 * analysis and rejects a stale or mismatched selection.</p>
 *
 * <p>Independent calls are safe while the owning integration remains open. Returned prepared
 * recipes borrow that integration's provider and coordination lifetime and must not outlive it.</p>
 */
public final class CpuLocalWorkloadTuning {
    private static final int VALUE_SCHEMA = 1;
    private static final int DECISION_MAGIC = 0x53435055;
    private static final int MAXIMUM_DECISION_BYTES = 512;

    private final CpuBackendComposition composition;
    private final byte[] sessionNonce;

    /**
     * Creates the retained collaboration for one CPU composition.
     *
     * @param composition non-null composition whose lifetime and resources are borrowed
     * @throws NullPointerException if {@code composition} is {@code null}
     * @hidden internal construction seam; supported callers use
     *     {@link CpuBackendIntegration#localWorkloadTuning()}
     */
    CpuLocalWorkloadTuning(CpuBackendComposition composition) {
        this.composition = Objects.requireNonNull(composition, "composition");
        UUID nonce = UUID.randomUUID();
        this.sessionNonce = ByteBuffer.allocate(16).putLong(nonce.getMostSignificantBits())
                .putLong(nonce.getLeastSignificantBits()).array();
    }

    /**
     * Produces the current complete CPU tuning handoff for one supported artifact, when eligible.
     *
     * @param artifacts non-null immutable artifacts containing exactly one non-empty CPU partition;
     *     retained by exact reference in a successful opaque batch and never mutated or closed
     * @return a non-null optional containing the exact partition, an associated opaque batch, and
     *     no selected decision; empty for a valid workload with no current tunable peer
     * @throws NullPointerException if {@code artifacts} is {@code null}
     * @throws IllegalArgumentException if the artifacts are zero-partition, empty-partition,
     *     non-CPU, mixed-owner, multi-partition, or otherwise invalid for CPU analysis
     * @throws IllegalStateException if the owning integration is closed
     */
    public Optional<BackendPartitionTuningHandoff<CandidateBatch, SelectedDecision>>
            candidateHandoff(CompileArtifacts artifacts) {
        requireOpen();
        Objects.requireNonNull(artifacts, "artifacts");
        return composition.tuningBatch(artifacts).map(internal -> {
            Association association = new Association(this, artifacts,
                    artifacts.partitions().getFirst(), internal);
            var candidates = new ArrayList<Candidate>(internal.candidates().size());
            for (int index = 0; index < internal.candidates().size(); index++) {
                candidates.add(new Candidate(association, index));
            }
            CandidateBatch batch = new CandidateBatch(association, List.copyOf(candidates));
            return new BackendPartitionTuningHandoff<>(association.partition, batch,
                    Optional.empty());
        });
    }

    /**
     * Returns every complete candidate in the authoritative portable-first CPU encounter order.
     *
     * @param batch non-null batch issued by this collaboration
     * @return the retained immutable non-empty candidate list; ownership is not transferred
     * @throws NullPointerException if {@code batch} is {@code null}
     * @throws IllegalArgumentException if it belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public List<Candidate> candidates(CandidateBatch batch) {
        requireBatch(batch);
        return batch.candidates;
    }

    /**
     * Projects the batch's exact versioned compatibility facts into defensive canonical bytes.
     *
     * @param batch non-null batch issued by this collaboration
     * @return a new immutable compatibility value; persistent only when the qualified binary has
     *     a stable persistent projection, otherwise bound to this integration session
     * @throws NullPointerException if {@code batch} is {@code null}
     * @throws IllegalArgumentException if it belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public Compatibility compatibility(CandidateBatch batch) {
        Association association = requireBatch(batch);
        return compatibility(association);
    }

    /**
     * Projects one exact candidate identity into defensive canonical bytes.
     *
     * @param candidate non-null candidate issued by this collaboration
     * @return a new non-null immutable identity value
     * @throws NullPointerException if {@code candidate} is {@code null}
     * @throws IllegalArgumentException if it belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public CandidateIdentity candidateIdentity(Candidate candidate) {
        Association association = requireCandidate(candidate);
        return identity(association.internal.candidates().get(candidate.index));
    }

    /**
     * Constructs an opaque decision for an exact candidate in an exact current batch.
     *
     * @param batch non-null batch issued by this collaboration
     * @param candidate non-null member of exactly {@code batch}
     * @return a non-null immutable decision associated with that exact batch
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if either value has another owner or association
     * @throws IllegalStateException if the owning integration is closed
     */
    public SelectedDecision selectedDecision(CandidateBatch batch, Candidate candidate) {
        Association association = requireBatch(batch);
        requireCandidate(candidate);
        if (candidate.association != association) {
            throw new IllegalArgumentException("candidate does not belong to batch");
        }
        return decision(association, candidate.index);
    }

    /**
     * Encodes a current associated decision using the bounded versioned CPU format.
     *
     * @param decision non-null decision issued by this collaboration
     * @return fresh caller-owned bytes containing only compatibility and candidate identity facts
     * @throws NullPointerException if {@code decision} is {@code null}
     * @throws IllegalArgumentException if it belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public byte[] encodeDecision(SelectedDecision decision) {
        requireDecision(decision);
        byte[] compatibility = compatibility(decision.association).encoded;
        byte[] identity = identity(decision.association.internal
                .find(decision.internal.selectedCandidate()).orElseThrow()).encoded;
        ByteBuffer output = ByteBuffer.allocate(17 + compatibility.length + identity.length);
        output.putInt(DECISION_MAGIC).putInt(VALUE_SCHEMA)
                .put((byte) reuseScope(decision.association).ordinal())
                .putInt(compatibility.length).put(compatibility)
                .putInt(identity.length).put(identity);
        return output.array();
    }

    /**
     * Decodes a decision only when every encoded fact matches the supplied current batch.
     *
     * @param batch non-null current batch issued by this collaboration
     * @param encodedDecision non-null caller-owned bytes, snapshotted before bounded parsing
     * @return a non-null optional containing a decision associated with {@code batch}, or empty for
     *     malformed, truncated, trailing, oversized, unsupported, stale, wrong-session, or unknown
     *     input
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if {@code batch} belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public Optional<SelectedDecision> decodeCompatibleDecision(CandidateBatch batch,
            byte[] encodedDecision) {
        Association association = requireBatch(batch);
        byte[] input = Objects.requireNonNull(encodedDecision, "encodedDecision").clone();
        if (input.length > MAXIMUM_DECISION_BYTES || input.length < 17) return Optional.empty();
        try {
            ByteBuffer bytes = ByteBuffer.wrap(input);
            if (bytes.getInt() != DECISION_MAGIC || bytes.getInt() != VALUE_SCHEMA
                    || Byte.toUnsignedInt(bytes.get()) != reuseScope(association).ordinal()) {
                return Optional.empty();
            }
            int compatibilityLength = bytes.getInt();
            if (compatibilityLength <= 0 || compatibilityLength > bytes.remaining() - 4) {
                return Optional.empty();
            }
            byte[] compatibility = new byte[compatibilityLength];
            bytes.get(compatibility);
            int identityLength = bytes.getInt();
            if (identityLength <= 0 || identityLength != bytes.remaining()) return Optional.empty();
            byte[] identity = new byte[identityLength];
            bytes.get(identity);
            if (!Arrays.equals(compatibility, compatibility(association).encoded)) {
                return Optional.empty();
            }
            for (int index = 0; index < association.internal.candidates().size(); index++) {
                if (Arrays.equals(identity,
                        identity(association.internal.candidates().get(index)).encoded)) {
                    return Optional.of(decision(association, index));
                }
            }
            return Optional.empty();
        } catch (RuntimeException malformed) {
            return Optional.empty();
        }
    }

    /**
     * Freshly prepares the exact requested candidate without running or measuring it.
     *
     * @param batch non-null originating batch
     * @param candidate non-null exact member of {@code batch}
     * @return one complete immutable reusable prepared execution
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if association, fresh eligibility, compatibility, or
     *     membership fails
     * @throws IllegalStateException if the owning integration is closed
     */
    public PreparedExecution prepareTrial(CandidateBatch batch, Candidate candidate) {
        return prepareSelected(batch, selectedDecision(batch, candidate));
    }

    /**
     * Freshly prepares the exact selected decision without heuristic fallback.
     *
     * @param batch non-null originating batch
     * @param decision non-null decision associated with exactly {@code batch}
     * @return one complete immutable reusable prepared execution
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if association, fresh eligibility, compatibility, or
     *     membership fails
     * @throws IllegalStateException if the owning integration is closed
     */
    public PreparedExecution prepareSelected(CandidateBatch batch, SelectedDecision decision) {
        Association association = requireBatch(batch);
        requireDecision(decision);
        if (decision.association != association) {
            throw new IllegalArgumentException("decision does not belong to batch");
        }
        return composition.prepareSelected(association.artifacts, decision.internal);
    }

    private void requireOpen() {
        composition.assertOpen();
    }

    private Association requireBatch(CandidateBatch batch) {
        requireOpen();
        Objects.requireNonNull(batch, "batch");
        if (batch.association.owner != this) {
            throw new IllegalArgumentException("batch belongs to another CPU integration");
        }
        return batch.association;
    }

    private Association requireCandidate(Candidate candidate) {
        requireOpen();
        Objects.requireNonNull(candidate, "candidate");
        if (candidate.association.owner != this) {
            throw new IllegalArgumentException("candidate belongs to another CPU integration");
        }
        return candidate.association;
    }

    private void requireDecision(SelectedDecision decision) {
        requireOpen();
        Objects.requireNonNull(decision, "decision");
        if (decision.association.owner != this) {
            throw new IllegalArgumentException("decision belongs to another CPU integration");
        }
    }

    private SelectedDecision decision(Association association, int index) {
        var candidate = association.internal.candidates().get(index);
        return new SelectedDecision(association, new CpuOpenBlasTuningDecision(
                association.internal.schemaVersion(), association.internal.workload(),
                candidate.identity()));
    }

    private Compatibility compatibility(Association association) {
        ReuseScope scope = reuseScope(association);
        CanonicalSink sink = new CanonicalSink();
        encodeWorkload(sink, association.internal.workload());
        byte[] digest = sink.digest();
        ByteBuffer bytes = ByteBuffer.allocate(5 + digest.length
                + (scope == ReuseScope.SESSION ? sessionNonce.length : 0));
        bytes.putInt(VALUE_SCHEMA).put((byte) scope.ordinal()).put(digest);
        if (scope == ReuseScope.SESSION) bytes.put(sessionNonce);
        return new Compatibility(VALUE_SCHEMA, scope, bytes.array());
    }

    private static ReuseScope reuseScope(Association association) {
        return association.internal.workload().persistentProjection().isPresent()
                ? ReuseScope.PERSISTENT : ReuseScope.SESSION;
    }

    private static CandidateIdentity identity(CpuOpenBlasTuningBatch.Candidate candidate) {
        CanonicalSink sink = new CanonicalSink();
        sink.tag(candidate.route());
        if (candidate.identity() instanceof CpuOpenBlasTuningBatch.PortableIdentity value) {
            sink.tag(value.strategy().compute()).tag(value.strategy().orchestration())
                    .integer(value.rangeCount())
                    .longValue(value.minimumElementsPerWorker())
                    .integer(value.vectorSpeciesBits()).integer(value.generatedArtifactSchema())
                    .text(value.structuralKey()).integer(value.resources().size());
            value.resources().forEach(resource -> sink.longValue(resource.byteSize())
                    .longValue(resource.byteAlignment()));
        } else if (candidate.identity() instanceof CpuOpenBlasTuningBatch.OpenBlasIdentity value) {
            sink.tag(value.representation()).integer(value.threadCount())
                    .integer(value.permitDemand()).integer(value.configuredThreadOrder())
                    .longs(value.workspaceRequirementIds()).longValue(value.workspaceBytes())
                    .longValue(value.inputCopiedElements()).longValue(value.outputCopiedElements());
        } else {
            throw new IllegalArgumentException("unsupported CPU tuning candidate identity");
        }
        byte[] digest = sink.digest();
        ByteBuffer bytes = ByteBuffer.allocate(9 + digest.length);
        bytes.putInt(VALUE_SCHEMA).put((byte) candidate.route().ordinal())
                .putInt(digest.length).put(digest);
        return new CandidateIdentity(bytes.array());
    }

    private static void encodeWorkload(CanonicalSink sink,
            CpuOpenBlasTuningBatch.WorkloadSignature value) {
        sink.tag(value.operationKind()).tag(value.operationAttributes()).tag(value.leftType())
                .tag(value.rightType()).tag(value.accumulationType()).tag(value.outputType())
                .integer(value.boundaries().size());
        value.boundaries().forEach(boundary -> {
            sink.tag(boundary.dataType());
            sink.integer(boundary.shape().rank());
            boundary.shape().dimensions().forEach(dimension ->
                    sink.longValue(dimension.staticSize().orElseThrow()));
            var layout = boundary.layout();
            sink.integer(layout.rank()).longs(layout.strides()).longValue(layout.storageOffset())
                    .tag(layout.kind()).bool(layout.isView())
                    .longValue(layout.referencedElementSpan()).tag(boundary.carrier())
                    .bool(boundary.storage().nativeSegment())
                    .longValue(boundary.storage().byteAlignment());
            encodeBinding(sink, boundary.access());
        });
        sink.tag(value.numericalMode()).tag(value.determinismMode());
        encodeQualification(sink, value.qualification());
        var hardware = value.hardware();
        sink.integer(hardware.schemaVersion()).text(hardware.architecture()).text(hardware.vendor())
                .text(hardware.model()).strings(hardware.features())
                .integer(value.cpuConcurrencyCapacity());
        var portable = value.portableExecution();
        sink.tag(portable.computePreference()).integer(portable.configuredMaximumParallelism())
                .integer(portable.availableParallelism())
                .longValue(portable.minimumElementsPerWorker())
                .tag(value.portableStrategy().compute())
                .tag(value.portableStrategy().orchestration())
                .integer(value.portableRangeCount()).integer(value.portableVectorSpeciesBits())
                .integers(value.openBlasThreadCounts());
        var cohort = value.cohort();
        sink.integer(cohort.schemaVersion()).text(cohort.cohortId())
                .longValue(cohort.expectedRunCount());
        var policy = value.materializationPolicy();
        sink.bool(policy.enabled()).longValue(policy.copyFixedCostUnits())
                .longValue(policy.copyCostUnitsPerElement())
                .longValue(policy.directKernelCostUnitsPerElement())
                .longValue(policy.contiguousKernelCostUnitsPerElement())
                .longValue(policy.expectedRunCount()).longValue(policy.maximumAdditionalBytes())
                .longValue(policy.minimumNetBenefitCostUnits())
                .integer(policy.minimumBenefitBasisPoints());
        encodeCosts(sink, value.portableCosts());
        var representation = value.representationCosts();
        sink.optionalLong(representation.workspaceAllocationAndBindingFixed())
                .optionalLong(representation.workspaceCostPerByte())
                .optionalLong(representation.copyInFixed())
                .optionalLong(representation.copyInPerElement())
                .optionalLong(representation.copyOutFixed())
                .optionalLong(representation.copyOutPerElement())
                .integer(value.openBlasThreadCandidates().size());
        value.openBlasThreadCandidates().forEach(candidate -> {
            sink.integer(candidate.threadCount());
            encodeCosts(sink, candidate.openBlasCosts());
        });
        sink.longValue(value.minimumNetBenefitCostUnits())
                .integer(value.minimumBenefitBasisPoints())
                .integer(value.generatedArtifactSchema()).integer(value.routePolicyVersion())
                .integer(value.costPolicyVersion());
    }

    private static void encodeBinding(CanonicalSink sink, CpuAccessPlan.Binding value) {
        var plan = value.plan();
        sink.tag(plan.accessKind()).tag(plan.regime()).integer(plan.iterationRank())
                .integer(plan.axisRoles().size());
        plan.axisRoles().forEach(sink::tag);
        sink.integer(plan.contiguousSuffix()).longs(value.extents())
                .longValue(value.baseElementOffset()).longs(value.effectiveStrides())
                .longValue(value.elementCount()).longValue(value.start()).longValue(value.end())
                .longValue(value.referencedElementSpan()).longs(value.startCoordinates())
                .longValue(value.startAddress()).longValue(value.accessedElementStart())
                .longValue(value.accessedElementEnd());
    }

    private static void encodeQualification(CanonicalSink sink,
            CpuOpenBlasTuningBatch.QualificationScope value) {
        sink.tag(value.scope());
        var target = value.targetFingerprint();
        sink.integer(target.schemaVersion()).tag(target.operatingSystem()).tag(target.machine())
                .integer(target.addressWidthBits()).integer(1);
        value.persistentIdentity().ifPresentOrElse(identity -> {
            sink.bool(true).integer(identity.qualificationSchemaVersion());
            sink.strings(identity.requiredSymbols()).tag(identity.blasIntAbi())
                    .text(identity.numericalCaseVersion());
            CpuOpenBlasQualification.BinaryIdentity binary = identity.binaryIdentity();
            sink.integer(binary.schemaVersion()).text(binary.digestAlgorithm())
                    .text(binary.sha256()).longValue(binary.byteLength())
                    .tag(binary.executableFormat()).tag(binary.machine());
        }, () -> sink.bool(false));
    }

    private static void encodeCosts(CanonicalSink sink,
            CpuPartitionAnalysisInputs.CostTerms value) {
        sink.optionalLong(value.fixedCostUnits()).optionalLong(value.costUnitsPerOutput())
                .optionalLong(value.costUnitsPerMac());
    }

    /** Scope within which compatibility and encoded decisions may be reused. */
    public enum ReuseScope {
        /** Reuse is limited to the exact live {@link CpuBackendIntegration}. */ SESSION,
        /** Stable qualified binary and workload facts permit persistent reuse. */ PERSISTENT
    }

    /** Opaque exact-association candidate batch transported through shared Prepare. */
    public static final class CandidateBatch implements BackendTuningCandidateBatch {
        private final Association association;
        private final List<Candidate> candidates;
        private CandidateBatch(Association association, List<Candidate> candidates) {
            this.association = association;
            this.candidates = candidates;
        }
        @Override public boolean equals(Object other) {
            return this == other || other instanceof CandidateBatch value
                    && association == value.association;
        }
        @Override public int hashCode() { return System.identityHashCode(association); }
        @Override public String toString() { return "CpuCandidateBatch[opaque]"; }
    }

    /** Opaque exact-association selected decision transported through shared Prepare. */
    public static final class SelectedDecision implements BackendTuningDecision {
        private final Association association;
        private final CpuOpenBlasTuningDecision internal;
        private SelectedDecision(Association association, CpuOpenBlasTuningDecision internal) {
            this.association = association;
            this.internal = internal;
        }
        @Override public boolean equals(Object other) {
            return this == other || other instanceof SelectedDecision value
                    && association == value.association && internal.equals(value.internal);
        }
        @Override public int hashCode() {
            return 31 * System.identityHashCode(association) + internal.hashCode();
        }
        @Override public String toString() { return "CpuSelectedDecision[opaque]"; }
    }

    /** Opaque exact-association complete CPU candidate. */
    public static final class Candidate {
        private final Association association;
        private final int index;
        private Candidate(Association association, int index) {
            this.association = association;
            this.index = index;
        }
        @Override public boolean equals(Object other) {
            return this == other || other instanceof Candidate value
                    && association == value.association && index == value.index;
        }
        @Override public int hashCode() {
            return 31 * System.identityHashCode(association) + index;
        }
        @Override public String toString() { return "CpuCandidate[opaque]"; }
    }

    /** Immutable canonical workload compatibility bytes and their reuse scope. */
    public static final class Compatibility {
        private final int schemaVersion;
        private final ReuseScope reuseScope;
        private final byte[] encoded;
        private Compatibility(int schemaVersion, ReuseScope reuseScope, byte[] encoded) {
            this.schemaVersion = schemaVersion;
            this.reuseScope = reuseScope;
            this.encoded = encoded.clone();
        }
        /** Returns the CPU compatibility schema version.
         * @return exact positive CPU compatibility schema version */
        public int schemaVersion() { return schemaVersion; }
        /** Copies the canonical compatibility encoding without exposing retained storage.
         * @return a fresh mutable copy of the non-empty canonical compatibility bytes */
        public byte[] bytes() { return encoded.clone(); }
        /** Returns the lifetime across which the encoded facts may be reused.
         * @return exact non-null reuse scope */
        public ReuseScope reuseScope() { return reuseScope; }
        @Override public boolean equals(Object other) {
            return this == other || other instanceof Compatibility value
                    && schemaVersion == value.schemaVersion && reuseScope == value.reuseScope
                    && Arrays.equals(encoded, value.encoded);
        }
        @Override public int hashCode() {
            return 31 * (31 * schemaVersion + reuseScope.hashCode()) + Arrays.hashCode(encoded);
        }
        @Override public String toString() {
            return "CpuCompatibility[schema=" + schemaVersion + ", scope=" + reuseScope
                    + ", bytes=" + encoded.length + "]";
        }
    }

    /** Immutable canonical candidate-identity bytes. */
    public static final class CandidateIdentity {
        private final byte[] encoded;
        private CandidateIdentity(byte[] encoded) { this.encoded = encoded.clone(); }
        /** Copies the canonical candidate identity without exposing retained storage.
         * @return a fresh mutable copy of the non-empty canonical identity bytes */
        public byte[] bytes() { return encoded.clone(); }
        @Override public boolean equals(Object other) {
            return this == other || other instanceof CandidateIdentity value
                    && Arrays.equals(encoded, value.encoded);
        }
        @Override public int hashCode() { return Arrays.hashCode(encoded); }
        @Override public String toString() {
            return "CpuCandidateIdentity[bytes=" + encoded.length + "]";
        }
    }

    private static final class Association {
        private final CpuLocalWorkloadTuning owner;
        private final CompileArtifacts artifacts;
        private final PlannedPartition partition;
        private final CpuOpenBlasTuningBatch internal;
        private Association(CpuLocalWorkloadTuning owner, CompileArtifacts artifacts,
                PlannedPartition partition, CpuOpenBlasTuningBatch internal) {
            this.owner = owner;
            this.artifacts = artifacts;
            this.partition = partition;
            this.internal = internal;
        }
    }

    private static final class CanonicalSink {
        private final MessageDigest digest;
        private CanonicalSink() {
            try {
                digest = MessageDigest.getInstance("SHA-256");
            } catch (NoSuchAlgorithmException impossible) {
                throw new ExceptionInInitializerError(impossible);
            }
        }
        private CanonicalSink bool(boolean value) { return integer(value ? 1 : 0); }
        private CanonicalSink tag(Enum<?> value) { return integer(value.ordinal() + 1); }
        private CanonicalSink integer(int value) {
            digest.update(ByteBuffer.allocate(4).putInt(value).array());
            return this;
        }
        private CanonicalSink longValue(long value) {
            digest.update(ByteBuffer.allocate(8).putLong(value).array());
            return this;
        }
        private CanonicalSink text(String value) {
            byte[] bytes = Objects.requireNonNull(value, "value").getBytes(StandardCharsets.UTF_8);
            if (bytes.length > 65_536) throw new IllegalArgumentException("canonical text too long");
            integer(bytes.length);
            digest.update(bytes);
            return this;
        }
        private CanonicalSink optionalLong(OptionalLong value) {
            bool(value.isPresent());
            if (value.isPresent()) longValue(value.getAsLong());
            return this;
        }
        private CanonicalSink longs(long[] values) {
            integer(values.length);
            for (long value : values) longValue(value);
            return this;
        }
        private CanonicalSink longs(List<Long> values) {
            integer(values.size());
            values.forEach(this::longValue);
            return this;
        }
        private CanonicalSink integers(List<Integer> values) {
            integer(values.size());
            values.forEach(this::integer);
            return this;
        }
        private CanonicalSink strings(List<String> values) {
            integer(values.size());
            values.forEach(this::text);
            return this;
        }
        private byte[] digest() { return digest.digest(); }
    }
}
