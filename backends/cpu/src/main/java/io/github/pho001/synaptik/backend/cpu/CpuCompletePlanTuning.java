package io.github.pho001.synaptik.backend.cpu;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuBackendComposition;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasTuningDecision;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;
import java.util.zip.CRC32;

/**
 * Supported CPU-owned collaboration for enumerating and freshly preparing complete CPU plans.
 *
 * <p>The collaboration is retained by one {@link CpuBackendIntegration}. For the same exact
 * projected sole non-empty CPU partition, each candidate fixes one
 * already-retained legal fusion/split topology, one direct, single-copy, or eligible
 * disjoint-two-copy representation, and the exact authenticated Phase-1 local route and
 * configuration when that workload is eligible. Upstream graph and publication semantics remain
 * outside this collaboration; the supplied Planning ownership, partition boundary, and logical
 * memory remain fixed.</p>
 *
 * <p>The collaboration exposes opaque, association-checked values only. It performs no
 * representative execution, measurement, Phase-1 ranking, cache input/output, Engine policy, or
 * Runtime selection. Candidate-only copied representations do not change ordinary heuristic
 * preparation. Trial selection returns a fresh backend preparation for Engine to compose; it
 * does not bind inputs, execute, or timestamp a trial.</p>
 *
 * <p>Batch, candidate, and decision values are immutable, own no closeable resource, and borrow
 * the exact owning integration, projected context, and partition association. Compatibility and
 * identity wrappers own defensive canonical bytes; current compatibility is session-scoped. Association
 * checks prevent accidental value mixing, while the checksum detects accidental byte corruption;
 * neither mechanism is a hostile-input security boundary.</p>
 */
public final class CpuCompletePlanTuning {
    private static final int VALUE_SCHEMA = 1;
    private static final int DECISION_MAGIC = 0x53435032;
    private static final int MAXIMUM_DECISION_BYTES = 2_048;
    private static final int CPU_BACKEND_TAG = 0x435055;
    private static final int CANDIDATE_SCHEMA = 1;

    private final CpuBackendComposition composition;
    private final CpuLocalWorkloadTuning localWorkloadTuning;
    private final byte[] sessionNonce;

    /**
     * Creates the retained collaboration for one exact CPU integration.
     * @param composition non-null borrowed CPU composition owner
     * @param localWorkloadTuning non-null retained Phase-1 collaboration
     * @throws NullPointerException if an argument is {@code null}
     * @hidden supported callers use {@link CpuBackendIntegration#completePlanTuning()}
     */
    CpuCompletePlanTuning(CpuBackendComposition composition,
            CpuLocalWorkloadTuning localWorkloadTuning) {
        this.composition = Objects.requireNonNull(composition, "composition");
        this.localWorkloadTuning = Objects.requireNonNull(localWorkloadTuning,
                "localWorkloadTuning");
        UUID nonce = UUID.randomUUID();
        sessionNonce = ByteBuffer.allocate(16).putLong(nonce.getMostSignificantBits())
                .putLong(nonce.getLeastSignificantBits()).array();
    }

    /**
     * Produces every complete retained Phase-2 alternative for one supported CPU projection.
     *
     * @param context exact non-null stable one-partition projection retained by a successful batch
     * @param phaseOneDecision non-null optional exact Phase-1 decision; required precisely when
     *     the projection has an eligible local tuning batch
     * @return an associated batch in stable topology/representation order, or empty when fewer
     *     than two complete alternatives exist or completeness cannot be proved
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the projection or Phase-1 association is invalid
     * @throws IllegalStateException if the owning integration is closed
     */
    public Optional<BackendPartitionTuningHandoff<CandidateBatch, SelectedDecision>>
            candidateHandoff(PrepareContext<?> context,
                    Optional<CpuLocalWorkloadTuning.SelectedDecision> phaseOneDecision) {
        requireOpen();
        Objects.requireNonNull(context, "context");
        phaseOneDecision = Objects.requireNonNull(phaseOneDecision, "phaseOneDecision");
        Optional<CpuOpenBlasTuningDecision> internalPhaseOne = localWorkloadTuning
                .authenticateCompletePlanPhaseOne(context, phaseOneDecision);
        byte[] phaseOneFingerprint = localWorkloadTuning
                .completePlanPhaseOneFingerprint(phaseOneDecision);
        CpuPartitionPreparer.CompletePlanCandidates retained = composition
                .completePlanCandidates(context, internalPhaseOne);
        List<CpuRepresentationDecision.VariantIdentity> identities = retained.identities().stream()
                .distinct().toList();
        if (!retained.complete() || identities.size() < 2) return Optional.empty();
        Association association = new Association(this, context,
                context.partition(), identities, phaseOneDecision,
                internalPhaseOne, phaseOneFingerprint);
        var candidates = new ArrayList<Candidate>(identities.size());
        for (int index = 0; index < identities.size(); index++) {
            candidates.add(new Candidate(association, index));
        }
        CandidateBatch batch = new CandidateBatch(association, List.copyOf(candidates));
        return Optional.of(new BackendPartitionTuningHandoff<>(association.partition, batch,
                Optional.empty()));
    }

    /**
     * Returns the immutable stable candidate list for an exact owned batch.
     *
     * @param batch non-null batch issued by this collaboration
     * @return retained immutable list with at least two opaque candidates
     * @throws NullPointerException if {@code batch} is {@code null}
     * @throws IllegalArgumentException if the batch has another owner
     * @throws IllegalStateException if the owning integration is closed
     */
    public List<Candidate> candidates(CandidateBatch batch) {
        return requireBatch(batch).candidates;
    }

    /**
     * Returns defensive canonical compatibility bytes for an exact owned batch.
     *
     * @param batch non-null batch issued by this collaboration
     * @return a new immutable session-scoped compatibility value whose byte accessor returns
     *     defensive caller-owned copies
     * @throws NullPointerException if {@code batch} is {@code null}
     * @throws IllegalArgumentException if the batch has another owner
     * @throws IllegalStateException if the owning integration is closed
     */
    public Compatibility compatibility(CandidateBatch batch) {
        return compatibility(requireBatch(batch).association);
    }

    /**
     * Returns defensive canonical identity bytes for one exact owned candidate.
     *
     * @param candidate non-null candidate issued by this collaboration
     * @return a new immutable canonical candidate identity whose byte accessor returns defensive
     *     caller-owned copies
     * @throws NullPointerException if {@code candidate} is {@code null}
     * @throws IllegalArgumentException if the candidate has another owner
     * @throws IllegalStateException if the owning integration is closed
     */
    public CandidateIdentity candidateIdentity(Candidate candidate) {
        Association association = requireCandidate(candidate);
        return identity(association, candidate.index);
    }

    /**
     * Creates one opaque exact-batch decision for the supplied candidate.
     *
     * @param batch non-null originating batch
     * @param candidate non-null exact member of {@code batch}
     * @return non-null immutable decision associated with the exact batch
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if either value has another association
     * @throws IllegalStateException if the owning integration is closed
     */
    public SelectedDecision selectedDecision(CandidateBatch batch, Candidate candidate) {
        Association association = requireBatch(batch).association;
        requireCandidate(candidate);
        if (candidate.association != association) {
            throw new IllegalArgumentException("candidate does not belong to batch");
        }
        return new SelectedDecision(association, candidate.index);
    }

    /**
     * Encodes an associated decision in the bounded checksummed CPU format.
     *
     * @param decision non-null decision issued by this collaboration
     * @return fresh caller-owned canonical bytes containing no live object or graph-local ID
     * @throws NullPointerException if {@code decision} is {@code null}
     * @throws IllegalArgumentException if it has another owner
     * @throws IllegalStateException if the owning integration is closed
     */
    public byte[] encodeDecision(SelectedDecision decision) {
        requireDecision(decision);
        byte[] compatibility = compatibility(decision.association).encoded;
        byte[] identity = identity(decision.association, decision.index).encoded;
        byte[] phaseOne = decision.association.phaseOneFingerprint;
        ByteBuffer body = ByteBuffer.allocate(21 + compatibility.length + identity.length
                + phaseOne.length);
        body.putInt(DECISION_MAGIC).putInt(VALUE_SCHEMA).put((byte) ReuseScope.SESSION.ordinal())
                .putInt(compatibility.length).put(compatibility)
                .putInt(identity.length).put(identity)
                .putInt(phaseOne.length).put(phaseOne);
        byte[] bodyBytes = body.array();
        CRC32 checksum = new CRC32();
        checksum.update(bodyBytes);
        return ByteBuffer.allocate(bodyBytes.length + 4).put(bodyBytes)
                .putInt((int) checksum.getValue()).array();
    }

    /**
     * Decodes only a checksummed decision compatible with the supplied exact current batch.
     *
     * @param batch non-null current batch issued by this collaboration
     * @param encodedDecision non-null caller-owned bytes, defensively copied before parsing
     * @return matching associated decision, or empty for malformed, corrupt, stale,
     *     incompatible, unknown, or trailing input
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if {@code batch} has another owner
     * @throws IllegalStateException if the owning integration is closed
     */
    public Optional<SelectedDecision> decodeCompatibleDecision(CandidateBatch batch,
            byte[] encodedDecision) {
        Association association = requireBatch(batch).association;
        byte[] input = Objects.requireNonNull(encodedDecision, "encodedDecision").clone();
        if (input.length < 25 || input.length > MAXIMUM_DECISION_BYTES) return Optional.empty();
        try {
            byte[] body = Arrays.copyOf(input, input.length - 4);
            CRC32 checksum = new CRC32();
            checksum.update(body);
            if ((int) checksum.getValue() != ByteBuffer.wrap(input, input.length - 4, 4).getInt()) {
                return Optional.empty();
            }
            ByteBuffer bytes = ByteBuffer.wrap(body);
            if (bytes.getInt() != DECISION_MAGIC || bytes.getInt() != VALUE_SCHEMA
                    || Byte.toUnsignedInt(bytes.get()) != ReuseScope.SESSION.ordinal()) {
                return Optional.empty();
            }
            byte[] currentCompatibility = compatibility(association).encoded;
            byte[] compatibility = boundedBytes(bytes, currentCompatibility.length);
            byte[] identity = boundedBytes(bytes, 512);
            byte[] phaseOne = boundedBytes(bytes, 64);
            if (bytes.hasRemaining()
                    || !Arrays.equals(compatibility, currentCompatibility)
                    || !Arrays.equals(phaseOne, association.phaseOneFingerprint)) {
                return Optional.empty();
            }
            for (int index = 0; index < association.identities.size(); index++) {
                if (Arrays.equals(identity, identity(association, index).encoded)) {
                    return Optional.of(new SelectedDecision(association, index));
                }
            }
            return Optional.empty();
        } catch (RuntimeException malformed) {
            return Optional.empty();
        }
    }

    /**
     * Freshly prepares the requested candidate without executing or measuring it.
     *
     * @param batch non-null originating batch
     * @param candidate non-null exact member of {@code batch}
     * @return the immutable CPU-owned partition preparation for the trial candidate
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if association or fresh compatibility fails
     * @throws IllegalStateException if the owning integration is closed
     */
    public PartitionPreparation<?, ?> trialPreparation(
            CandidateBatch batch, Candidate candidate) {
        return selectedPreparation(batch, selectedDecision(batch, candidate));
    }

    /**
     * Freshly prepares exactly the selected retained plan without heuristic fallback.
     *
     * @param batch non-null originating batch
     * @param decision non-null decision associated with exactly {@code batch}
     * @return the immutable CPU-owned partition preparation for the selected decision
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if association, Phase-1 state, candidate set, or fresh
     *     exact selection fails
     * @throws IllegalStateException if the owning integration is closed
     */
    public PartitionPreparation<?, ?> selectedPreparation(
            CandidateBatch batch, SelectedDecision decision) {
        Association association = requireBatch(batch).association;
        requireDecision(decision);
        if (decision.association != association) {
            throw new IllegalArgumentException("decision does not belong to batch");
        }
        Optional<CpuOpenBlasTuningDecision> phaseOne = localWorkloadTuning
                .authenticateCompletePlanPhaseOne(association.context,
                        association.phaseOneDecision);
        if (!phaseOne.equals(association.internalPhaseOne)) {
            throw new IllegalArgumentException("CPU Phase-1 decision association changed");
        }
        CpuPartitionPreparer.CompletePlanCandidates fresh = composition.completePlanCandidates(
                association.context, phaseOne);
        if (!fresh.complete() || !fresh.identities().equals(association.identities)) {
            throw new IllegalArgumentException("CPU complete-plan candidate set changed");
        }
        var selected = new CpuPartitionPreparer.SelectedCompletePlan(VALUE_SCHEMA,
                association.identities.get(decision.index), unsigned(
                        association.phaseOneFingerprint));
        return composition.completePlanPreparation(association.context, phaseOne, selected);
    }

    private void requireOpen() { composition.assertOpen(); }

    private CandidateBatch requireBatch(CandidateBatch batch) {
        requireOpen();
        Objects.requireNonNull(batch, "batch");
        if (batch.association.owner != this) {
            throw new IllegalArgumentException("batch belongs to another CPU integration");
        }
        return batch;
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

    private static byte[] boundedBytes(ByteBuffer bytes, int maximum) {
        int length = bytes.getInt();
        if (length <= 0 || length > maximum || length > bytes.remaining()) {
            throw new IllegalArgumentException("invalid encoded CPU decision length");
        }
        byte[] value = new byte[length];
        bytes.get(value);
        return value;
    }

    private CandidateIdentity identity(Association association, int index) {
        CanonicalSink sink = new CanonicalSink();
        sink.integer(VALUE_SCHEMA).bytes(compatibility(association).encoded)
                .bytes(association.phaseOneFingerprint);
        encodeVariant(sink, association.identities.get(index));
        byte[] digest = sink.digest();
        return new CandidateIdentity(ByteBuffer.allocate(8 + digest.length)
                .putInt(VALUE_SCHEMA).putInt(digest.length).put(digest).array());
    }

    private Compatibility compatibility(Association association) {
        CanonicalSink sink = new CanonicalSink();
        sink.integer(VALUE_SCHEMA).integer(CANDIDATE_SCHEMA).integer(CPU_BACKEND_TAG)
                .bytes(sessionNonce).integer(association.identities.size());
        encodeProjection(sink, association.context, association.partition);
        association.identities.forEach(identity -> encodeVariant(sink, identity));
        encodeEnumerationProfile(sink);
        sink.bytes(association.phaseOneFingerprint);
        byte[] digest = sink.digest();
        byte[] encoded = ByteBuffer.allocate(9 + sessionNonce.length + digest.length)
                .putInt(VALUE_SCHEMA).put((byte) ReuseScope.SESSION.ordinal())
                .putInt(sessionNonce.length).put(sessionNonce).put(digest).array();
        return new Compatibility(VALUE_SCHEMA, ReuseScope.SESSION, encoded);
    }

    private static void encodeProjection(CanonicalSink sink, PrepareContext<?> context,
            PlannedPartition partition) {
        var values = new LinkedHashMap<ValueId, Integer>();
        for (int index = 0; index < context.values().size(); index++) {
            values.put(context.values().get(index).id(), index);
        }
        sink.integer(context.values().size()).integer(partition.nodeIds().size());
        context.values().forEach(value -> {
            var descriptor = value.descriptor();
            sink.tag(descriptor.dataType()).integer(descriptor.shape().rank());
            descriptor.shape().dimensions().forEach(dimension ->
                    sink.longValue(dimension.staticSize().orElseThrow()));
            var layout = descriptor.layout().orElseThrow();
            sink.integer(layout.rank()).longs(layout.strides()).longValue(layout.storageOffset())
                    .tag(layout.kind()).bool(layout.isView())
                    .longValue(layout.referencedElementSpan());
        });
        sink.integer(context.memoryRequirements().size());
        context.memoryRequirements().forEach(requirement -> {
            sink.integer(values.get(requirement.valueId()));
            sink.integer(requirement.producerPartition().isEmpty()
                    ? 0
                    : requirement.producerPartition().orElseThrow() == partition ? 1 : 2);
            sink.integer(requirement.consumerPartitions().size());
            requirement.consumerPartitions().forEach(consumer ->
                    sink.integer(consumer == partition ? 1 : 0));
            sink.bool(requirement.graphOutput());
        });
        var nodes = new LinkedHashMap<io.github.pho001.synaptik.model.graph.NodeId,
                io.github.pho001.synaptik.model.graph.CompiledNode>();
        context.partitionDag().nodes().forEach(node -> nodes.put(node.id(), node));
        partition.nodeIds().forEach(nodeId -> {
            var node = Objects.requireNonNull(nodes.get(nodeId), "partition node");
            sink.integer(node.inputs().size());
            node.inputs().forEach(value -> sink.integer(values.get(value)));
            sink.integer(node.outputs().size());
            node.outputs().forEach(value -> sink.integer(values.get(value)));
        });
        var inputIds = context.memoryRequirements().stream()
                .filter(value -> value.producerPartition().isEmpty())
                .map(value -> value.valueId()).toList();
        sink.integer(inputIds.size());
        inputIds.forEach(value -> sink.integer(values.get(value)));
        inputIds.forEach(valueId -> {
            var scalar = context.constants().get(valueId);
            sink.bool(scalar != null);
            if (scalar != null) {
                sink.tag(scalar.dataType());
                switch (scalar.dataType()) {
                    case FLOAT64 -> sink.longValue(Double.doubleToRawLongBits(
                            scalar.float64Value()));
                    case FLOAT32 -> sink.integer(Float.floatToRawIntBits(scalar.float32Value()));
                    case BFLOAT16 -> sink.integer(Short.toUnsignedInt(scalar.bfloat16Bits()));
                    case INT64 -> sink.longValue(scalar.int64Value());
                    case INT32 -> sink.integer(scalar.int32Value());
                    case BOOL -> sink.bool(scalar.booleanValue());
                }
            }
        });
    }

    private static void encodeEnumerationProfile(CanonicalSink sink) {
        sink.integer(1).bool(true).longValue(0).longValue(1).longValue(3)
                .longValue(1).longValue(1).longValue(Long.MAX_VALUE)
                .longValue(0).integer(0);
    }

    private static void encodeVariant(CanonicalSink sink,
            CpuRepresentationDecision.VariantIdentity identity) {
        CpuFusionDecision.CandidateIdentity topology = identity.topology();
        sink.integer(topology.units().size());
        topology.units().forEach(unit -> {
            sink.integers(unit.memberNodePositions()).integers(unit.dependencyUnitPositions())
                    .integers(unit.portableIrStructuralKey().octets());
            encodeSpecialization(sink, unit.specialization());
            sink.tag(unit.strategy()).tag(unit.topology()).integer(unit.boundaries().size());
            unit.boundaries().forEach(boundary -> sink.integer(boundary.relativeBoundaryPosition())
                    .integer(boundary.unitBoundaryPosition()).tag(boundary.role())
                    .tag(boundary.regime()).longValue(boundary.referencedBytes())
                    .longValue(boundary.byteAlignment()));
            sink.bool(unit.workspace().isPresent());
            unit.workspace().ifPresent(workspace -> sink.tag(workspace.role())
                    .longValue(workspace.byteSize()).longValue(workspace.byteAlignment()));
        });
        sink.integer(identity.materializations().size());
        identity.materializations().forEach(value -> {
            sink.integer(value.sourceBoundaryPosition()).tag(value.dataType())
                    .tag(value.sourceCarrier());
            encodeBinding(sink, value.sourceBinding());
            sink.tag(value.consumerCarrier());
            encodeBinding(sink, value.consumerBinding());
            sink.integer(value.consumers().size());
            value.consumers().forEach(consumer -> sink.integer(consumer.unitPosition())
                    .integer(consumer.boundaryPosition())
                    .longValue(consumer.instructionUseCount()));
            sink.longValue(value.instructionUseCount()).integer(value.reuseUnitCount())
                    .integer(value.reusePositionCount()).longValue(value.elementCount())
                    .longValue(value.byteCount()).integer(value.workspaceRequirementId())
                    .longValue(value.workspaceBytes()).longValue(value.workspaceAlignment())
                    .tag(value.copyStrategy()).integers(value.copyStructuralKey().octets());
            encodeSpecialization(sink, value.copySpecialization());
        });
    }

    private static void encodeSpecialization(CanonicalSink sink,
            CpuKernelSpecialization value) {
        sink.text(value.loweringFingerprint().hex()).tag(value.numericalMode())
                .tag(value.executionStrategy().compute())
                .integer(value.boundaryDataTypes().size());
        value.boundaryDataTypes().forEach(sink::tag);
        sink.integer(value.carrierPattern().size());
        value.carrierPattern().forEach(sink::tag);
        sink.integer(value.vectorSpeciesBitSize()).integer(value.materializedSourcePosition())
                .bool(value.scratchParameter()).integer(value.classIdentitySchema())
                .text(value.structuralKey());
    }

    private static void encodeBinding(CanonicalSink sink, CpuAccessPlan.Binding value) {
        CpuAccessPlan plan = value.plan();
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

    private static List<Integer> unsigned(byte[] values) {
        var result = new ArrayList<Integer>(values.length);
        for (byte value : values) result.add(Byte.toUnsignedInt(value));
        return List.copyOf(result);
    }

    /**
     * Scope within which compatibility and encoded decisions may be reused.
     *
     * <p>The current complete-plan producer emits only {@link #SESSION}; {@link #PERSISTENT}
     * reserves the explicit value for a later producer that can project every target fact
     * persistently.</p>
     */
    public enum ReuseScope {
        /** Reuse is limited to the exact live integration. */ SESSION,
        /** All target facts have a stable persistent projection. */ PERSISTENT
    }

    /**
     * Opaque immutable complete-plan candidate batch associated with one exact live integration,
     * projected context, partition, and authenticated Phase-1 state.
     *
     * <p>The batch owns no resource and is meaningful only through its originating
     * {@link CpuCompletePlanTuning} while that integration remains open.</p>
     */
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
        @Override public String toString() { return "CpuCompletePlanCandidateBatch[opaque]"; }
    }

    /**
     * Opaque immutable selection of one candidate in one exact complete-plan batch.
     *
     * <p>The decision owns no resource and cannot be applied to another batch or integration.</p>
     */
    public static final class SelectedDecision implements BackendTuningDecision {
        private final Association association;
        private final int index;
        private SelectedDecision(Association association, int index) {
            this.association = association;
            this.index = index;
        }
        @Override public boolean equals(Object other) {
            return this == other || other instanceof SelectedDecision value
                    && association == value.association && index == value.index;
        }
        @Override public int hashCode() {
            return 31 * System.identityHashCode(association) + index;
        }
        @Override public String toString() { return "CpuCompletePlanSelectedDecision[opaque]"; }
    }

    /**
     * Opaque immutable complete CPU candidate associated with one exact batch.
     *
     * <p>The value owns no resource and exposes no topology, representation, route, slot, or
     * executable detail.</p>
     */
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
        @Override public String toString() { return "CpuCompletePlanCandidate[opaque]"; }
    }

    /**
     * Immutable complete-plan compatibility value with defensively exposed canonical bytes.
     * Equality compares the schema, reuse scope, and byte content rather than live association.
     */
    public static final class Compatibility {
        private final int schemaVersion;
        private final ReuseScope reuseScope;
        private final byte[] encoded;
        private Compatibility(int schemaVersion, ReuseScope reuseScope, byte[] encoded) {
            this.schemaVersion = schemaVersion;
            this.reuseScope = reuseScope;
            this.encoded = encoded.clone();
        }
        /** Returns the positive CPU complete-plan schema version.
         * @return exact positive schema version */
        public int schemaVersion() { return schemaVersion; }
        /** Returns a fresh caller-owned copy of the canonical bytes.
         * @return non-empty mutable copy independent of retained storage */
        public byte[] bytes() { return encoded.clone(); }
        /** Returns the exact compatibility reuse scope.
         * @return non-null reuse scope */
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
            return "CpuCompletePlanCompatibility[schema=" + schemaVersion + ", scope="
                    + reuseScope + ", bytes=" + encoded.length + "]";
        }
    }

    /**
     * Immutable candidate identity with defensively exposed canonical bytes.
     * Equality compares byte content and exposes no graph-local numeric identity.
     */
    public static final class CandidateIdentity {
        private final byte[] encoded;
        private CandidateIdentity(byte[] encoded) { this.encoded = encoded.clone(); }
        /** Returns a fresh caller-owned copy of the canonical identity bytes.
         * @return non-empty mutable copy independent of retained storage */
        public byte[] bytes() { return encoded.clone(); }
        @Override public boolean equals(Object other) {
            return this == other || other instanceof CandidateIdentity value
                    && Arrays.equals(encoded, value.encoded);
        }
        @Override public int hashCode() { return Arrays.hashCode(encoded); }
        @Override public String toString() {
            return "CpuCompletePlanCandidateIdentity[bytes=" + encoded.length + "]";
        }
    }

    private static final class Association {
        private final CpuCompletePlanTuning owner;
        private final PrepareContext<?> context;
        private final PlannedPartition partition;
        private final List<CpuRepresentationDecision.VariantIdentity> identities;
        private final Optional<CpuLocalWorkloadTuning.SelectedDecision> phaseOneDecision;
        private final Optional<CpuOpenBlasTuningDecision> internalPhaseOne;
        private final byte[] phaseOneFingerprint;
        private Association(CpuCompletePlanTuning owner, PrepareContext<?> context,
                PlannedPartition partition,
                List<CpuRepresentationDecision.VariantIdentity> identities,
                Optional<CpuLocalWorkloadTuning.SelectedDecision> phaseOneDecision,
                Optional<CpuOpenBlasTuningDecision> internalPhaseOne,
                byte[] phaseOneFingerprint) {
            this.owner = owner;
            this.context = context;
            this.partition = partition;
            this.identities = List.copyOf(identities);
            this.phaseOneDecision = phaseOneDecision;
            this.internalPhaseOne = internalPhaseOne;
            this.phaseOneFingerprint = phaseOneFingerprint.clone();
        }
    }

    private static final class CanonicalSink {
        private final MessageDigest digest;
        private CanonicalSink() {
            try { digest = MessageDigest.getInstance("SHA-256"); }
            catch (NoSuchAlgorithmException impossible) {
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
            byte[] bytes = Objects.requireNonNull(value, "value")
                    .getBytes(java.nio.charset.StandardCharsets.UTF_8);
            integer(bytes.length);
            digest.update(bytes);
            return this;
        }
        private CanonicalSink bytes(byte[] values) {
            integer(values.length);
            digest.update(values);
            return this;
        }
        private CanonicalSink integers(List<Integer> values) {
            integer(values.size());
            values.forEach(this::integer);
            return this;
        }
        private CanonicalSink longs(List<Long> values) {
            integer(values.size());
            values.forEach(this::longValue);
            return this;
        }
        private CanonicalSink longs(long[] values) {
            integer(values.length);
            for (long value : values) longValue(value);
            return this;
        }
        private byte[] digest() { return digest.digest(); }
    }
}
