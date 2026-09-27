package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Generates complete, stable, budget-bounded Metal supported-operation route candidates.
 *
 * <p>The workload fingerprint uses only versioned semantics and structural positions, including
 * the cold numerical profile, schema-fifteen program image, exact logical descriptors, ordered
 * edges, explicit value states, target sets, dense represented-order geometry, ABI identity,
 * typed splats, and the scalar-composition source wire, exact raw constant, rank-one
 * {@code [1]} shape, operand order, and primitive opcode. Graph-local identities, partition object
 * identity, native handles, measurements, and cache state are excluded. Generation is cold,
 * thread-safe, deterministic, and performs no native work.</p>
 */
final class MetalNegRouteCandidateGenerator {
    private static final long UINT32_MAX = 0xffff_ffffL;
    private static final int WORKLOAD_SIGNATURE_VERSION = 21;
    private static final int EXACT_DEFAULT_POLICY = 21;

    /**
     * Generates every currently valid complete candidate up to a positive budget.
     *
     * @param context non-null current validated Prepare projection
     * @param plan non-null plan containing the validated structural lowering facts; its retained
     *     route does not restrict otherwise valid candidates, and it must belong to
     *     {@code context}
     * @param budget positive maximum number of candidates
     * @return non-null immutable batch whose first candidate is the existing safe heuristic
     * @throws NullPointerException if a reference is {@code null}
     * @throws IllegalArgumentException if the budget is not positive or plan association differs
     */
    MetalNegTuningBatch generate(
            PrepareContext<MetalNegAnalysisInputs> context,
            MetalNegPreparationPlan plan,
            int budget) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(plan, "plan");
        if (budget <= 0) throw new IllegalArgumentException("budget must be positive");
        if (plan.numericalProfile() != context.numericalProfile()
                || plan.partition() != context.partition()
                || plan.partitionDag() != context.partitionDag()
                || plan.context() != context.backendInputs().context()) {
            throw new IllegalArgumentException("Metal NEG candidate facts disagree");
        }

        var candidates = new ArrayList<MetalNegTuningBatch.Candidate>(2);
        if (plan.route() == MetalPreparedRoute.CUSTOM_PROGRAM) {
            candidates.add(MetalNegTuningBatch.Candidate.CUSTOM_PROGRAM);
            if (directCustomProgramCandidateIsValid(plan)) {
                candidates.add(MetalNegTuningBatch.Candidate.MPSGRAPH);
            }
        } else if (customCandidateIsValid(plan)) {
            candidates.add(MetalNegTuningBatch.Candidate.CUSTOM_SINGLE_NEG);
            candidates.add(MetalNegTuningBatch.Candidate.MPSGRAPH);
        } else {
            candidates.add(MetalNegTuningBatch.Candidate.MPSGRAPH);
        }
        List<MetalNegTuningBatch.Candidate> bounded = List.copyOf(
                candidates.subList(0, Math.min(budget, candidates.size())));
        var compatibility = new MetalNegTuningBatch.Compatibility(
                MetalNegTuningBatch.COMPATIBILITY_SCHEMA_VERSION,
                MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION,
                MetalNegTuningBatch.ROUTE_POLICY_VERSION,
                plan.numericalProfile(),
                signature(context, plan),
                new MetalNegTuningBatch.TargetCompatibility(
                        MetalNativeApi.ABI_VERSION, plan.context().sessionNonce()));
        return new MetalNegTuningBatch(compatibility, bounded);
    }

    /** Returns whether the validated structural facts admit the exact custom singleton route. */
    private static boolean customCandidateIsValid(MetalNegPreparationPlan plan) {
        if (plan.partitionDag().nodes().size() != 1
                || plan.graphProgram().nodes().getFirst().kind()
                        != MetalMpsGraphProgram.NodeKind.NEG
                || plan.feedValueIds().size() != 1
                || plan.targetValueIds().size() != 1) {
            return false;
        }
        long feedBytes = plan.feedRequiredBytes()[0];
        return feedBytes >= Float.BYTES
                && feedBytes % Float.BYTES == 0L
                && feedBytes / Float.BYTES <= UINT32_MAX;
    }

    /** Returns whether one exact custom-program occurrence has a direct MPSGraph candidate. */
    private static boolean directCustomProgramCandidateIsValid(MetalNegPreparationPlan plan) {
        if (plan.partitionDag().nodes().size() != 1
                || plan.graphProgram().nodes().size() != 1
                || plan.targetValueIds().size() != 1
                || !plan.internalValueIds().isEmpty()) {
            return false;
        }
        int wire = plan.graphProgram().nodes().getFirst().kind().wireIdentity();
        return wire >= 40 && wire <= 45
                || wire == 51
                || wire >= 60 && wire <= 63;
    }

    /**
     * Constructs an opaque no-decision handoff for callers that want the current batch.
     *
     * @param context non-null current Prepare projection
     * @param plan non-null associated provisional analysis plan
     * @param budget positive candidate budget
     * @return non-null exact-partition handoff with an empty selected decision
     * @throws NullPointerException if a reference is {@code null}
     * @throws IllegalArgumentException if the budget or plan association is invalid
     */
    BackendPartitionTuningHandoff<MetalNegTuningBatch, MetalNegTuningDecision> absentHandoff(
            PrepareContext<MetalNegAnalysisInputs> context,
            MetalNegPreparationPlan plan,
            int budget) {
        MetalNegTuningBatch batch = generate(context, plan, budget);
        return new BackendPartitionTuningHandoff<>(
                context.partition(), batch, Optional.empty());
    }

    /**
     * Constructs one opaque decision-present handoff for a member of an existing Metal batch.
     *
     * @param partition exact non-null partition associated with {@code batch}
     * @param batch non-null Metal candidate batch
     * @param selectedCandidate non-null current candidate member to select
     * @return non-null handoff carrying a structurally complete untrusted decision
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the candidate is not in {@code batch}
     */
    BackendPartitionTuningHandoff<MetalNegTuningBatch, MetalNegTuningDecision> presentHandoff(
            io.github.pho001.synaptik.planning.partition.PlannedPartition partition,
            MetalNegTuningBatch batch,
            MetalNegTuningBatch.Candidate selectedCandidate) {
        Objects.requireNonNull(partition, "partition");
        Objects.requireNonNull(batch, "batch");
        Objects.requireNonNull(selectedCandidate, "selectedCandidate");
        if (batch.find(selectedCandidate).isEmpty()) {
            throw new IllegalArgumentException("selected Metal NEG candidate is absent");
        }
        var decision = new MetalNegTuningDecision(
                MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION,
                batch.compatibility(), selectedCandidate);
        return new BackendPartitionTuningHandoff<>(
                partition, batch, Optional.of(decision));
    }

    private static MetalNegTuningBatch.WorkloadSignature signature(
            PrepareContext<MetalNegAnalysisInputs> context, MetalNegPreparationPlan plan) {
        MessageDigest digest = sha256();
        updateInt(digest, WORKLOAD_SIGNATURE_VERSION);
        updateInt(digest, EXACT_DEFAULT_POLICY);
        updateInt(digest, numericalProfileWireValue(plan.numericalProfile()));
        updateInt(digest, MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION);
        updateInt(digest, MetalNegTuningBatch.ROUTE_POLICY_VERSION);
        updateInt(digest, MetalNativeApi.ABI_VERSION);
        updateInt(digest, MetalMpsGraphProgram.SCHEMA_VERSION);

        List<ValueId> valueIds = plan.valueIds();
        Map<ValueId, Integer> valuePositions = new LinkedHashMap<>();
        for (int index = 0; index < valueIds.size(); index++) {
            valuePositions.put(valueIds.get(index), index);
        }
        updateInt(digest, context.nodes().size());
        updateInt(digest, plan.route().wireIdentity());
        MetalPreparedRoute imageRoute = plan.route() == MetalPreparedRoute.CUSTOM_SINGLE_NEG
                ? MetalPreparedRoute.MPSGRAPH
                : plan.route();
        plan.graphProgram().updateDigest(
                digest, plan.programValueDescriptors(),
                plan.feedValueIndices(), plan.targetValueIndices(), imageRoute);
        updateInt(digest, plan.valueStates().size());
        for (MetalMpsGraphProgram.ValueState state : plan.valueStates()) {
            updateInt(digest, state.wireIdentity());
        }
        updateLongs(digest, plan.feedRequiredBytes());
        updateLongs(digest, plan.targetRequiredBytes());

        updateInt(digest, plan.descriptors().size());
        for (var descriptor : plan.descriptors()) {
            updateInt(digest, dataTypeWireValue(descriptor.dataType()));
            updateBoolean(digest, descriptor.requiresGrad());
            long[] dimensions = descriptor.shape().toLongArray();
            updateLongs(digest, dimensions);
            var layout = descriptor.layout().orElseThrow();
            updateInt(digest, layout.kind().ordinal());
            updateLongs(digest, layout.strides());
            updateLong(digest, layout.storageOffset());
            updateBoolean(digest, layout.isView());
            updateLong(digest, layout.referencedElementSpan());
        }

        Map<ValueId, LogicalMemoryRequirement> requirements = new LinkedHashMap<>();
        context.memoryRequirements().forEach(
                requirement -> requirements.put(requirement.valueId(), requirement));
        for (ValueId valueId : valueIds) {
            LogicalMemoryRequirement requirement = Objects.requireNonNull(
                    requirements.get(valueId), "logical requirement");
            int producerKind = requirement.producerPartition().isEmpty() ? 0
                    : requirement.producerPartition().orElseThrow() == context.partition() ? 1 : 2;
            updateInt(digest, producerKind);
            int localConsumers = 0;
            int externalConsumers = 0;
            for (var consumer : requirement.consumerPartitions()) {
                if (consumer == context.partition()) localConsumers++;
                else externalConsumers++;
            }
            updateInt(digest, localConsumers);
            updateInt(digest, externalConsumers);
            updateBoolean(digest, requirement.graphOutput());
        }

        updatePositions(digest, plan.feedValueIds(), valuePositions);
        updatePositions(digest, plan.targetValueIds(), valuePositions);
        updateInt(digest, plan.feedSplats().size());
        for (var splat : plan.feedSplats()) {
            updateBoolean(digest, splat.isPresent());
            if (splat.isPresent()) {
                var scalar = splat.orElseThrow();
                updateInt(digest, dataTypeWireValue(scalar.dataType()));
                switch (scalar.dataType()) {
                    case FLOAT64 ->
                            updateLong(digest, Double.doubleToRawLongBits(scalar.float64Value()));
                    case FLOAT32 ->
                            updateInt(digest, Float.floatToRawIntBits(scalar.float32Value()));
                    case BFLOAT16 -> updateInt(digest, scalar.bfloat16Bits() & 0xffff);
                    case INT64 -> updateLong(digest, scalar.int64Value());
                    case INT32 -> updateInt(digest, scalar.int32Value());
                    case BOOL -> updateBoolean(digest, scalar.booleanValue());
                }
            }
        }
        return new MetalNegTuningBatch.WorkloadSignature(digest.digest());
    }

    private static int dataTypeWireValue(DataType dataType) {
        return MetalMpsGraphProgram.dataTypeWire(dataType);
    }

    private static int numericalProfileWireValue(NumericalProfile profile) {
        return switch (profile) {
            case STRICT_IEEE -> 0x53545249;
            case ACCELERATOR -> 0x41434345;
        };
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    private static void updatePositions(
            MessageDigest digest, List<ValueId> ids, Map<ValueId, Integer> positions) {
        updateInt(digest, ids.size());
        for (ValueId id : ids) {
            Integer position = positions.get(id);
            if (position == null) throw new IllegalArgumentException("boundary value is absent");
            updateInt(digest, position);
        }
    }

    private static void updateInts(MessageDigest digest, int[] values) {
        updateInt(digest, values.length);
        for (int value : values) updateInt(digest, value);
    }

    private static void updateLongs(MessageDigest digest, long[] values) {
        updateInt(digest, values.length);
        for (long value : values) updateLong(digest, value);
    }

    private static void updateBytes(MessageDigest digest, byte[] values) {
        updateInt(digest, values.length);
        digest.update(values);
    }

    private static void updateBoolean(MessageDigest digest, boolean value) {
        digest.update((byte) (value ? 1 : 0));
    }

    private static void updateInt(MessageDigest digest, int value) {
        digest.update(ByteBuffer.allocate(Integer.BYTES).putInt(value).array());
    }

    private static void updateLong(MessageDigest digest, long value) {
        digest.update(ByteBuffer.allocate(Long.BYTES).putLong(value).array());
    }
}
