package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Retains the immutable, shape-specialized lowering and route facts for one whole supported Metal
 * partition.
 *
 * <p>The retained numerical profile closes the operation domain: both profiles admit the common
 * exact rows and no-gradient promoted INT32/INT64 MATMUL, while accelerator additionally admits
 * its arithmetic/reduction rows, general positive-static FLOAT32 MATMUL, and no-gradient mixed
 * BFLOAT16/FLOAT32 MATMUL. An affine MATMUL input must be the exact local identity-prefix,
 * last-two-axis transpose authenticated on that consuming edge; the same affine value may
 * otherwise be consumed or published normally. Value indices, explicit canonical/affine-view
 * states, typed
 * MPSGraph nodes, feeds, targets, and declarations are already in their stable ABI order. The
 * route is either the safe heuristic, a freshly authenticated session-compatible decision, or an
 * approved package-private test force, and is fixed before this plan's declarations escape
 * analysis. The plan contains no assigned slot, tuning value, native executable, physical buffer,
 * or per-run state. Affine targets retain exact logical view descriptors alongside their full
 * dense represented-order byte extents. The address workspace is absent only for the singleton
 * custom NEG route. Primitive arrays are privately snapshotted and copied when marshalled.</p>
 */
final class MetalNegPreparationPlan implements BackendPreparationPlan {

    private final NumericalProfile numericalProfile;
    private final PlannedPartition partition;
    private final PartitionDag partitionDag;
    private final MetalDeviceContext context;
    private final MetalPreparedRoute route;
    private final List<ValueId> valueIds;
    private final List<TensorDescriptor> descriptors;
    private final List<MetalMpsGraphProgram.ValueDescriptor> programValueDescriptors;
    private final List<MetalMpsGraphProgram.ValueState> valueStates;
    private final MetalMpsGraphProgram graphProgram;
    private final List<ValueId> feedValueIds;
    private final int[] feedValueIndices;
    private final List<ValueId> targetValueIds;
    private final int[] targetValueIndices;
    private final List<ValueId> internalValueIds;
    private final int[] internalValueIndices;
    private final long[] internalRequiredBytes;
    private final List<PreparationResourceRequirement.Buffer> declarations;
    private final List<Optional<ScalarValue>> feedSplats;
    private final boolean[] feedSplatSources;
    private final Optional<PreparationResourceRequirement.Workspace> addressWorkspace;
    private final long[] feedRequiredBytes;
    private final long[] targetRequiredBytes;
    private final MetalTraceProducer.PreparedUnit traceUnit;

    /**
     * Creates one completely validated analysis result snapshot.
     *
     * @param numericalProfile non-null immutable graph-wide numerical-profile identity
     * @param partition exact non-null planned partition analyzed to produce this plan
     * @param partitionDag exact non-null partition topology retaining {@code partition}
     * @param context exact non-null Metal device context retained by identity
     * @param route non-null canonical implementation route selected during analysis
     * @param valueIds non-null stable indexed value identities
     * @param descriptors non-null descriptors aligned with {@code valueIds}
     * @param valueStates non-null explicit validated states aligned with {@code valueIds}
     * @param graphProgram non-null versioned typed node table in partition order
     * @param feedValueIds non-null unique boundary inputs in stable feed order
     * @param feedValueIndices non-null value indices aligned with feeds
     * @param targetValueIds non-null unique boundary outputs in stable target order
     * @param targetValueIndices non-null value indices aligned with targets
     * @param declarations non-null exact feed-then-target buffer declarations
     * @param feedSplats non-null optional exact typed splats aligned with feeds
     * @param feedSplatSources non-null source-owner facts aligned with feeds; a true entry requires
     *     a present splat
     * @param addressWorkspace non-null optional workspace, absent only for custom singleton NEG
     * @param feedRequiredBytes non-null required logical byte extents aligned with feeds
     * @param targetRequiredBytes non-null required logical byte extents aligned with targets
     * @throws NullPointerException if a required reference or list element is {@code null}
     * @throws IllegalArgumentException if aligned cardinalities disagree
     */
    MetalNegPreparationPlan(
            NumericalProfile numericalProfile,
            PlannedPartition partition,
            PartitionDag partitionDag,
            MetalDeviceContext context,
            MetalPreparedRoute route,
            List<ValueId> valueIds,
            List<TensorDescriptor> descriptors,
            List<MetalMpsGraphProgram.ValueState> valueStates,
            MetalMpsGraphProgram graphProgram,
            List<ValueId> feedValueIds,
            int[] feedValueIndices,
            List<ValueId> targetValueIds,
            int[] targetValueIndices,
            List<ValueId> internalValueIds,
            int[] internalValueIndices,
            long[] internalRequiredBytes,
            List<PreparationResourceRequirement.Buffer> declarations,
            List<Optional<ScalarValue>> feedSplats,
            boolean[] feedSplatSources,
            Optional<PreparationResourceRequirement.Workspace> addressWorkspace,
            long[] feedRequiredBytes,
            long[] targetRequiredBytes) {
        this(
                numericalProfile,
                partition,
                partitionDag,
                context,
                route,
                valueIds,
                descriptors,
                valueStates,
                graphProgram,
                feedValueIds,
                feedValueIndices,
                targetValueIds,
                targetValueIndices,
                internalValueIds,
                internalValueIndices,
                internalRequiredBytes,
                declarations,
                feedSplats,
                feedSplatSources,
                addressWorkspace,
                feedRequiredBytes,
                targetRequiredBytes,
                null);
    }

    MetalNegPreparationPlan(
            NumericalProfile numericalProfile,
            PlannedPartition partition,
            PartitionDag partitionDag,
            MetalDeviceContext context,
            MetalPreparedRoute route,
            List<ValueId> valueIds,
            List<TensorDescriptor> descriptors,
            List<MetalMpsGraphProgram.ValueState> valueStates,
            MetalMpsGraphProgram graphProgram,
            List<ValueId> feedValueIds,
            int[] feedValueIndices,
            List<ValueId> targetValueIds,
            int[] targetValueIndices,
            List<ValueId> internalValueIds,
            int[] internalValueIndices,
            long[] internalRequiredBytes,
            List<PreparationResourceRequirement.Buffer> declarations,
            List<Optional<ScalarValue>> feedSplats,
            boolean[] feedSplatSources,
            Optional<PreparationResourceRequirement.Workspace> addressWorkspace,
            long[] feedRequiredBytes,
            long[] targetRequiredBytes,
            MetalTraceProducer.PreparedUnit traceUnit) {
        this.partition = Objects.requireNonNull(partition, "partition");
        this.numericalProfile =
                Objects.requireNonNull(numericalProfile, "numericalProfile");
        this.partitionDag = Objects.requireNonNull(partitionDag, "partitionDag");
        if (this.partitionDag.partition() != this.partition) {
            throw new IllegalArgumentException(
                    "Metal NEG partition DAG must retain the analyzed partition");
        }
        this.context = Objects.requireNonNull(context, "context");
        this.route = Objects.requireNonNull(route, "route");
        this.valueIds = List.copyOf(valueIds);
        this.descriptors = List.copyOf(descriptors);
        this.valueStates = List.copyOf(valueStates);
        if (this.descriptors.size() != this.valueStates.size()) {
            throw new IllegalArgumentException(
                    "Metal descriptor and value-state counts must match");
        }
        var encodedValues = new java.util.ArrayList<
                MetalMpsGraphProgram.ValueDescriptor>(this.descriptors.size());
        for (int value = 0; value < this.descriptors.size(); value++) {
            encodedValues.add(MetalMpsGraphProgram.ValueDescriptor.from(
                    this.descriptors.get(value), this.valueStates.get(value)));
        }
        this.programValueDescriptors = List.copyOf(encodedValues);
        this.graphProgram = Objects.requireNonNull(graphProgram, "graphProgram");
        this.feedValueIds = List.copyOf(feedValueIds);
        this.feedValueIndices = feedValueIndices.clone();
        this.targetValueIds = List.copyOf(targetValueIds);
        this.targetValueIndices = targetValueIndices.clone();
        this.internalValueIds = List.copyOf(internalValueIds);
        this.internalValueIndices = internalValueIndices.clone();
        this.internalRequiredBytes = internalRequiredBytes.clone();
        this.declarations = List.copyOf(declarations);
        this.feedSplats = List.copyOf(feedSplats);
        this.feedSplatSources = feedSplatSources.clone();
        this.addressWorkspace = Objects.requireNonNull(addressWorkspace, "addressWorkspace");
        this.feedRequiredBytes = feedRequiredBytes.clone();
        this.targetRequiredBytes = targetRequiredBytes.clone();
        this.traceUnit = traceUnit;
        if (this.valueIds.size() != this.descriptors.size()
                || this.valueIds.size() != this.valueStates.size()
                || this.valueStates.contains(MetalMpsGraphProgram.ValueState.UNAVAILABLE)
                || this.graphProgram.nodes().size() != partitionDag.nodes().size()
                || this.feedValueIds.size() != this.feedRequiredBytes.length
                || this.feedValueIds.size() != this.feedSplats.size()
                || this.feedValueIds.size() != this.feedSplatSources.length
                || this.targetValueIds.size() != this.targetValueIndices.length
                || this.targetValueIds.size() != this.targetRequiredBytes.length
                || this.internalValueIds.size() != this.internalValueIndices.length
                || this.internalValueIds.size() != this.internalRequiredBytes.length
                || this.declarations.size() != this.feedValueIds.size()
                        + this.targetValueIds.size() + this.internalValueIds.size()) {
            throw new IllegalArgumentException("Metal NEG preparation-plan cardinalities disagree");
        }
        for (int index = 0; index < this.feedSplatSources.length; index++) {
            if (this.feedSplatSources[index] && this.feedSplats.get(index).isEmpty()) {
                throw new IllegalArgumentException(
                        "Metal source splat requires an exact scalar value");
            }
        }
        boolean containsCustomOperation = this.graphProgram.nodes().stream()
                .anyMatch(node -> node.kind().isCustomProgramOperation()
                        || usesCustomMatmul(node, this.descriptors));
        boolean containsCustomOnlyOperation = this.graphProgram.nodes().stream()
                .anyMatch(node -> {
                    int wire = node.kind().wireIdentity();
                    return wire >= 20 && wire <= 34
                            || usesCustomMatmul(node, this.descriptors);
                });
        if ((this.route == MetalPreparedRoute.CUSTOM_SINGLE_NEG
                        && (partitionDag.nodes().size() != 1
                                || this.graphProgram.nodes().getFirst().kind()
                                        != MetalMpsGraphProgram.NodeKind.NEG
                                || this.feedValueIds.size() != 1
                                || this.targetValueIds.size() != 1
                                || !this.internalValueIds.isEmpty()
                                || this.addressWorkspace.isPresent()))
                || (this.route == MetalPreparedRoute.CUSTOM_PROGRAM
                        && (!containsCustomOperation || this.addressWorkspace.isEmpty()))
                || (this.route == MetalPreparedRoute.MPSGRAPH
                        && (containsCustomOnlyOperation
                                || !this.internalValueIds.isEmpty()
                                || this.addressWorkspace.isEmpty()))) {
            throw new IllegalArgumentException("Metal route and workspace facts disagree");
        }
        if (this.route == MetalPreparedRoute.CUSTOM_PROGRAM) {
            boolean[] covered = new boolean[this.valueIds.size()];
            cover(covered, this.feedValueIndices);
            cover(covered, this.targetValueIndices);
            cover(covered, this.internalValueIndices);
            for (boolean present : covered) {
                if (!present) {
                    throw new IllegalArgumentException(
                            "Metal custom program must materialize every stable value");
                }
            }
        }
    }
    private static boolean usesCustomMatmul(
            MetalMpsGraphProgram.Node node, List<TensorDescriptor> descriptors) {
        if (node.kind() != MetalMpsGraphProgram.NodeKind.MATMUL) return false;
        TensorDescriptor left = descriptors.get(node.firstInputIndex());
        TensorDescriptor right = descriptors.get(node.secondInputIndex());
        TensorDescriptor output = descriptors.get(node.outputIndex());
        return left.dataType() != DataType.FLOAT32
                || right.dataType() != DataType.FLOAT32
                || output.dataType() != DataType.FLOAT32
                || left.shape().rank() != 2
                || right.shape().rank() != 2
                || output.shape().rank() != 2;
    }

    /** @return exact immutable graph-wide numerical-profile identity */
    NumericalProfile numericalProfile() { return numericalProfile; }

    PlannedPartition partition() { return partition; }
    PartitionDag partitionDag() { return partitionDag; }
    MetalDeviceContext context() { return context; }
    /** @return the deterministic backend-private route selected before shared declarations */
    MetalPreparedRoute route() { return route; }
    /** @return immutable trace facts, or {@code null} on the no-trace/disabled path */
    MetalTraceProducer.PreparedUnit traceUnit() { return traceUnit; }
    List<ValueId> valueIds() { return valueIds; }
    List<TensorDescriptor> descriptors() { return descriptors; }
    List<MetalMpsGraphProgram.ValueDescriptor> programValueDescriptors() {
        return programValueDescriptors;
    }
    List<MetalMpsGraphProgram.ValueState> valueStates() { return valueStates; }
    MetalMpsGraphProgram graphProgram() { return graphProgram; }
    List<ValueId> feedValueIds() { return feedValueIds; }
    int[] feedValueIndices() { return feedValueIndices.clone(); }
    List<ValueId> targetValueIds() { return targetValueIds; }
    int[] targetValueIndices() { return targetValueIndices.clone(); }
    List<ValueId> internalValueIds() { return internalValueIds; }
    int[] internalValueIndices() { return internalValueIndices.clone(); }
    long[] internalRequiredBytes() { return internalRequiredBytes.clone(); }
    long[] materializedValueRequiredBytes() {
        if (route != MetalPreparedRoute.CUSTOM_PROGRAM) {
            throw new IllegalStateException("Metal plan is not a custom program");
        }
        long[] bytes = new long[valueIds.size()];
        place(bytes, feedValueIndices, feedRequiredBytes);
        place(bytes, targetValueIndices, targetRequiredBytes);
        place(bytes, internalValueIndices, internalRequiredBytes);
        return bytes;
    }
    List<PreparationResourceRequirement.Buffer> declarations() { return declarations; }
    List<Optional<ScalarValue>> feedSplats() { return feedSplats; }
    boolean[] feedSplatSources() { return feedSplatSources.clone(); }
    Optional<PreparationResourceRequirement.Workspace> addressWorkspace() {
        return addressWorkspace;
    }
    long[] feedRequiredBytes() { return feedRequiredBytes.clone(); }
    long[] targetRequiredBytes() { return targetRequiredBytes.clone(); }


    Optional<MetalMpsGraphProgram.NodeKind> publicationProducerKind(
            int targetPosition,
            ValueId valueId,
            TensorDescriptor descriptor,
            long byteSize) {
        Objects.requireNonNull(valueId, "valueId");
        Objects.requireNonNull(descriptor, "descriptor");
        if ((route != MetalPreparedRoute.MPSGRAPH && route != MetalPreparedRoute.CUSTOM_PROGRAM)
                || targetPosition < 0
                || targetPosition >= targetValueIds.size()
                || !targetValueIds.get(targetPosition).equals(valueId)
                || targetRequiredBytes[targetPosition] != byteSize) {
            return Optional.empty();
        }
        int value = targetValueIndices[targetPosition];
        MetalMpsGraphProgram.ValueState state = valueStates.get(value);
        if (!descriptors.get(value).equals(descriptor)
                || (state != MetalMpsGraphProgram.ValueState.AFFINE_VIEW
                        && state != MetalMpsGraphProgram.ValueState.MATERIALIZED_LAYOUT)) {
            return Optional.empty();
        }
        for (MetalMpsGraphProgram.Node node : graphProgram.nodes()) {
            if (node.outputIndex() == value) {
                return node.kind().publishesAuthenticatedLayout()
                        ? Optional.of(node.kind()) : Optional.empty();
            }
        }
        return Optional.empty();
    }

    private static void place(long[] target, int[] indices, long[] values) {
        for (int position = 0; position < indices.length; position++) {
            target[indices[position]] = values[position];
        }
    }

    private static void cover(boolean[] covered, int[] indices) {
        for (int index : indices) {
            if (index < 0 || index >= covered.length || covered[index]) {
                throw new IllegalArgumentException(
                        "Metal materialized value indices are malformed");
            }
            covered[index] = true;
        }
    }

    /**
     * Returns the custom singleton's checked positive element count.
     *
     * @return a value in the inclusive range {@code 1..UINT32_MAX}
     * @throws IllegalStateException if this plan selected MPSGraph
     */
    long customElementCount() {
        if (route != MetalPreparedRoute.CUSTOM_SINGLE_NEG) {
            throw new IllegalStateException("Metal NEG plan did not select the custom route");
        }
        return feedRequiredBytes[0] / Float.BYTES;
    }
}
