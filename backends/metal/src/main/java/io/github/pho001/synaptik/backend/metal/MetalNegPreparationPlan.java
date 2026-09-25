package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
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
 * <p>The retained numerical profile closes the operation domain: both profiles admit NEG, ABS,
 * affine transforms, and CONTIGUOUS, while accelerator additionally admits tensor ADD, SUB, MUL,
 * DIV, the admitted reductions, and rank-two MATMUL. An affine MATMUL input must be the exact local
 * rank-two transpose authenticated on that consuming edge; the same affine value may otherwise be
 * consumed or published normally. Value indices, explicit canonical/affine-view states, typed
 * MPSGraph nodes, feeds, targets, and declarations are already in their stable ABI order. The
 * route is either the safe heuristic or a freshly authenticated session-compatible decision, and
 * is fixed before this plan's declarations escape analysis. The plan contains no assigned slot,
 * tuning value, native executable, physical buffer, or per-run state. Affine targets retain exact
 * logical view descriptors alongside their full dense represented-order byte extents. The address
 * workspace is present only for MPSGraph. Primitive arrays are privately snapshotted and copied
 * when marshalled.</p>
 */
final class MetalNegPreparationPlan implements BackendPreparationPlan {
    /** Closed private implementation choice made during analysis. */
    enum Route {
        /** Exact singleton NEG with one feed, one target, and {@code 1..UINT32_MAX} elements. */
        CUSTOM_SINGLE_NEG,

        /** Every supported partition except an eligible custom singleton NEG. */
        MPSGRAPH
    }

    private final NumericalProfile numericalProfile;
    private final PlannedPartition partition;
    private final PartitionDag partitionDag;
    private final MetalDeviceContext context;
    private final Route route;
    private final List<ValueId> valueIds;
    private final List<TensorDescriptor> descriptors;
    private final List<MetalMpsGraphProgram.ValueState> valueStates;
    private final int[] valueRanks;
    private final long[] valueDimensions;
    private final MetalMpsGraphProgram graphProgram;
    private final List<ValueId> feedValueIds;
    private final int[] feedValueIndices;
    private final List<ValueId> targetValueIds;
    private final int[] targetValueIndices;
    private final List<PreparationResourceRequirement.Buffer> declarations;
    private final List<Optional<ScalarValue>> feedSplats;
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
     * @param route non-null closed implementation route selected during analysis
     * @param valueIds non-null stable indexed value identities
     * @param descriptors non-null descriptors aligned with {@code valueIds}
     * @param valueStates non-null explicit validated states aligned with {@code valueIds}
     * @param valueRanks non-null ranks aligned with values
     * @param valueDimensions non-null row-major value-count by sixteen dimension table
     * @param graphProgram non-null versioned typed node table in partition order
     * @param feedValueIds non-null unique boundary inputs in stable feed order
     * @param feedValueIndices non-null value indices aligned with feeds
     * @param targetValueIds non-null unique boundary outputs in stable target order
     * @param targetValueIndices non-null value indices aligned with targets
     * @param declarations non-null exact feed-then-target buffer declarations
     * @param feedSplats non-null optional FLOAT32 splats aligned with feeds
     * @param addressWorkspace non-null optional workspace, present exactly for MPSGraph
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
            Route route,
            List<ValueId> valueIds,
            List<TensorDescriptor> descriptors,
            List<MetalMpsGraphProgram.ValueState> valueStates,
            int[] valueRanks,
            long[] valueDimensions,
            MetalMpsGraphProgram graphProgram,
            List<ValueId> feedValueIds,
            int[] feedValueIndices,
            List<ValueId> targetValueIds,
            int[] targetValueIndices,
            List<PreparationResourceRequirement.Buffer> declarations,
            List<Optional<ScalarValue>> feedSplats,
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
                valueRanks,
                valueDimensions,
                graphProgram,
                feedValueIds,
                feedValueIndices,
                targetValueIds,
                targetValueIndices,
                declarations,
                feedSplats,
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
            Route route,
            List<ValueId> valueIds,
            List<TensorDescriptor> descriptors,
            List<MetalMpsGraphProgram.ValueState> valueStates,
            int[] valueRanks,
            long[] valueDimensions,
            MetalMpsGraphProgram graphProgram,
            List<ValueId> feedValueIds,
            int[] feedValueIndices,
            List<ValueId> targetValueIds,
            int[] targetValueIndices,
            List<PreparationResourceRequirement.Buffer> declarations,
            List<Optional<ScalarValue>> feedSplats,
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
        this.valueRanks = valueRanks.clone();
        this.valueDimensions = valueDimensions.clone();
        this.graphProgram = Objects.requireNonNull(graphProgram, "graphProgram");
        this.feedValueIds = List.copyOf(feedValueIds);
        this.feedValueIndices = feedValueIndices.clone();
        this.targetValueIds = List.copyOf(targetValueIds);
        this.targetValueIndices = targetValueIndices.clone();
        this.declarations = List.copyOf(declarations);
        this.feedSplats = List.copyOf(feedSplats);
        this.addressWorkspace = Objects.requireNonNull(addressWorkspace, "addressWorkspace");
        this.feedRequiredBytes = feedRequiredBytes.clone();
        this.targetRequiredBytes = targetRequiredBytes.clone();
        this.traceUnit = traceUnit;
        if (this.valueIds.size() != this.descriptors.size()
                || this.valueIds.size() != this.valueStates.size()
                || this.valueIds.size() != this.valueRanks.length
                || this.valueDimensions.length != this.valueIds.size() * 16
                || this.valueStates.contains(MetalMpsGraphProgram.ValueState.UNAVAILABLE)
                || this.graphProgram.nodes().size() != partitionDag.nodes().size()
                || this.feedValueIds.size() != this.feedValueIndices.length
                || this.feedValueIds.size() != this.feedRequiredBytes.length
                || this.feedValueIds.size() != this.feedSplats.size()
                || this.targetValueIds.size() != this.targetValueIndices.length
                || this.targetValueIds.size() != this.targetRequiredBytes.length
                || this.declarations.size()
                        != this.feedValueIds.size() + this.targetValueIds.size()) {
            throw new IllegalArgumentException("Metal NEG preparation-plan cardinalities disagree");
        }
        if ((this.route == Route.CUSTOM_SINGLE_NEG
                        && (partitionDag.nodes().size() != 1
                                || this.graphProgram.nodes().getFirst().kind()
                                        != MetalMpsGraphProgram.NodeKind.NEG
                                || this.feedValueIds.size() != 1
                                || this.targetValueIds.size() != 1
                                || this.addressWorkspace.isPresent()))
                || (this.route == Route.MPSGRAPH && this.addressWorkspace.isEmpty())) {
            throw new IllegalArgumentException("Metal NEG route and workspace facts disagree");
        }
    }
    /** @return exact immutable graph-wide numerical-profile identity */
    NumericalProfile numericalProfile() { return numericalProfile; }

    PlannedPartition partition() { return partition; }
    PartitionDag partitionDag() { return partitionDag; }
    MetalDeviceContext context() { return context; }
    /** @return the deterministic backend-private route selected before shared declarations */
    Route route() { return route; }
    /** @return immutable trace facts, or {@code null} on the no-trace/disabled path */
    MetalTraceProducer.PreparedUnit traceUnit() { return traceUnit; }
    List<ValueId> valueIds() { return valueIds; }
    List<TensorDescriptor> descriptors() { return descriptors; }
    List<MetalMpsGraphProgram.ValueState> valueStates() { return valueStates; }
    int[] valueRanks() { return valueRanks.clone(); }
    long[] valueDimensions() { return valueDimensions.clone(); }
    MetalMpsGraphProgram graphProgram() { return graphProgram; }
    List<ValueId> feedValueIds() { return feedValueIds; }
    int[] feedValueIndices() { return feedValueIndices.clone(); }
    List<ValueId> targetValueIds() { return targetValueIds; }
    int[] targetValueIndices() { return targetValueIndices.clone(); }
    List<PreparationResourceRequirement.Buffer> declarations() { return declarations; }
    List<Optional<ScalarValue>> feedSplats() { return feedSplats; }
    Optional<PreparationResourceRequirement.Workspace> addressWorkspace() {
        return addressWorkspace;
    }
    long[] feedRequiredBytes() { return feedRequiredBytes.clone(); }
    long[] targetRequiredBytes() { return targetRequiredBytes.clone(); }


    Optional<MetalMpsGraphProgram.NodeKind> denseAffineProducerKind(
            int targetPosition,
            ValueId valueId,
            TensorDescriptor descriptor,
            long byteSize) {
        Objects.requireNonNull(valueId, "valueId");
        Objects.requireNonNull(descriptor, "descriptor");
        if (route != Route.MPSGRAPH
                || targetPosition < 0
                || targetPosition >= targetValueIds.size()
                || !targetValueIds.get(targetPosition).equals(valueId)
                || targetRequiredBytes[targetPosition] != byteSize) {
            return Optional.empty();
        }
        int value = targetValueIndices[targetPosition];
        if (!descriptors.get(value).equals(descriptor)
                || valueStates.get(value) != MetalMpsGraphProgram.ValueState.AFFINE_VIEW) {
            return Optional.empty();
        }
        for (MetalMpsGraphProgram.Node node : graphProgram.nodes()) {
            if (node.outputIndex() == value) {
                return node.kind().isAffine()
                        ? Optional.of(node.kind()) : Optional.empty();
            }
        }
        return Optional.empty();
    }

    /**
     * Returns the custom singleton's checked positive element count.
     *
     * @return a value in the inclusive range {@code 1..UINT32_MAX}
     * @throws IllegalStateException if this plan selected MPSGraph
     */
    long customElementCount() {
        if (route != Route.CUSTOM_SINGLE_NEG) {
            throw new IllegalStateException("Metal NEG plan did not select the custom route");
        }
        return feedRequiredBytes[0] / Float.BYTES;
    }
}
