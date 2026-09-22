package io.github.pho001.synaptik.backend.metal.internal;

import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Retains the immutable, shape-specialized lowering and route facts for one whole NEG partition.
 *
 * <p>Value indices, node arrays, feeds, targets, and declarations are already in their stable ABI
 * order. The route is either the safe heuristic or a freshly authenticated session-compatible
 * decision, and is fixed before this plan's declarations escape analysis. The plan contains no
 * assigned slot, tuning value, native executable, physical buffer, or per-run state. The address
 * workspace is present only for MPSGraph. Primitive arrays are privately snapshotted and copied
 * when marshalled.</p>
 */
final class MetalNegPreparationPlan implements BackendPreparationPlan {
    /** Closed private implementation choice made during analysis. */
    enum Route {
        /** Exact singleton NEG with one feed, one target, and {@code 1..UINT32_MAX} elements. */
        CUSTOM_SINGLE_NEG,

        /** Every other partition in the unchanged supported Metal NEG capability domain. */
        MPSGRAPH
    }

    private final PlannedPartition partition;
    private final PartitionDag partitionDag;
    private final MetalDeviceContext context;
    private final Route route;
    private final List<ValueId> valueIds;
    private final List<TensorDescriptor> descriptors;
    private final int[] valueRanks;
    private final long[] valueDimensions;
    private final int[] nodeInputValueIndices;
    private final int[] nodeOutputValueIndices;
    private final List<ValueId> feedValueIds;
    private final int[] feedValueIndices;
    private final List<ValueId> targetValueIds;
    private final int[] targetValueIndices;
    private final List<PreparationResourceRequirement.Buffer> declarations;
    private final List<Optional<ScalarValue>> feedSplats;
    private final Optional<PreparationResourceRequirement.Workspace> addressWorkspace;
    private final long[] feedRequiredBytes;
    private final long[] targetRequiredBytes;

    /**
     * Creates one completely validated analysis result snapshot.
     *
     * @param partition exact non-null planned partition analyzed to produce this plan
     * @param partitionDag exact non-null partition topology retaining {@code partition}
     * @param context exact non-null Metal device context retained by identity
     * @param route non-null closed implementation route selected during analysis
     * @param valueIds non-null stable indexed value identities
     * @param descriptors non-null descriptors aligned with {@code valueIds}
     * @param valueRanks non-null ranks aligned with values
     * @param valueDimensions non-null row-major value-count by sixteen dimension table
     * @param nodeInputValueIndices non-null node input indices in partition order
     * @param nodeOutputValueIndices non-null node output indices in partition order
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
            PlannedPartition partition,
            PartitionDag partitionDag,
            MetalDeviceContext context,
            Route route,
            List<ValueId> valueIds,
            List<TensorDescriptor> descriptors,
            int[] valueRanks,
            long[] valueDimensions,
            int[] nodeInputValueIndices,
            int[] nodeOutputValueIndices,
            List<ValueId> feedValueIds,
            int[] feedValueIndices,
            List<ValueId> targetValueIds,
            int[] targetValueIndices,
            List<PreparationResourceRequirement.Buffer> declarations,
            List<Optional<ScalarValue>> feedSplats,
            Optional<PreparationResourceRequirement.Workspace> addressWorkspace,
            long[] feedRequiredBytes,
            long[] targetRequiredBytes) {
        this.partition = Objects.requireNonNull(partition, "partition");
        this.partitionDag = Objects.requireNonNull(partitionDag, "partitionDag");
        if (this.partitionDag.partition() != this.partition) {
            throw new IllegalArgumentException(
                    "Metal NEG partition DAG must retain the analyzed partition");
        }
        this.context = Objects.requireNonNull(context, "context");
        this.route = Objects.requireNonNull(route, "route");
        this.valueIds = List.copyOf(valueIds);
        this.descriptors = List.copyOf(descriptors);
        this.valueRanks = valueRanks.clone();
        this.valueDimensions = valueDimensions.clone();
        this.nodeInputValueIndices = nodeInputValueIndices.clone();
        this.nodeOutputValueIndices = nodeOutputValueIndices.clone();
        this.feedValueIds = List.copyOf(feedValueIds);
        this.feedValueIndices = feedValueIndices.clone();
        this.targetValueIds = List.copyOf(targetValueIds);
        this.targetValueIndices = targetValueIndices.clone();
        this.declarations = List.copyOf(declarations);
        this.feedSplats = List.copyOf(feedSplats);
        this.addressWorkspace = Objects.requireNonNull(addressWorkspace, "addressWorkspace");
        this.feedRequiredBytes = feedRequiredBytes.clone();
        this.targetRequiredBytes = targetRequiredBytes.clone();
        if (this.valueIds.size() != this.descriptors.size()
                || this.valueIds.size() != this.valueRanks.length
                || this.valueDimensions.length != this.valueIds.size() * 16
                || this.nodeInputValueIndices.length != partitionDag.nodes().size()
                || this.nodeOutputValueIndices.length != partitionDag.nodes().size()
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
                                || this.feedValueIds.size() != 1
                                || this.targetValueIds.size() != 1
                                || this.addressWorkspace.isPresent()))
                || (this.route == Route.MPSGRAPH && this.addressWorkspace.isEmpty())) {
            throw new IllegalArgumentException("Metal NEG route and workspace facts disagree");
        }
    }

    PlannedPartition partition() { return partition; }
    PartitionDag partitionDag() { return partitionDag; }
    MetalDeviceContext context() { return context; }
    /** @return the deterministic backend-private route selected before shared declarations */
    Route route() { return route; }
    List<ValueId> valueIds() { return valueIds; }
    List<TensorDescriptor> descriptors() { return descriptors; }
    int[] valueRanks() { return valueRanks.clone(); }
    long[] valueDimensions() { return valueDimensions.clone(); }
    int[] nodeInputValueIndices() { return nodeInputValueIndices.clone(); }
    int[] nodeOutputValueIndices() { return nodeOutputValueIndices.clone(); }
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
