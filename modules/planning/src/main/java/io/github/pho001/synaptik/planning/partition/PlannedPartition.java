package io.github.pho001.synaptik.planning.partition;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.graph.NodeId;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * Records one immutable compile-time recipe for a backend-owned graph region.
 *
 * <p>A planned partition associates one exact backend ownership identity with a non-empty,
 * ordered sequence of graph-node occurrences. The node order is supplied by the owning compiled
 * graph and is preserved by this value. The recipe does not contain graph values, boundary
 * edges, transfers, logical or physical memory, a selected device, lowering, kernels, executable
 * state, or runtime state.</p>
 *
 * <p>The node-ID container is an immutable membership snapshot. Record-generated equality and
 * hashing compare the owner and ordered node IDs by value. The generated diagnostic
 * {@link #toString()} is not a serialization, preparation, or execution format.</p>
 *
 * @param owner non-null backend identity that owns every listed node; the exact immutable
 *     reference is retained
 * @param nodeIds non-null, non-empty ordered node identities; elements must be non-null and
 *     unique by equality, list membership is copied, and exact element references are retained
 */
public record PlannedPartition(BackendId owner, List<NodeId> nodeIds) {
    /**
     * Creates an immutable owner-plus-node-ID partition recipe.
     *
     * <p>The owner and list references are validated first. Node IDs are then inspected in
     * encounter order for nulls and the first later duplicate before membership is copied with
     * {@link List#copyOf(java.util.Collection)}. The copy preserves the exact immutable element
     * references but does not promise container identity with the supplied list.</p>
     *
     * @param owner non-null backend ownership identity to retain exactly
     * @param nodeIds non-null, non-empty ordered node identities to snapshot; elements must be
     *     non-null and unique by equality
     * @throws NullPointerException if {@code owner} is {@code null}; the message is {@code owner}
     * @throws NullPointerException if {@code nodeIds} is {@code null}; the message is
     *     {@code nodeIds}
     * @throws NullPointerException if an element is {@code null}; the message is
     *     {@code nodeIds[index]} with its zero-based encounter index
     * @throws IllegalArgumentException if {@code nodeIds} is empty; the message is
     *     {@code nodeIds must not be empty}
     * @throws IllegalArgumentException if a later element repeats an earlier node identity; the
     *     message is {@code nodeIds[index] duplicates NodeId[value=n]} with the later zero-based
     *     encounter index and duplicate identity's diagnostic text
     */
    public PlannedPartition {
        Objects.requireNonNull(owner, "owner");
        Objects.requireNonNull(nodeIds, "nodeIds");
        if (nodeIds.isEmpty()) {
            throw new IllegalArgumentException("nodeIds must not be empty");
        }

        var observedNodeIds = new HashSet<NodeId>();
        for (int index = 0; index < nodeIds.size(); index++) {
            NodeId nodeId = Objects.requireNonNull(nodeIds.get(index), "nodeIds[" + index + "]");
            if (!observedNodeIds.add(nodeId)) {
                throw new IllegalArgumentException(
                        "nodeIds[" + index + "] duplicates " + nodeId);
            }
        }
        nodeIds = List.copyOf(nodeIds);
    }

    /**
     * Returns the backend identity that owns every node in this recipe.
     *
     * @return the exact non-null immutable {@link BackendId} reference supplied at construction;
     *     the identity names ownership only and is not a live backend service or selected device
     */
    @Override
    public BackendId owner() {
        return owner;
    }

    /**
     * Returns the immutable ordered snapshot of graph-node identities in this partition.
     *
     * <p>The result is non-empty, contains no null or duplicate identity, preserves encounter
     * order, and cannot be mutated. Its element references are those supplied at construction;
     * no identity relationship with the original list container is promised.</p>
     *
     * @return the non-null, non-empty immutable ordered snapshot of unique node identities
     */
    @Override
    public List<NodeId> nodeIds() {
        return nodeIds;
    }
}
