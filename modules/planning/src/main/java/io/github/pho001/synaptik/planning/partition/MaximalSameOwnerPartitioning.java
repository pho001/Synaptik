package io.github.pho001.synaptik.planning.partition;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.NodeId;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Generates maximal consecutive same-owner partitions for one compiled graph.
 *
 * <p>This stateless package-private operation consumes the graph's stored topological node order
 * and one complete node-to-owner assignment. Consecutive nodes join while their
 * {@link BackendId} owners are equal; graph edges, phases, value boundaries, and map iteration
 * order do not redefine adjacency. The result records backend ownership only and performs no
 * scoring, lowering, routing, memory planning, preparation, or execution.</p>
 */
final class MaximalSameOwnerPartitioning {
    /**
     * Prevents instances because partition generation retains no state between calls.
     */
    private MaximalSameOwnerPartitioning() {}

    /**
     * Groups the graph's stored node sequence into maximal consecutive equal-owner runs.
     *
     * <p>The supplied map must contain exactly one association for each graph node and no unknown
     * key. Keys associate by {@link NodeId#equals(Object)}. Null keys and unknown keys are
     * validated before graph coverage; complete coverage is validated before owners. Unknown
     * keys are checked in ascending numeric node-ID order, while missing associations and null
     * owners are checked in stored graph order. All validation completes before the first output
     * partition is constructed.</p>
     *
     * <p>Each output stores the exact {@code NodeId} references from
     * {@link CompiledGraphModel#nodes()} and the exact owner reference mapped for the run's first
     * node. Equal but non-identical later owner references join without replacing that reference.
     * The immutable outer result and each partition's immutable membership preserve graph order.
     * A valid zero-node graph requires an empty map and produces an immutable empty list.</p>
     *
     * <p>Graph inputs and outputs remain values rather than synthetic partition members. A
     * multi-output producer remains one indivisible node, and fan-out, merges, repeated inputs,
     * output publication, cross-owner values, and phase changes do not independently split a
     * same-owner run. Boundary, transfer, materialization, memory, device, route, kernel,
     * preparation, and execution decisions are outside this operation.</p>
     *
     * @param graph the non-null immutable compiled graph whose stored topological node order is
     *     the sole ordering and adjacency source
     * @param ownershipByNodeId the non-null complete node-to-owner association; keys and values
     *     must be non-null, keys must name exactly the graph nodes by equality, and the map is
     *     inspected but not retained or mutated
     * @return a non-null immutable ordered list of maximal, non-empty partition recipes; empty
     *     only for a valid zero-node graph
     * @throws NullPointerException if {@code graph} is {@code null}; the message is {@code graph}
     * @throws NullPointerException if {@code ownershipByNodeId} is {@code null}; the message is
     *     {@code ownershipByNodeId}
     * @throws NullPointerException if the map contains a null key; the message is
     *     {@code ownershipByNodeId contains null key}
     * @throws NullPointerException if a graph node's associated owner is {@code null}; the
     *     message is {@code ownershipByNodeId[NodeId[value=n]]}, with nodes checked in stored
     *     graph order after complete key and coverage validation
     * @throws IllegalArgumentException if the map contains a key that does not name a graph node;
     *     the message is {@code ownershipByNodeId contains unknown NodeId[value=n]}, with keys
     *     checked in ascending numeric order
     * @throws IllegalArgumentException if the map has no equal key for a graph node; the message
     *     is {@code ownershipByNodeId missing NodeId[value=n]}, with nodes checked in stored graph
     *     order before any owner value is validated
     */
    static List<PlannedPartition> partition(
            CompiledGraphModel graph,
            Map<NodeId, BackendId> ownershipByNodeId) {
        Objects.requireNonNull(graph, "graph");
        Objects.requireNonNull(ownershipByNodeId, "ownershipByNodeId");

        for (NodeId nodeId : ownershipByNodeId.keySet()) {
            if (nodeId == null) {
                throw new NullPointerException("ownershipByNodeId contains null key");
            }
        }

        List<CompiledNode> nodes = graph.nodes();
        Set<NodeId> graphNodeIds = new HashSet<>();
        for (CompiledNode node : nodes) {
            graphNodeIds.add(node.id());
        }

        List<NodeId> sortedOwnershipKeys = new ArrayList<>(ownershipByNodeId.keySet());
        sortedOwnershipKeys.sort(Comparator.comparingLong(NodeId::value));
        for (NodeId nodeId : sortedOwnershipKeys) {
            if (!graphNodeIds.contains(nodeId)) {
                throw new IllegalArgumentException(
                        "ownershipByNodeId contains unknown " + nodeId);
            }
        }

        for (CompiledNode node : nodes) {
            if (!ownershipByNodeId.containsKey(node.id())) {
                throw new IllegalArgumentException(
                        "ownershipByNodeId missing " + node.id());
            }
        }

        List<BackendId> owners = new ArrayList<>(nodes.size());
        for (CompiledNode node : nodes) {
            owners.add(Objects.requireNonNull(
                    ownershipByNodeId.get(node.id()),
                    "ownershipByNodeId[" + node.id() + "]"));
        }

        if (nodes.isEmpty()) {
            return List.of();
        }

        List<PlannedPartition> partitions = new ArrayList<>();
        BackendId currentOwner = owners.getFirst();
        List<NodeId> currentNodeIds = new ArrayList<>();
        currentNodeIds.add(nodes.getFirst().id());
        for (int index = 1; index < nodes.size(); index++) {
            BackendId owner = owners.get(index);
            if (!currentOwner.equals(owner)) {
                partitions.add(new PlannedPartition(currentOwner, currentNodeIds));
                currentOwner = owner;
                currentNodeIds = new ArrayList<>();
            }
            currentNodeIds.add(nodes.get(index).id());
        }
        partitions.add(new PlannedPartition(currentOwner, currentNodeIds));
        return List.copyOf(partitions);
    }
}
