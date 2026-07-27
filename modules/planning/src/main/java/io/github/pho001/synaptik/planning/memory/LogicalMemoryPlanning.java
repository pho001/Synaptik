package io.github.pho001.synaptik.planning.memory;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Derives logical value requirements from one closed graph and its complete partition recipes.
 *
 * <p>This public stateless package-owned seam is invoked by the package-private compiler
 * orchestrator after maximal partition generation. It validates exact graph-order partition
 * coverage and maximal adjacent owner runs before deriving producer, consumer, and graph-output
 * facts. It performs no graph repair, physical memory planning, publication binding, transfer
 * selection, preparation, or execution, and it is not a graph-wide Planning workflow.</p>
 */
public final class LogicalMemoryPlanning {
    /** Prevents instances because logical-memory derivation retains no state. */
    private LogicalMemoryPlanning() {}

    /**
     * Builds one logical requirement for every graph value in stored value order.
     *
     * <p>The supplied partitions must cover every graph node exactly once, in the graph's stored
     * node order, and adjacent partitions must have unequal owners. Node identities associate by
     * equality, while generated requirements retain exact graph value, descriptor, and supplied
     * partition references. Each supplied partition that consumes a value contributes at most
     * one entry, and consumer entries follow supplied partition order.</p>
     *
     * @param graph non-null structurally closed immutable graph supplying all values, nodes,
     *     descriptors, relationships, and graph-output boundaries
     * @param partitions non-null ordered complete maximal partition recipes; elements must be
     *     non-null, contain every graph node exactly once in graph order, and have unequal owners
     *     at each adjacent boundary; the list container is not retained
     * @return non-null immutable plan with one requirement per graph value in graph-value order;
     *     producer and consumer entries retain exact supplied partition-element references
     * @throws NullPointerException if {@code graph}, {@code partitions}, or a partition element is
     *     {@code null}, with the exact indexed messages described by the package contract
     * @throws IllegalArgumentException if a partition names an unknown or duplicate node, misses
     *     a graph node, violates graph order, or is adjacent to an equal-owner partition
     */
    public static LogicalMemoryPlan plan(
            CompiledGraphModel graph,
            List<PlannedPartition> partitions) {
        Objects.requireNonNull(graph, "graph");
        Objects.requireNonNull(partitions, "partitions");

        for (int index = 0; index < partitions.size(); index++) {
            Objects.requireNonNull(partitions.get(index), "partitions[" + index + "]");
        }

        List<CompiledNode> graphNodes = graph.nodes();
        Set<NodeId> graphNodeIds = new HashSet<>();
        Map<NodeId, CompiledNode> graphNodesById = new HashMap<>();
        for (CompiledNode node : graphNodes) {
            graphNodeIds.add(node.id());
            graphNodesById.put(node.id(), node);
        }

        Set<NodeId> observedNodeIds = new HashSet<>();
        for (int partitionIndex = 0; partitionIndex < partitions.size(); partitionIndex++) {
            List<NodeId> nodeIds = partitions.get(partitionIndex).nodeIds();
            for (int nodeIndex = 0; nodeIndex < nodeIds.size(); nodeIndex++) {
                NodeId nodeId = nodeIds.get(nodeIndex);
                if (!graphNodeIds.contains(nodeId)) {
                    throw new IllegalArgumentException(
                            "partitions[" + partitionIndex + "].nodeIds[" + nodeIndex
                                    + "] references unknown " + nodeId);
                }
                if (!observedNodeIds.add(nodeId)) {
                    throw new IllegalArgumentException(
                            "partitions[" + partitionIndex + "].nodeIds[" + nodeIndex
                                    + "] duplicates " + nodeId);
                }
            }
        }

        for (CompiledNode node : graphNodes) {
            if (!observedNodeIds.contains(node.id())) {
                throw new IllegalArgumentException("partitions missing " + node.id());
            }
        }

        int graphNodeIndex = 0;
        for (int partitionIndex = 0; partitionIndex < partitions.size(); partitionIndex++) {
            List<NodeId> nodeIds = partitions.get(partitionIndex).nodeIds();
            for (int nodeIndex = 0; nodeIndex < nodeIds.size(); nodeIndex++) {
                NodeId expected = graphNodes.get(graphNodeIndex).id();
                if (!nodeIds.get(nodeIndex).equals(expected)) {
                    throw new IllegalArgumentException(
                            "partitions[" + partitionIndex + "].nodeIds[" + nodeIndex
                                    + "] is out of graph order: expected " + expected);
                }
                graphNodeIndex++;
            }
        }

        for (int index = 1; index < partitions.size(); index++) {
            if (partitions.get(index - 1).owner().equals(partitions.get(index).owner())) {
                throw new IllegalArgumentException(
                        "partitions[" + index + "].owner equals previous owner "
                                + partitions.get(index).owner());
            }
        }

        Map<NodeId, PlannedPartition> partitionByNodeId = new HashMap<>();
        for (PlannedPartition partition : partitions) {
            for (NodeId nodeId : partition.nodeIds()) {
                partitionByNodeId.put(nodeId, partition);
            }
        }

        Map<ValueId, PlannedPartition> producerByValueId = new HashMap<>();
        for (CompiledNode node : graphNodes) {
            PlannedPartition producerPartition = partitionByNodeId.get(node.id());
            for (ValueId output : node.outputs()) {
                producerByValueId.put(output, producerPartition);
            }
        }

        Set<ValueId> graphOutputIds = new HashSet<>(graph.outputs());
        List<LogicalMemoryRequirement> requirements = new ArrayList<>(graph.values().size());
        for (GraphValue value : graph.values()) {
            List<PlannedPartition> consumerPartitions = new ArrayList<>();
            for (PlannedPartition partition : partitions) {
                boolean consumed = false;
                for (NodeId nodeId : partition.nodeIds()) {
                    if (graphNodesById.get(nodeId).inputs().contains(value.id())) {
                        consumed = true;
                        break;
                    }
                }
                if (consumed) {
                    consumerPartitions.add(partition);
                }
            }
            requirements.add(new LogicalMemoryRequirement(
                    value.id(),
                    value.descriptor(),
                    Optional.ofNullable(producerByValueId.get(value.id())),
                    consumerPartitions,
                    graphOutputIds.contains(value.id())));
        }
        return new LogicalMemoryPlan(requirements);
    }
}
