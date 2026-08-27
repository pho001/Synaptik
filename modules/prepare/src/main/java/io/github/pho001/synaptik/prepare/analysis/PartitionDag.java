package io.github.pho001.synaptik.prepare.analysis;

import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Provides an immutable partition-local directed acyclic graph (DAG) for exactly one planned
 * partition.
 *
 * <p>The projection retains the partition's exact compiled-node references in stable topological
 * order and derives producer, consumer, edge, external-input-occurrence, and local-sink facts
 * from their ordered ports. It contains no complete compiled model, out-of-partition node,
 * backend policy, route, memory decision, publication decision, or execution schedule. An input
 * with no producer in this partition is an external input occurrence even when another partition
 * produces the value. Repeated input ports remain distinct consumer and edge occurrences, and
 * every output port of a multi-output node has its own producer occurrence.</p>
 *
 * <p>All returned lists and facts are immutable deterministic snapshots. Construction validates
 * the complete structure before an instance can reach backend analysis. Equality and hashing use
 * the exact planned partition and ordered node-list values; the diagnostic textual form is not a
 * serialization format.</p>
 */
public final class PartitionDag {
    private final PlannedPartition partition;
    private final List<CompiledNode> nodes;
    private final Map<NodeId, CompiledNode> nodesById;
    private final Map<ValueId, ProducerOccurrence> producersByValue;
    private final Map<ValueId, List<ConsumerOccurrence>> consumersByValue;
    private final List<Edge> edges;
    private final List<ConsumerOccurrence> externalInputs;
    private final List<CompiledNode> localSinks;

    /**
     * Constructs and validates one partition-local directed acyclic graph projection.
     *
     * @param partition non-null planned partition retained by exact reference
     * @param nodes non-null ordered node list to snapshot; elements must be non-null, have unique
     *     IDs, and match the partition's node IDs exactly in size, identity, and order
     * @throws NullPointerException if {@code partition}, {@code nodes}, a node, or a node port is
     *     {@code null}
     * @throws IllegalArgumentException if node identity/order disagrees with the partition, a
     *     node ID or produced value is duplicated, or an input is produced by its own or a later
     *     partition node
     */
    public PartitionDag(PlannedPartition partition, List<CompiledNode> nodes) {
        this.partition = Objects.requireNonNull(partition, "partition");
        Objects.requireNonNull(nodes, "nodes");

        var nodeIndex = new LinkedHashMap<NodeId, CompiledNode>();
        for (int index = 0; index < nodes.size(); index++) {
            CompiledNode node = Objects.requireNonNull(nodes.get(index), "nodes[" + index + "]");
            if (nodeIndex.putIfAbsent(node.id(), node) != null) {
                throw new IllegalArgumentException("nodes[" + index + "].id duplicates " + node.id());
            }
        }
        this.nodes = List.copyOf(nodes);

        if (this.nodes.size() != partition.nodeIds().size()) {
            throw new IllegalArgumentException(
                    "nodes size " + this.nodes.size() + " does not match partition nodeIds size "
                            + partition.nodeIds().size());
        }
        for (int index = 0; index < this.nodes.size(); index++) {
            NodeId expected = partition.nodeIds().get(index);
            NodeId actual = this.nodes.get(index).id();
            if (!actual.equals(expected)) {
                throw new IllegalArgumentException(
                        "nodes[" + index + "].id must equal partition.nodeIds[" + index
                                + "]: expected " + expected + " but was " + actual);
            }
        }
        nodesById = Collections.unmodifiableMap(nodeIndex);

        var producers = new LinkedHashMap<ValueId, ProducerOccurrence>();
        for (int nodePosition = 0; nodePosition < this.nodes.size(); nodePosition++) {
            CompiledNode node = this.nodes.get(nodePosition);
            for (int outputPosition = 0; outputPosition < node.outputs().size(); outputPosition++) {
                ValueId valueId = Objects.requireNonNull(
                        node.outputs().get(outputPosition),
                        "nodes[" + nodePosition + "].outputs[" + outputPosition + "]");
                ProducerOccurrence occurrence =
                        new ProducerOccurrence(valueId, nodePosition, node, outputPosition);
                ProducerOccurrence previous = producers.putIfAbsent(valueId, occurrence);
                if (previous != null) {
                    throw new IllegalArgumentException(
                            "nodes[" + nodePosition + "].outputs[" + outputPosition
                                    + "] duplicates produced value " + valueId);
                }
            }
        }
        producersByValue = Collections.unmodifiableMap(producers);

        var mutableConsumers = new LinkedHashMap<ValueId, List<ConsumerOccurrence>>();
        var derivedEdges = new ArrayList<Edge>();
        var derivedExternalInputs = new ArrayList<ConsumerOccurrence>();
        var locallyConsumedOutputs = new java.util.HashSet<ValueId>();
        for (int nodePosition = 0; nodePosition < this.nodes.size(); nodePosition++) {
            CompiledNode node = this.nodes.get(nodePosition);
            for (int inputPosition = 0; inputPosition < node.inputs().size(); inputPosition++) {
                ValueId valueId = Objects.requireNonNull(
                        node.inputs().get(inputPosition),
                        "nodes[" + nodePosition + "].inputs[" + inputPosition + "]");
                ConsumerOccurrence consumer =
                        new ConsumerOccurrence(valueId, nodePosition, node, inputPosition);
                mutableConsumers.computeIfAbsent(valueId, ignored -> new ArrayList<>()).add(consumer);
                ProducerOccurrence producer = producers.get(valueId);
                if (producer == null) {
                    derivedExternalInputs.add(consumer);
                } else {
                    if (producer.nodePosition() >= nodePosition) {
                        throw new IllegalArgumentException(
                                "nodes[" + nodePosition + "].inputs[" + inputPosition
                                        + "] is produced by its own or a later node: " + valueId);
                    }
                    derivedEdges.add(new Edge(producer, consumer));
                    locallyConsumedOutputs.add(valueId);
                }
            }
        }
        var consumers = new LinkedHashMap<ValueId, List<ConsumerOccurrence>>();
        mutableConsumers.forEach((valueId, occurrences) ->
                consumers.put(valueId, List.copyOf(occurrences)));
        consumersByValue = Collections.unmodifiableMap(consumers);
        edges = List.copyOf(derivedEdges);
        externalInputs = List.copyOf(derivedExternalInputs);

        var sinks = new ArrayList<CompiledNode>();
        for (CompiledNode node : this.nodes) {
            boolean locallyConsumed = false;
            for (ValueId output : node.outputs()) {
                locallyConsumed |= locallyConsumedOutputs.contains(output);
            }
            if (!locallyConsumed) {
                sinks.add(node);
            }
        }
        localSinks = List.copyOf(sinks);
    }

    /**
     * Returns the planned partition whose membership and order define this projection.
     *
     * @return the exact non-null planned partition supplied at construction
     */
    public PlannedPartition partition() {
        return partition;
    }

    /**
     * Returns the local nodes in stable topological order.
     *
     * @return the non-null immutable nodes in exact planned-partition order with exact retained
     *     references
     */
    public List<CompiledNode> nodes() {
        return nodes;
    }

    /**
     * Finds a local node by graph-local identity.
     *
     * @param nodeId non-null node identity to query
     * @return the exact local node, or empty when the identity is outside this partition
     * @throws NullPointerException if {@code nodeId} is {@code null}
     */
    public Optional<CompiledNode> node(NodeId nodeId) {
        return Optional.ofNullable(nodesById.get(Objects.requireNonNull(nodeId, "nodeId")));
    }

    /**
     * Finds the unique local producer of a value.
     *
     * @param valueId non-null logical value identity to query
     * @return its local producer occurrence, or empty for a partition-external or unknown value
     * @throws NullPointerException if {@code valueId} is {@code null}
     */
    public Optional<ProducerOccurrence> producer(ValueId valueId) {
        return Optional.ofNullable(
                producersByValue.get(Objects.requireNonNull(valueId, "valueId")));
    }

    /**
     * Returns all local consumer occurrences of a value in node and input-port order.
     *
     * @param valueId non-null logical value identity to query
     * @return immutable ordered occurrences, preserving repeated input ports; empty if unconsumed
     * @throws NullPointerException if {@code valueId} is {@code null}
     */
    public List<ConsumerOccurrence> consumers(ValueId valueId) {
        return consumersByValue.getOrDefault(
                Objects.requireNonNull(valueId, "valueId"), List.of());
    }

    /**
     * Returns every local producer-to-consumer occurrence.
     *
     * @return the non-null immutable local edges in consumer-node and input-port order; repeated
     *     input ports produce distinct edges
     */
    public List<Edge> edges() {
        return edges;
    }

    /**
     * Returns input-port occurrences whose values have no producer inside this partition.
     *
     * @return immutable partition-external input occurrences in node and input-port order;
     *     repeated ports remain distinct
     */
    public List<ConsumerOccurrence> externalInputs() {
        return externalInputs;
    }

    /**
     * Returns nodes whose output ports have no consumer inside this partition.
     *
     * @return immutable local sink nodes in partition order; publication and cross-partition use
     *     do not affect this topology-only result
     */
    public List<CompiledNode> localSinks() {
        return localSinks;
    }

    @Override
    public boolean equals(Object other) {
        return this == other
                || other instanceof PartitionDag that
                        && partition.equals(that.partition)
                        && nodes.equals(that.nodes);
    }

    @Override
    public int hashCode() {
        return Objects.hash(partition, nodes);
    }

    @Override
    public String toString() {
        return "PartitionDag[partition=" + partition + ", nodes=" + nodes + "]";
    }

    /**
     * Identifies one exact local output-port occurrence.
     *
     * @param valueId non-null produced value identity
     * @param nodePosition zero-based position in {@link PartitionDag#nodes()}
     * @param node exact non-null producing node reference
     * @param outputPosition zero-based position in {@link CompiledNode#outputs()}
     */
    public record ProducerOccurrence(
            ValueId valueId, int nodePosition, CompiledNode node, int outputPosition) {
        /**
         * Validates one immutable producer occurrence.
         *
         * @throws NullPointerException if {@code valueId} or {@code node} is {@code null}
         * @throws IllegalArgumentException if either position is negative or the output port does
         *     not contain {@code valueId}
         */
        public ProducerOccurrence {
            Objects.requireNonNull(valueId, "valueId");
            Objects.requireNonNull(node, "node");
            if (nodePosition < 0 || outputPosition < 0
                    || outputPosition >= node.outputs().size()
                    || !node.outputs().get(outputPosition).equals(valueId)) {
                throw new IllegalArgumentException("producer occurrence disagrees with node output");
            }
        }
    }

    /**
     * Identifies one exact local input-port occurrence, including a repeated or external input.
     *
     * @param valueId non-null consumed value identity
     * @param nodePosition zero-based position in {@link PartitionDag#nodes()}
     * @param node exact non-null consuming node reference
     * @param inputPosition zero-based position in {@link CompiledNode#inputs()}
     */
    public record ConsumerOccurrence(
            ValueId valueId, int nodePosition, CompiledNode node, int inputPosition) {
        /**
         * Validates one immutable consumer occurrence.
         *
         * @throws NullPointerException if {@code valueId} or {@code node} is {@code null}
         * @throws IllegalArgumentException if either position is negative or the input port does
         *     not contain {@code valueId}
         */
        public ConsumerOccurrence {
            Objects.requireNonNull(valueId, "valueId");
            Objects.requireNonNull(node, "node");
            if (nodePosition < 0 || inputPosition < 0
                    || inputPosition >= node.inputs().size()
                    || !node.inputs().get(inputPosition).equals(valueId)) {
                throw new IllegalArgumentException("consumer occurrence disagrees with node input");
            }
        }
    }

    /**
     * Associates one exact local producer output port with one exact local consumer input port.
     *
     * @param producer non-null producer occurrence
     * @param consumer non-null consumer occurrence of the same value at a later node position
     */
    public record Edge(ProducerOccurrence producer, ConsumerOccurrence consumer) {
        /**
         * Validates one immutable forward local edge.
         *
         * @throws NullPointerException if either occurrence is {@code null}
         * @throws IllegalArgumentException if value identities differ or the edge is not forward
         */
        public Edge {
            Objects.requireNonNull(producer, "producer");
            Objects.requireNonNull(consumer, "consumer");
            if (!producer.valueId().equals(consumer.valueId())
                    || producer.nodePosition() >= consumer.nodePosition()) {
                throw new IllegalArgumentException("edge occurrences disagree");
            }
        }
    }
}
