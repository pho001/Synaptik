package io.github.pho001.synaptik.model.graph;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Stores an immutable, structurally closed compile-time graph.
 *
 * <p>Values describe logical data, nodes describe computation in topological order, input and
 * output lists declare ordered graph boundaries, and the phase map classifies every node as
 * forward or backward work. The record owns immutable snapshots of those five components and no
 * derived producer, consumer, value, node, or phase index.</p>
 *
 * <p>This model validates structural closure only. It does not capture or transform a graph,
 * validate operation families or descriptors, allocate IDs or storage, build compile artifacts
 * or publication plans, retain public tensors, plan backend ownership, prepare execution, or hold
 * runtime state. Compiler and later lifecycle layers own those behaviors.</p>
 *
 * <p>Record-generated equality and hashing include all ordered lists and the structural phase
 * map. The generated {@link #toString()} is diagnostic text rather than serialization, validated
 * execution data, a schedule, or backend dispatch state.</p>
 *
 * @param values non-null ordered graph values with unique IDs; an immutable snapshot preserves
 *     encounter order
 * @param nodes non-null topologically ordered nodes with unique IDs; an immutable snapshot
 *     preserves encounter order
 * @param inputs non-null ordered unique graph-input IDs; every input is listed in {@code values},
 *     has no producer, and an immutable snapshot preserves encounter order
 * @param outputs non-null, non-empty ordered unique graph-output IDs; every output is listed in
 *     {@code values}, and an immutable snapshot preserves encounter order
 * @param nodePhases non-null structural mapping containing exactly one non-null phase for every
 *     node and no absent-node key; an immutable snapshot is retained without an iteration-order
 *     guarantee
 */
public record CompiledGraphModel(
        List<GraphValue> values,
        List<CompiledNode> nodes,
        List<ValueId> inputs,
        List<ValueId> outputs,
        Map<NodeId, GraphPhase> nodePhases) {
    /**
     * Creates an immutable graph snapshot after validating its complete structural closure.
     *
     * <p>Container references are checked in component order. Values, nodes, inputs, and outputs
     * are then checked in list order for null elements and duplicate identities and copied with
     * {@link List#copyOf(java.util.Collection)}. Outputs must be non-empty. The phase map rejects
     * a null key first, checks phases in ascending numeric node-ID order, and is copied with
     * {@link Map#copyOf(Map)}; map iteration order is not part of the result contract.</p>
     *
     * <p>Every boundary and node reference must resolve to a listed value. Inputs are
     * producer-free; every non-input value has exactly one producer. In stored node order, an
     * input must be a graph input or an output of an earlier node, so self-dependencies,
     * later-node dependencies, and cycles are rejected. Every node has exactly one phase and no
     * phase key names an absent node. Repeated node inputs, zero-input nodes, unused graph inputs,
     * and a zero-node pass-through graph are valid.</p>
     *
     * <p>Validation follows the component and structural order described above. Indexed messages
     * use zero-based positions and identifier text such as {@code ValueId[value=7]} and
     * {@code NodeId[value=3]}.</p>
     *
     * @param values non-null ordered values to validate and snapshot; elements and IDs must be
     *     non-null and IDs must be unique
     * @param nodes non-null topologically ordered nodes to validate and snapshot; elements and IDs
     *     must be non-null and IDs must be unique
     * @param inputs non-null ordered graph-input IDs to validate and snapshot; elements must be
     *     non-null, unique, listed in {@code values}, and producer-free
     * @param outputs non-null, non-empty ordered graph-output IDs to validate and snapshot;
     *     elements must be non-null, unique, and listed in {@code values}
     * @param nodePhases non-null node-to-phase mapping to validate and snapshot; keys and values
     *     must be non-null and coverage must exactly match {@code nodes}
     * @throws NullPointerException if a component reference is {@code null}; the message is the
     *     first null component name in declaration order
     * @throws NullPointerException if a list element is {@code null}; the message is the component
     *     name followed by its zero-based index, such as {@code values[1]}
     * @throws NullPointerException if the phase map contains a null key; the message is
     *     {@code nodePhases contains null key}
     * @throws NullPointerException if the phase for a non-null key is {@code null}; the message is
     *     {@code nodePhases[NodeId[value=n]]}, with keys checked in ascending numeric order
     * @throws IllegalArgumentException if a later value, node, input, or output ID duplicates an
     *     earlier ID; the message is {@code component[index] duplicates identifier}
     * @throws IllegalArgumentException if {@code outputs} is empty; the message is
     *     {@code outputs must not be empty}
     * @throws IllegalArgumentException if an input or output boundary references an unknown value;
     *     the message is {@code component[index] references unknown ValueId[value=n]}
     * @throws IllegalArgumentException if a node input references an unknown value; the message is
     *     {@code nodes[nodeIndex].inputs[inputIndex] references unknown ValueId[value=n]}
     * @throws IllegalArgumentException if a node input is not a graph input or earlier-node
     *     output; the message is
     *     {@code nodes[nodeIndex].inputs[inputIndex] is not available before NodeId[value=n]: ValueId[value=m]}
     * @throws IllegalArgumentException if a node output references an unknown value; the message
     *     is {@code nodes[nodeIndex].outputs[outputIndex] references unknown ValueId[value=n]}
     * @throws IllegalArgumentException if a node output produces a declared graph input; the
     *     message is
     *     {@code nodes[nodeIndex].outputs[outputIndex] produces graph input ValueId[value=n]}
     * @throws IllegalArgumentException if a node output gives a value a second producer; the
     *     message is
     *     {@code nodes[nodeIndex].outputs[outputIndex] gives ValueId[value=n] a second producer; first producer is NodeId[value=m]}
     * @throws IllegalArgumentException if a listed non-input value has no producer; the message is
     *     {@code values[index] is neither a graph input nor a node output: ValueId[value=n]}
     * @throws IllegalArgumentException if a node has no phase or the phase map names an absent
     *     node; the message is {@code nodePhases missing NodeId[value=n]} in stored node order or
     *     {@code nodePhases contains unknown NodeId[value=n]} in ascending numeric key order
     */
    public CompiledGraphModel {
        Objects.requireNonNull(values, "values");
        Objects.requireNonNull(nodes, "nodes");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(outputs, "outputs");
        Objects.requireNonNull(nodePhases, "nodePhases");

        Set<ValueId> valueIds = new HashSet<>();
        for (int index = 0; index < values.size(); index++) {
            GraphValue value = Objects.requireNonNull(values.get(index), "values[" + index + "]");
            if (!valueIds.add(value.id())) {
                throw new IllegalArgumentException(
                        "values[" + index + "] duplicates " + value.id());
            }
        }
        values = List.copyOf(values);

        Set<NodeId> nodeIds = new HashSet<>();
        for (int index = 0; index < nodes.size(); index++) {
            CompiledNode node = Objects.requireNonNull(nodes.get(index), "nodes[" + index + "]");
            if (!nodeIds.add(node.id())) {
                throw new IllegalArgumentException(
                        "nodes[" + index + "] duplicates " + node.id());
            }
        }
        nodes = List.copyOf(nodes);

        Set<ValueId> inputIds = new HashSet<>();
        for (int index = 0; index < inputs.size(); index++) {
            ValueId input = Objects.requireNonNull(inputs.get(index), "inputs[" + index + "]");
            if (!inputIds.add(input)) {
                throw new IllegalArgumentException(
                        "inputs[" + index + "] duplicates " + input);
            }
        }
        inputs = List.copyOf(inputs);

        if (outputs.isEmpty()) {
            throw new IllegalArgumentException("outputs must not be empty");
        }
        Set<ValueId> outputIds = new HashSet<>();
        for (int index = 0; index < outputs.size(); index++) {
            ValueId output = Objects.requireNonNull(outputs.get(index), "outputs[" + index + "]");
            if (!outputIds.add(output)) {
                throw new IllegalArgumentException(
                        "outputs[" + index + "] duplicates " + output);
            }
        }
        outputs = List.copyOf(outputs);

        for (NodeId nodeId : nodePhases.keySet()) {
            if (nodeId == null) {
                throw new NullPointerException("nodePhases contains null key");
            }
        }
        List<NodeId> sortedPhaseKeys = new ArrayList<>(nodePhases.keySet());
        sortedPhaseKeys.sort(Comparator.comparingLong(NodeId::value));
        for (NodeId nodeId : sortedPhaseKeys) {
            Objects.requireNonNull(nodePhases.get(nodeId), "nodePhases[" + nodeId + "]");
        }
        nodePhases = Map.copyOf(nodePhases);

        for (int index = 0; index < inputs.size(); index++) {
            ValueId input = inputs.get(index);
            if (!valueIds.contains(input)) {
                throw new IllegalArgumentException(
                        "inputs[" + index + "] references unknown " + input);
            }
        }

        for (int index = 0; index < outputs.size(); index++) {
            ValueId output = outputs.get(index);
            if (!valueIds.contains(output)) {
                throw new IllegalArgumentException(
                        "outputs[" + index + "] references unknown " + output);
            }
        }

        Set<ValueId> available = new HashSet<>(inputIds);
        Map<ValueId, NodeId> producers = new HashMap<>();
        for (int nodeIndex = 0; nodeIndex < nodes.size(); nodeIndex++) {
            CompiledNode node = nodes.get(nodeIndex);
            for (int inputIndex = 0; inputIndex < node.inputs().size(); inputIndex++) {
                ValueId input = node.inputs().get(inputIndex);
                if (!valueIds.contains(input)) {
                    throw new IllegalArgumentException("nodes[" + nodeIndex + "].inputs["
                            + inputIndex + "] references unknown " + input);
                }
                if (!available.contains(input)) {
                    throw new IllegalArgumentException("nodes[" + nodeIndex + "].inputs["
                            + inputIndex + "] is not available before " + node.id() + ": " + input);
                }
            }
            for (int outputIndex = 0; outputIndex < node.outputs().size(); outputIndex++) {
                ValueId output = node.outputs().get(outputIndex);
                if (!valueIds.contains(output)) {
                    throw new IllegalArgumentException("nodes[" + nodeIndex + "].outputs["
                            + outputIndex + "] references unknown " + output);
                }
                if (inputIds.contains(output)) {
                    throw new IllegalArgumentException("nodes[" + nodeIndex + "].outputs["
                            + outputIndex + "] produces graph input " + output);
                }
                NodeId firstProducer = producers.putIfAbsent(output, node.id());
                if (firstProducer != null) {
                    throw new IllegalArgumentException("nodes[" + nodeIndex + "].outputs["
                            + outputIndex + "] gives " + output
                            + " a second producer; first producer is " + firstProducer);
                }
            }
            available.addAll(node.outputs());
        }

        for (int index = 0; index < values.size(); index++) {
            ValueId valueId = values.get(index).id();
            if (!inputIds.contains(valueId) && !producers.containsKey(valueId)) {
                throw new IllegalArgumentException("values[" + index
                        + "] is neither a graph input nor a node output: " + valueId);
            }
        }

        for (CompiledNode node : nodes) {
            if (!nodePhases.containsKey(node.id())) {
                throw new IllegalArgumentException("nodePhases missing " + node.id());
            }
        }

        List<NodeId> sortedPhaseSnapshotKeys = new ArrayList<>(nodePhases.keySet());
        sortedPhaseSnapshotKeys.sort(Comparator.comparingLong(NodeId::value));
        for (NodeId nodeId : sortedPhaseSnapshotKeys) {
            if (!nodeIds.contains(nodeId)) {
                throw new IllegalArgumentException("nodePhases contains unknown " + nodeId);
            }
        }
    }

    /**
     * Returns graph values in their caller-defined deterministic order.
     *
     * <p>The result is a non-null immutable snapshot with unique value IDs. Mutation attempts
     * throw {@link UnsupportedOperationException}; no identity relationship with the supplied or
     * another equal list is promised.</p>
     *
     * @return the immutable ordered snapshot of non-null graph values
     */
    public List<GraphValue> values() {
        return values;
    }

    /**
     * Returns computation nodes in validated topological order.
     *
     * <p>Contained nodes may have zero inputs or repeated input IDs. The list is a non-null
     * immutable snapshot; mutation attempts throw {@link UnsupportedOperationException}, and no
     * list-container identity is promised.</p>
     *
     * @return the immutable topologically ordered snapshot of non-null nodes
     */
    public List<CompiledNode> nodes() {
        return nodes;
    }

    /**
     * Returns the caller-defined ordered graph-input boundary.
     *
     * <p>The unique IDs are listed values with no producer. The result is a non-null immutable
     * snapshot; mutation attempts throw {@link UnsupportedOperationException}, and no
     * list-container identity is promised.</p>
     *
     * @return the immutable ordered snapshot of unique non-null graph-input IDs
     */
    public List<ValueId> inputs() {
        return inputs;
    }

    /**
     * Returns the caller-defined ordered graph-output boundary.
     *
     * <p>The result contains at least one unique listed value ID and is a non-null immutable
     * snapshot. Mutation attempts throw {@link UnsupportedOperationException}; no list-container
     * identity is promised.</p>
     *
     * @return the immutable non-empty ordered snapshot of unique non-null graph-output IDs
     */
    public List<ValueId> outputs() {
        return outputs;
    }

    /**
     * Returns the exact structural forward/backward classification of the stored nodes.
     *
     * <p>The result contains one non-null phase for each node and no other key. It is a non-null
     * immutable snapshot with structural map semantics; mutation attempts throw
     * {@link UnsupportedOperationException}. Neither insertion, sorted, nor hash iteration order
     * is part of this accessor's contract.</p>
     *
     * @return the immutable exact node-to-phase mapping with unspecified iteration order
     */
    public Map<NodeId, GraphPhase> nodePhases() {
        return nodePhases;
    }
}
