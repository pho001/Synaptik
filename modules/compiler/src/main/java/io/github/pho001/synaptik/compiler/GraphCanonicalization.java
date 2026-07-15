package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Rebuilds an immutable graph with deterministic dense graph-local identities.
 *
 * <p>Inputs are allocated first in boundary order, followed by node outputs in stored topological
 * and output-slot order; node identifiers follow the same stored node order. The rebuild preserves
 * every occurrence, boundary position, phase, repeated input position, and output slot. It changes
 * only graph-local identifiers and storage order.</p>
 */
final class GraphCanonicalization {
    private GraphCanonicalization() {}

    /**
     * Canonicalizes a structurally closed graph in input, node, and output-slot order.
     *
     * @param graph the non-null immutable graph to rebuild without semantic transformation; it is
     *     not mutated
     * @return a new non-null immutable graph whose values and nodes have dense identifiers from
     *     zero; exact operation and descriptor element references are retained
     * @throws NullPointerException if {@code graph} is {@code null}
     */
    static CompiledGraphModel canonicalize(CompiledGraphModel graph) {
        Objects.requireNonNull(graph, "graph");

        Map<ValueId, GraphValue> originalValues = valuesById(graph);
        Map<ValueId, ValueId> valueRemapping = new HashMap<>();
        List<GraphValue> values = new ArrayList<>(graph.values().size());
        List<ValueId> inputs = new ArrayList<>(graph.inputs().size());
        long nextValueId = 0;
        for (ValueId input : graph.inputs()) {
            ValueId canonicalInput = new ValueId(nextValueId++);
            valueRemapping.put(input, canonicalInput);
            inputs.add(canonicalInput);
            values.add(new GraphValue(canonicalInput, originalValues.get(input).descriptor()));
        }

        List<CompiledNode> nodes = new ArrayList<>(graph.nodes().size());
        Map<NodeId, GraphPhase> phases = new LinkedHashMap<>();
        long nextNodeId = 0;
        for (CompiledNode node : graph.nodes()) {
            List<ValueId> remappedInputs = remap(node.inputs(), valueRemapping);
            List<ValueId> remappedOutputs = new ArrayList<>(node.outputs().size());
            for (ValueId output : node.outputs()) {
                ValueId canonicalOutput = new ValueId(nextValueId++);
                valueRemapping.put(output, canonicalOutput);
                remappedOutputs.add(canonicalOutput);
                values.add(new GraphValue(
                        canonicalOutput, originalValues.get(output).descriptor()));
            }
            NodeId canonicalNode = new NodeId(nextNodeId++);
            nodes.add(new CompiledNode(
                    canonicalNode, node.operation(), remappedInputs, remappedOutputs));
            phases.put(canonicalNode, graph.nodePhases().get(node.id()));
        }

        return new CompiledGraphModel(
                values, nodes, inputs, remap(graph.outputs(), valueRemapping), phases);
    }

    private static Map<ValueId, GraphValue> valuesById(CompiledGraphModel graph) {
        Map<ValueId, GraphValue> result = new HashMap<>();
        for (GraphValue value : graph.values()) {
            result.put(value.id(), value);
        }
        return result;
    }

    private static List<ValueId> remap(
            List<ValueId> original, Map<ValueId, ValueId> valueRemapping) {
        List<ValueId> result = new ArrayList<>(original.size());
        for (ValueId value : original) {
            result.add(valueRemapping.get(value));
        }
        return result;
    }
}
