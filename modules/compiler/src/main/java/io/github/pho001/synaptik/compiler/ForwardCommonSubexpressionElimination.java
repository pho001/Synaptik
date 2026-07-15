package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Merges exact internal forward common subexpressions in one topological scan.
 *
 * <p>Equality covers the forward phase, complete immutable operation value, ordered
 * already-remapped inputs including repetitions, and every ordered output descriptor. A merge
 * remaps every output slot as one indivisible occurrence. Graph-output producers and non-forward
 * occurrences are never candidates or representatives.</p>
 */
final class ForwardCommonSubexpressionElimination {
    private ForwardCommonSubexpressionElimination() {}

    /**
     * Reuses the first eligible exact forward occurrence and remaps every output position.
     *
     * @param graph the non-null immutable graph in topological order; it is not mutated
     * @return the exact {@code graph} reference when no occurrence merges; otherwise a canonical,
     *     non-null immutable graph with all graph inputs and remapped graph outputs in their
     *     original order
     * @throws NullPointerException if {@code graph} is {@code null}
     */
    static CompiledGraphModel eliminate(CompiledGraphModel graph) {
        Objects.requireNonNull(graph, "graph");

        Map<ValueId, GraphValue> originalValues = new HashMap<>();
        for (GraphValue value : graph.values()) {
            originalValues.put(value.id(), value);
        }
        Set<ValueId> graphOutputs = new HashSet<>(graph.outputs());
        Map<ValueId, ValueId> valueRemapping = new HashMap<>();
        List<GraphValue> values = new ArrayList<>();
        List<ValueId> inputs = new ArrayList<>(graph.inputs().size());
        long nextValueId = 0;
        for (ValueId input : graph.inputs()) {
            ValueId rebuiltInput = new ValueId(nextValueId++);
            valueRemapping.put(input, rebuiltInput);
            inputs.add(rebuiltInput);
            values.add(new GraphValue(rebuiltInput, originalValues.get(input).descriptor()));
        }

        Map<ExpressionKey, List<ValueId>> representatives = new HashMap<>();
        List<CompiledNode> nodes = new ArrayList<>(graph.nodes().size());
        Map<NodeId, GraphPhase> phases = new LinkedHashMap<>();
        long nextNodeId = 0;
        boolean changed = false;
        for (CompiledNode node : graph.nodes()) {
            GraphPhase phase = graph.nodePhases().get(node.id());
            List<ValueId> remappedInputs = remap(node.inputs(), valueRemapping);
            List<TensorDescriptor> outputDescriptors = descriptors(
                    node.outputs(), originalValues);
            boolean eligible = phase == GraphPhase.FORWARD
                    && node.outputs().stream().noneMatch(graphOutputs::contains);
            ExpressionKey key = eligible
                    ? new ExpressionKey(phase, node.operation(), remappedInputs, outputDescriptors)
                    : null;
            List<ValueId> representativeOutputs = eligible ? representatives.get(key) : null;
            if (representativeOutputs != null) {
                for (int output = 0; output < node.outputs().size(); output++) {
                    valueRemapping.put(node.outputs().get(output), representativeOutputs.get(output));
                }
                changed = true;
                continue;
            }

            List<ValueId> rebuiltOutputs = new ArrayList<>(node.outputs().size());
            for (int output = 0; output < node.outputs().size(); output++) {
                ValueId rebuiltOutput = new ValueId(nextValueId++);
                valueRemapping.put(node.outputs().get(output), rebuiltOutput);
                rebuiltOutputs.add(rebuiltOutput);
                values.add(new GraphValue(rebuiltOutput, outputDescriptors.get(output)));
            }
            NodeId rebuiltNode = new NodeId(nextNodeId++);
            nodes.add(new CompiledNode(
                    rebuiltNode, node.operation(), remappedInputs, rebuiltOutputs));
            phases.put(rebuiltNode, phase);
            if (eligible) {
                representatives.put(key, List.copyOf(rebuiltOutputs));
            }
        }

        if (!changed) {
            return graph;
        }
        return new CompiledGraphModel(
                values, nodes, inputs, remap(graph.outputs(), valueRemapping), phases);
    }

    private static List<TensorDescriptor> descriptors(
            List<ValueId> ids, Map<ValueId, GraphValue> values) {
        List<TensorDescriptor> result = new ArrayList<>(ids.size());
        for (ValueId id : ids) {
            result.add(values.get(id).descriptor());
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

    private record ExpressionKey(
            GraphPhase phase,
            Operation operation,
            List<ValueId> inputs,
            List<TensorDescriptor> outputs) {
        private ExpressionKey {
            inputs = List.copyOf(inputs);
            outputs = List.copyOf(outputs);
        }
    }
}
