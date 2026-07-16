package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Removes unreachable forward occurrences while retaining every non-forward dependency closure.
 *
 * <p>Liveness is walked iteratively from graph outputs and every non-forward occurrence. Those
 * occurrences and their dependency closures are roots outside forward-only elimination. A live
 * node is indivisible: all of its output slots are retained even when only one slot is needed.</p>
 */
final class ForwardDeadCodeElimination {
    private ForwardDeadCodeElimination() {}

    /**
     * Eliminates only forward nodes unreachable from observable and phase-classified roots.
     *
     * @param graph the non-null immutable graph whose stored node order is topological; it is not
     *     mutated
     * @return the exact {@code graph} reference when no node is removed; otherwise a canonical,
     *     non-null immutable graph retaining every graph input, graph output, non-forward
     *     dependency closure, and complete output set of every live node
     * @throws NullPointerException if {@code graph} is {@code null}
     */
    static CompiledGraphModel eliminate(CompiledGraphModel graph) {
        Objects.requireNonNull(graph, "graph");

        Map<ValueId, Integer> producerIndexes = new HashMap<>();
        for (int index = 0; index < graph.nodes().size(); index++) {
            for (ValueId output : graph.nodes().get(index).outputs()) {
                producerIndexes.put(output, index);
            }
        }

        boolean[] liveNodes = new boolean[graph.nodes().size()];
        Set<ValueId> neededValues = new HashSet<>();
        ArrayDeque<ValueId> work = new ArrayDeque<>();
        addNeeded(graph.outputs(), neededValues, work);
        for (int index = 0; index < graph.nodes().size(); index++) {
            CompiledNode node = graph.nodes().get(index);
            if (graph.nodePhases().get(node.id()) != GraphPhase.FORWARD) {
                liveNodes[index] = true;
                addNeeded(node.outputs(), neededValues, work);
                addNeeded(node.inputs(), neededValues, work);
            }
        }

        while (!work.isEmpty()) {
            Integer producerIndex = producerIndexes.get(work.removeLast());
            if (producerIndex == null || liveNodes[producerIndex]) {
                continue;
            }
            liveNodes[producerIndex] = true;
            CompiledNode producer = graph.nodes().get(producerIndex);
            addNeeded(producer.outputs(), neededValues, work);
            addNeeded(producer.inputs(), neededValues, work);
        }

        int liveCount = 0;
        for (boolean live : liveNodes) {
            if (live) {
                liveCount++;
            }
        }
        if (liveCount == graph.nodes().size()) {
            return graph;
        }

        return rebuild(graph, liveNodes, liveCount);
    }

    /**
     * Eliminates dead forward work and prunes only unused fixed constant sources.
     *
     * <p>The graph-only pass runs first and preserves its existing contract. Source roles are then
     * remapped by input position. Every bindable input remains even when unused; a constant input
     * remains only when it is a graph output or an ordered input of a retained node.</p>
     *
     * @param constantGraph non-null immutable graph and exact source facts; it is not mutated
     * @return the exact argument when neither graph work nor a constant source is removed;
     *     otherwise a non-null immutable sidecar with dense input-first IDs and remapped facts,
     *     nodes, phases, and outputs
     * @throws NullPointerException if {@code constantGraph} is null
     */
    static CompileTimeConstantGraph eliminate(CompileTimeConstantGraph constantGraph) {
        Objects.requireNonNull(constantGraph, "constantGraph");
        CompiledGraphModel eliminated = eliminate(constantGraph.graph());
        CompileTimeConstantGraph current =
                constantGraph.replaceGraphPreservingInputRoles(eliminated);

        Set<ValueId> liveConstants = new HashSet<>(eliminated.outputs());
        for (CompiledNode node : eliminated.nodes()) {
            liveConstants.addAll(node.inputs());
        }
        int retainedInputs = 0;
        for (ValueId input : eliminated.inputs()) {
            if (!current.constants().containsKey(input) || liveConstants.contains(input)) {
                retainedInputs++;
            }
        }
        if (retainedInputs == eliminated.inputs().size()) {
            return current;
        }
        return pruneConstantInputs(current, liveConstants, retainedInputs);
    }

    private static void addNeeded(
            List<ValueId> values, Set<ValueId> neededValues, ArrayDeque<ValueId> work) {
        for (ValueId value : values) {
            if (neededValues.add(value)) {
                work.addLast(value);
            }
        }
    }

    private static CompiledGraphModel rebuild(
            CompiledGraphModel graph, boolean[] liveNodes, int liveCount) {
        Map<ValueId, GraphValue> originalValues = new HashMap<>();
        for (GraphValue value : graph.values()) {
            originalValues.put(value.id(), value);
        }

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

        List<CompiledNode> nodes = new ArrayList<>(liveCount);
        Map<NodeId, GraphPhase> phases = new LinkedHashMap<>();
        long nextNodeId = 0;
        for (int index = 0; index < graph.nodes().size(); index++) {
            if (!liveNodes[index]) {
                continue;
            }
            CompiledNode node = graph.nodes().get(index);
            List<ValueId> rebuiltInputs = remap(node.inputs(), valueRemapping);
            List<ValueId> rebuiltOutputs = new ArrayList<>(node.outputs().size());
            for (ValueId output : node.outputs()) {
                ValueId rebuiltOutput = new ValueId(nextValueId++);
                valueRemapping.put(output, rebuiltOutput);
                rebuiltOutputs.add(rebuiltOutput);
                values.add(new GraphValue(
                        rebuiltOutput, originalValues.get(output).descriptor()));
            }
            NodeId rebuiltNode = new NodeId(nextNodeId++);
            nodes.add(new CompiledNode(
                    rebuiltNode, node.operation(), rebuiltInputs, rebuiltOutputs));
            phases.put(rebuiltNode, graph.nodePhases().get(node.id()));
        }

        return new CompiledGraphModel(
                values, nodes, inputs, remap(graph.outputs(), valueRemapping), phases);
    }

    private static CompileTimeConstantGraph pruneConstantInputs(
            CompileTimeConstantGraph source,
            Set<ValueId> liveConstants,
            int retainedInputCount) {
        CompiledGraphModel graph = source.graph();
        Map<ValueId, GraphValue> originalValues = new HashMap<>();
        for (GraphValue value : graph.values()) {
            originalValues.put(value.id(), value);
        }

        Map<ValueId, ValueId> remapping = new HashMap<>();
        Map<ValueId, CompileTimeConstantGraph.Splat> constants = new HashMap<>();
        List<GraphValue> values = new ArrayList<>(
                graph.values().size() - (graph.inputs().size() - retainedInputCount));
        List<ValueId> inputs = new ArrayList<>(retainedInputCount);
        long nextValueId = 0;
        for (ValueId input : graph.inputs()) {
            CompileTimeConstantGraph.Splat splat = source.constants().get(input);
            if (splat != null && !liveConstants.contains(input)) {
                continue;
            }
            ValueId rebuilt = new ValueId(nextValueId++);
            remapping.put(input, rebuilt);
            inputs.add(rebuilt);
            values.add(new GraphValue(rebuilt, originalValues.get(input).descriptor()));
            if (splat != null) {
                constants.put(rebuilt, splat);
            }
        }

        List<CompiledNode> nodes = new ArrayList<>(graph.nodes().size());
        Map<NodeId, GraphPhase> phases = new LinkedHashMap<>();
        long nextNodeId = 0;
        for (CompiledNode node : graph.nodes()) {
            List<ValueId> rebuiltInputs = remap(node.inputs(), remapping);
            List<ValueId> rebuiltOutputs = new ArrayList<>(node.outputs().size());
            for (ValueId output : node.outputs()) {
                ValueId rebuiltOutput = new ValueId(nextValueId++);
                remapping.put(output, rebuiltOutput);
                rebuiltOutputs.add(rebuiltOutput);
                values.add(new GraphValue(rebuiltOutput, originalValues.get(output).descriptor()));
            }
            NodeId rebuiltNode = new NodeId(nextNodeId++);
            nodes.add(new CompiledNode(
                    rebuiltNode, node.operation(), rebuiltInputs, rebuiltOutputs));
            phases.put(rebuiltNode, graph.nodePhases().get(node.id()));
        }

        CompiledGraphModel rebuilt = new CompiledGraphModel(
                values, nodes, inputs, remap(graph.outputs(), remapping), phases);
        return new CompileTimeConstantGraph(rebuilt, constants);
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
