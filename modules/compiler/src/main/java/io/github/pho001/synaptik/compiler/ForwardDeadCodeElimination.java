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
 * Removes unreachable occurrences from the complete selected graph regardless of phase.
 *
 * <p>Liveness is walked iteratively from the ordered graph-output boundary through producer
 * dependencies. A {@link GraphPhase#BACKWARD BACKWARD} occurrence is not retained merely because
 * of its phase. A live producer occurrence remains indivisible: all of its output slots are
 * retained when any slot is needed. Rebuilds preserve every live node's exact operation,
 * descriptors, and phase.</p>
 */
final class ForwardDeadCodeElimination {
    private ForwardDeadCodeElimination() {}

    /**
     * Eliminates every node unreachable from the complete graph-output boundary.
     *
     * @param graph the non-null immutable graph whose stored node order is topological; it is not
     *     mutated
     * @return the exact {@code graph} reference when no node is removed; otherwise a canonical,
     *     non-null immutable graph retaining every graph input, graph output, complete dependency
     *     closure, and complete output set of every live node
     * @throws NullPointerException if {@code graph} is {@code null}
     */
    static CompiledGraphModel eliminate(CompiledGraphModel graph) {
        return eliminateGraph(graph).graph();
    }

    private static Rebuild eliminateGraph(CompiledGraphModel graph) {
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
            return new Rebuild(
                    graph, graph.nodes().stream().map(CompiledNode::id).toList());
        }

        return rebuild(graph, liveNodes, liveCount);
    }

    /**
     * Eliminates dead whole-graph work and prunes only unused fixed constant sources.
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

    /** Derivative-order-aware DCE entry used by combined functional graphs. */
    static final class DerivativeAware {
        private DerivativeAware() {}

        /**
         * Eliminates dead work and remaps derivative metadata.
         *
         * @param constantGraph non-null graph and source facts
         * @param derivatives non-null metadata owning the exact graph
         * @return non-null eliminated graph/source facts and matching metadata
         */
        static Result eliminate(
                CompileTimeConstantGraph constantGraph,
                DerivativeGraphMetadata derivatives) {
            Objects.requireNonNull(constantGraph, "constantGraph");
            Objects.requireNonNull(derivatives, "derivatives");
            if (derivatives.graph() != constantGraph.graph()) {
                throw new IllegalArgumentException(
                        "derivatives graph must be the exact graph being eliminated");
            }
            Rebuild graphRebuild = eliminateGraph(constantGraph.graph());
            DerivativeGraphMetadata currentDerivatives =
                    graphRebuild.graph() == derivatives.graph()
                            ? derivatives
                            : DerivativeGraphMetadata.remap(
                                    derivatives,
                                    graphRebuild.graph(),
                                    graphRebuild.sourceNodeIds());
            CompileTimeConstantGraph current =
                    constantGraph.replaceGraphPreservingInputRoles(graphRebuild.graph());

            Set<ValueId> liveConstants = new HashSet<>(current.graph().outputs());
            for (CompiledNode node : current.graph().nodes()) {
                liveConstants.addAll(node.inputs());
            }
            int retainedInputs = 0;
            for (ValueId input : current.graph().inputs()) {
                if (!current.constants().containsKey(input) || liveConstants.contains(input)) {
                    retainedInputs++;
                }
            }
            if (retainedInputs == current.graph().inputs().size()) {
                return new Result(current, currentDerivatives);
            }
            CompileTimeConstantGraph pruned =
                    pruneConstantInputs(current, liveConstants, retainedInputs);
            DerivativeGraphMetadata prunedDerivatives = DerivativeGraphMetadata.remap(
                    currentDerivatives,
                    pruned.graph(),
                    current.graph().nodes().stream().map(CompiledNode::id).toList());
            return new Result(pruned, prunedDerivatives);
        }
    }

    private static void addNeeded(
            List<ValueId> values, Set<ValueId> neededValues, ArrayDeque<ValueId> work) {
        for (ValueId value : values) {
            if (neededValues.add(value)) {
                work.addLast(value);
            }
        }
    }

    private static Rebuild rebuild(
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
        List<NodeId> sourceNodeIds = new ArrayList<>(liveCount);
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
            sourceNodeIds.add(node.id());
            phases.put(rebuiltNode, graph.nodePhases().get(node.id()));
        }

        return new Rebuild(
                new CompiledGraphModel(
                        values, nodes, inputs, remap(graph.outputs(), valueRemapping), phases),
                List.copyOf(sourceNodeIds));
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

    /**
     * Eliminated graph/source facts and their matching derivative metadata.
     *
     * @param constantGraph non-null eliminated graph and source facts
     * @param derivatives non-null matching metadata
     */
    record Result(
            CompileTimeConstantGraph constantGraph,
            DerivativeGraphMetadata derivatives) {}

    private record Rebuild(CompiledGraphModel graph, List<NodeId> sourceNodeIds) {}
}
