package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
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
 * Merges exact internal common subexpressions within each graph phase in one topological scan.
 *
 * <p>Equality covers the exact {@link GraphPhase}, complete immutable operation value, ordered
 * already-remapped inputs including repetitions, and every ordered output descriptor. A merge
 * remaps every output slot as one indivisible occurrence. {@code FORWARD} and {@code BACKWARD}
 * occurrences never compare equal, and graph-output producers are never candidates or
 * representatives. Fixed recurrent-scan occurrences are also excluded because each Model
 * producer is an identity-distinct occurrence even when its immutable structure is equal.</p>
 */
final class ForwardCommonSubexpressionElimination {
    private ForwardCommonSubexpressionElimination() {}

    /**
     * Reuses the first eligible exact occurrence in the same phase and remaps every output
     * position.
     *
     * @param graph the non-null immutable graph in topological order; it is not mutated
     * @return the exact {@code graph} reference when no occurrence merges; otherwise a canonical,
     *     non-null immutable graph with all graph inputs and remapped graph outputs in their
     *     original order
     * @throws NullPointerException if {@code graph} is {@code null}
     */
    static CompiledGraphModel eliminate(CompiledGraphModel graph) {
        return rebuild(graph, null).graph();
    }

    /** Derivative-order-aware CSE entry used by combined functional graphs. */
    static final class DerivativeAware {
        private DerivativeAware() {}

        /**
         * Merges exact expressions only within equal phase and derivative order.
         *
         * @param derivatives non-null metadata owning the input graph
         * @return non-null eliminated graph and matching metadata
         */
        static Result eliminate(DerivativeGraphMetadata derivatives) {
            Objects.requireNonNull(derivatives, "derivatives");
            Rebuild rebuild = rebuild(derivatives.graph(), derivatives);
            if (rebuild.graph() == derivatives.graph()) {
                return new Result(rebuild.graph(), derivatives);
            }
            return new Result(
                    rebuild.graph(),
                    DerivativeGraphMetadata.remap(
                            derivatives, rebuild.graph(), rebuild.sourceNodeIds()));
        }
    }

    private static Rebuild rebuild(
            CompiledGraphModel graph, DerivativeGraphMetadata derivatives) {
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
        List<NodeId> sourceNodeIds = new ArrayList<>(graph.nodes().size());
        Map<NodeId, GraphPhase> phases = new LinkedHashMap<>();
        long nextNodeId = 0;
        boolean changed = false;
        for (CompiledNode node : graph.nodes()) {
            GraphPhase phase = graph.nodePhases().get(node.id());
            int derivativeOrder = derivatives == null
                    ? (phase == GraphPhase.FORWARD ? 0 : 1)
                    : derivatives.derivativeOrderByNode().get(node.id());
            List<ValueId> remappedInputs = remap(node.inputs(), valueRemapping);
            List<TensorDescriptor> outputDescriptors = descriptors(
                    node.outputs(), originalValues);
            boolean eligible = !(node.operation().kind() instanceof RecurrentScanKind)
                    && node.outputs().stream().noneMatch(graphOutputs::contains);
            ExpressionKey key = eligible
                    ? new ExpressionKey(
                            phase,
                            derivativeOrder,
                            node.operation(),
                            remappedInputs,
                            outputDescriptors)
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
            sourceNodeIds.add(node.id());
            phases.put(rebuiltNode, phase);
            if (eligible) {
                representatives.put(key, List.copyOf(rebuiltOutputs));
            }
        }

        if (!changed) {
            return new Rebuild(graph, List.copyOf(sourceNodeIds));
        }
        return new Rebuild(
                new CompiledGraphModel(
                        values, nodes, inputs, remap(graph.outputs(), valueRemapping), phases),
                List.copyOf(sourceNodeIds));
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
            int derivativeOrder,
            Operation operation,
            List<ValueId> inputs,
            List<TensorDescriptor> outputs) {
        private ExpressionKey {
            inputs = List.copyOf(inputs);
            outputs = List.copyOf(outputs);
        }
    }

    /**
     * Eliminated graph and its matching derivative metadata.
     *
     * @param graph non-null eliminated graph
     * @param derivatives non-null matching metadata
     */
    record Result(CompiledGraphModel graph, DerivativeGraphMetadata derivatives) {}

    private record Rebuild(CompiledGraphModel graph, List<NodeId> sourceNodeIds) {}
}
