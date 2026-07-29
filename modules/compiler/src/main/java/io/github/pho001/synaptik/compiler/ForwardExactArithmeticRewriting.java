package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
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
 * Bypasses a closed seven-rule set of exact non-gradient arithmetic identities in one
 * whole-graph topological scan.
 *
 * <p>The rules are duplicate-input binary {@code MIN} and {@code MAX}; scalar {@code MUL} by exact
 * typed positive one for BFLOAT16, FLOAT32, FLOAT64, INT32, and INT64; scalar {@code DIV} and
 * {@code POW} by exact typed positive one for the three floating types; and scalar {@code ADD} and
 * {@code SUB} by exact typed zero for INT32 and INT64. Scalar values come only from immutable
 * {@code ScalarValueAttrs} operation metadata. They are not Tensor constants, host storage, or
 * evaluated values.</p>
 *
 * <p>A guarded occurrence may be {@link GraphPhase#FORWARD FORWARD} or
 * {@link GraphPhase#BACKWARD BACKWARD}. Every bypass still requires exactly one non-graph-output
 * result, complete equality between that output descriptor and the selected input descriptor,
 * and {@code requiresGrad == false}. The scan compares already-remapped inputs, so an earlier
 * bypass can expose a later identity without iteration. It preserves the occurrence phase and
 * neither discovers or creates Tensor constants nor evaluates arithmetic.</p>
 */
final class ForwardExactArithmeticRewriting {
    private ForwardExactArithmeticRewriting() {}

    /**
     * Rewrites the guarded seven-rule set in either graph phase while rebuilding changed
     * graph-local IDs.
     *
     * @param graph the non-null, successfully validated immutable graph in topological order; it
     *     is not mutated
     * @return the exact {@code graph} reference when no occurrence is bypassed; otherwise a new
     *     non-null immutable graph with dense input-first IDs, preserved ordered boundaries, and
     *     exact retained operation, descriptor, and phase references
     * @throws NullPointerException if {@code graph} is {@code null}
     */
    static CompiledGraphModel rewrite(CompiledGraphModel graph) {
        return rebuild(graph).graph();
    }

    /**
     * Rewrites the graph and remaps derivative metadata when IDs change.
     *
     * @param derivatives non-null metadata owning the input graph
     * @return non-null rewritten graph and matching metadata
     */
    static Result rewrite(DerivativeGraphMetadata derivatives) {
        Objects.requireNonNull(derivatives, "derivatives");
        Rebuild rebuild = rebuild(derivatives.graph());
        if (rebuild.graph() == derivatives.graph()) {
            return new Result(rebuild.graph(), derivatives);
        }
        return new Result(
                rebuild.graph(),
                DerivativeGraphMetadata.remap(
                        derivatives, rebuild.graph(), rebuild.sourceNodeIds()));
    }

    private static Rebuild rebuild(CompiledGraphModel graph) {
        Objects.requireNonNull(graph, "graph");

        Map<ValueId, GraphValue> originalValues = valuesById(graph);
        Set<ValueId> graphOutputs = new HashSet<>(graph.outputs());
        Map<ValueId, ValueId> valueRemapping = new HashMap<>();
        List<GraphValue> values = new ArrayList<>(graph.values().size());
        List<ValueId> inputs = new ArrayList<>(graph.inputs().size());
        long nextValueId = 0;
        for (ValueId input : graph.inputs()) {
            ValueId rebuiltInput = new ValueId(nextValueId++);
            valueRemapping.put(input, rebuiltInput);
            inputs.add(rebuiltInput);
            values.add(new GraphValue(rebuiltInput, originalValues.get(input).descriptor()));
        }

        List<CompiledNode> nodes = new ArrayList<>(graph.nodes().size());
        List<NodeId> sourceNodeIds = new ArrayList<>(graph.nodes().size());
        Map<NodeId, GraphPhase> phases = new LinkedHashMap<>();
        long nextNodeId = 0;
        boolean changed = false;
        for (CompiledNode node : graph.nodes()) {
            List<ValueId> remappedInputs = remap(node.inputs(), valueRemapping);
            ValueId bypassInput = bypassInput(
                    graph, node, remappedInputs, originalValues, graphOutputs);
            if (bypassInput != null) {
                valueRemapping.put(node.outputs().getFirst(), bypassInput);
                changed = true;
                continue;
            }

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
                    rebuiltNode, node.operation(), remappedInputs, rebuiltOutputs));
            sourceNodeIds.add(node.id());
            phases.put(rebuiltNode, graph.nodePhases().get(node.id()));
        }

        if (!changed) {
            return new Rebuild(graph, List.copyOf(sourceNodeIds));
        }
        return new Rebuild(
                new CompiledGraphModel(
                        values, nodes, inputs, remap(graph.outputs(), valueRemapping), phases),
                List.copyOf(sourceNodeIds));
    }

    private static ValueId bypassInput(
            CompiledGraphModel graph,
            CompiledNode node,
            List<ValueId> remappedInputs,
            Map<ValueId, GraphValue> originalValues,
            Set<ValueId> graphOutputs) {
        if (node.outputs().size() != 1
                || graphOutputs.contains(node.outputs().getFirst())) {
            return null;
        }

        int inputIndex = selectedInputIndex(node.operation(), remappedInputs, originalValues, node);
        if (inputIndex < 0) {
            return null;
        }
        TensorDescriptor inputDescriptor =
                originalValues.get(node.inputs().get(inputIndex)).descriptor();
        TensorDescriptor outputDescriptor =
                originalValues.get(node.outputs().getFirst()).descriptor();
        if (outputDescriptor.requiresGrad() || !outputDescriptor.equals(inputDescriptor)) {
            return null;
        }
        return remappedInputs.get(inputIndex);
    }

    private static int selectedInputIndex(
            Operation operation,
            List<ValueId> remappedInputs,
            Map<ValueId, GraphValue> originalValues,
            CompiledNode node) {
        if (operation.kind() instanceof BinaryArithmeticKind kind) {
            return isDuplicateExtrema(operation, kind, remappedInputs) ? 0 : -1;
        }
        if (!(operation.kind() instanceof ScalarElementwiseKind kind)
                || !(operation.attrs() instanceof ScalarValueAttrs attrs)
                || remappedInputs.size() != 1) {
            return -1;
        }

        DataType inputType = originalValues.get(node.inputs().getFirst()).descriptor().dataType();
        DataType outputType = originalValues.get(node.outputs().getFirst()).descriptor().dataType();
        ScalarValue value = attrs.value();
        if (value.dataType() != inputType || value.dataType() != outputType) {
            return -1;
        }
        return selectedScalarIdentity(kind, value) ? 0 : -1;
    }

    private static boolean isDuplicateExtrema(
            Operation operation, BinaryArithmeticKind kind, List<ValueId> remappedInputs) {
        return operation.attrs() == NoOperationAttrs.INSTANCE
                && (kind == BinaryArithmeticKind.MIN || kind == BinaryArithmeticKind.MAX)
                && remappedInputs.size() == 2
                && remappedInputs.get(0).equals(remappedInputs.get(1));
    }

    private static boolean selectedScalarIdentity(
            ScalarElementwiseKind kind, ScalarValue value) {
        return switch (kind) {
            case MUL -> isPositiveOne(value);
            case DIV, POW -> value.dataType().isFloating() && isPositiveOne(value);
            case ADD, SUB -> value.dataType().isIntegral() && isZero(value);
            case MIN, MAX, CLAMP -> false;
        };
    }

    private static boolean isPositiveOne(ScalarValue value) {
        return switch (value.dataType()) {
            case FLOAT64 -> value.float64Value() == 1.0d;
            case FLOAT32 -> value.float32Value() == 1.0f;
            case BFLOAT16 -> value.bfloat16Bits() == (short) 0x3F80;
            case INT32 -> value.int32Value() == 1;
            case INT64 -> value.int64Value() == 1L;
            case BOOL -> false;
        };
    }

    private static boolean isZero(ScalarValue value) {
        return switch (value.dataType()) {
            case INT32 -> value.int32Value() == 0;
            case INT64 -> value.int64Value() == 0L;
            case FLOAT64, FLOAT32, BFLOAT16, BOOL -> false;
        };
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

    /**
     * Rewritten graph and its matching derivative metadata.
     *
     * @param graph non-null rewritten graph
     * @param derivatives non-null matching metadata
     */
    record Result(CompiledGraphModel graph, DerivativeGraphMetadata derivatives) {}

    private record Rebuild(CompiledGraphModel graph, List<NodeId> sourceNodeIds) {}
}
