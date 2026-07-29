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
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
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
 * Folds a closed set of exact logical-splat operations in one whole-graph topological scan.
 *
 * <p>The selected set is BOOL {@code NOT}/{@code AND}/{@code OR}, signed-integral binary
 * {@code ADD}/{@code SUB}/{@code MUL}/{@code MIN}/{@code MAX}, all signed-integral binary
 * comparisons, and same-typed signed-integral scalar {@code ADD}/{@code SUB}/{@code MUL}/
 * {@code MIN}/{@code MAX}. Floating and BFLOAT16 evaluation, casts, other operations, graph-output
 * producers, gradient-eligible results, and multi-output occurrences remain unchanged. An
 * otherwise eligible occurrence may be {@link GraphPhase#FORWARD FORWARD} or
 * {@link GraphPhase#BACKWARD BACKWARD}; its phase is not a constant-folding policy.</p>
 *
 * <p>A folded result becomes a new fixed structural source with one scalar payload. The pass does
 * not enumerate elements, read Tensor storage, allocate physical storage, intern constants, or
 * iterate to a fixed point.</p>
 */
final class ForwardConstantFolding {
    private ForwardConstantFolding() {}

    /**
     * Folds selected exact occurrences in either phase and propagates their splats to later nodes
     * in the same scan.
     *
     * @param constantGraph non-null successfully validated immutable graph and source facts; it is
     *     not mutated
     * @return the exact {@code constantGraph} when no occurrence folds; otherwise a non-null
     *     immutable result with original inputs first, synthetic constants in fold order, dense
     *     IDs, and retained boundaries, operations, descriptors, and phases
     * @throws NullPointerException if {@code constantGraph} is null
     */
    static CompileTimeConstantGraph fold(CompileTimeConstantGraph constantGraph) {
        Objects.requireNonNull(constantGraph, "constantGraph");
        Map<NodeId, CompileTimeConstantGraph.Splat> folded = foldedNodes(constantGraph);
        if (folded.isEmpty()) {
            return constantGraph;
        }
        return rebuild(
                constantGraph,
                folded,
                valuesById(constantGraph.graph())).constantGraph();
    }

    /**
     * Folds selected exact occurrences while preserving derivative orders.
     *
     * @param constantGraph non-null graph and source facts
     * @param derivatives non-null metadata owning the exact graph
     * @return non-null folded graph/source facts and matching metadata
     */
    static Result fold(
            CompileTimeConstantGraph constantGraph,
            DerivativeGraphMetadata derivatives) {
        Objects.requireNonNull(constantGraph, "constantGraph");
        Objects.requireNonNull(derivatives, "derivatives");
        if (derivatives.graph() != constantGraph.graph()) {
            throw new IllegalArgumentException(
                    "derivatives graph must be the exact graph being folded");
        }
        Map<NodeId, CompileTimeConstantGraph.Splat> folded = foldedNodes(constantGraph);
        if (folded.isEmpty()) {
            return new Result(constantGraph, derivatives);
        }
        Rebuild rebuild = rebuild(
                constantGraph, folded, valuesById(constantGraph.graph()));
        return new Result(
                rebuild.constantGraph(),
                DerivativeGraphMetadata.remap(
                        derivatives,
                        rebuild.constantGraph().graph(),
                        rebuild.sourceNodeIds()));
    }

    private static Map<NodeId, CompileTimeConstantGraph.Splat> foldedNodes(
            CompileTimeConstantGraph constantGraph) {
        CompiledGraphModel graph = constantGraph.graph();
        Map<ValueId, GraphValue> originalValues = valuesById(graph);
        Set<ValueId> graphOutputs = new HashSet<>(graph.outputs());
        Map<ValueId, CompileTimeConstantGraph.Splat> propagated =
                new HashMap<>(constantGraph.constants());
        Map<NodeId, CompileTimeConstantGraph.Splat> folded = new HashMap<>();

        for (CompiledNode node : graph.nodes()) {
            CompileTimeConstantGraph.Splat result = evaluate(
                    graph, node, propagated, originalValues, graphOutputs);
            if (result != null) {
                folded.put(node.id(), result);
                propagated.put(node.outputs().getFirst(), result);
            }
        }
        return folded;
    }

    private static CompileTimeConstantGraph.Splat evaluate(
            CompiledGraphModel graph,
            CompiledNode node,
            Map<ValueId, CompileTimeConstantGraph.Splat> propagated,
            Map<ValueId, GraphValue> values,
            Set<ValueId> graphOutputs) {
        if (node.outputs().size() != 1
                || graphOutputs.contains(node.outputs().getFirst())) {
            return null;
        }
        TensorDescriptor outputDescriptor = values.get(node.outputs().getFirst()).descriptor();
        if (outputDescriptor.requiresGrad()) {
            return null;
        }

        List<CompileTimeConstantGraph.Splat> inputs = new ArrayList<>(node.inputs().size());
        for (ValueId input : node.inputs()) {
            CompileTimeConstantGraph.Splat splat = propagated.get(input);
            if (splat == null
                    || splat.value().dataType() != values.get(input).descriptor().dataType()) {
                return null;
            }
            inputs.add(splat);
        }

        ScalarValue result = evaluateSelected(node.operation(), inputs, outputDescriptor.dataType());
        if (result == null || result.dataType() != outputDescriptor.dataType()) {
            return null;
        }
        return new CompileTimeConstantGraph.Splat(result);
    }

    private static ScalarValue evaluateSelected(
            Operation operation,
            List<CompileTimeConstantGraph.Splat> inputs,
            DataType outputType) {
        if (operation.kind() instanceof BooleanLogicalKind kind
                && operation.attrs() == NoOperationAttrs.INSTANCE) {
            return evaluateLogical(kind, inputs);
        }
        if (operation.kind() instanceof BinaryArithmeticKind kind
                && operation.attrs() == NoOperationAttrs.INSTANCE) {
            return evaluateBinaryArithmetic(kind, inputs, outputType);
        }
        if (operation.kind() instanceof BinaryComparisonKind kind
                && operation.attrs() == NoOperationAttrs.INSTANCE) {
            return evaluateComparison(kind, inputs, outputType);
        }
        if (operation.kind() instanceof ScalarElementwiseKind kind
                && operation.attrs() instanceof ScalarValueAttrs attrs) {
            return evaluateScalar(kind, inputs, attrs.value(), outputType);
        }
        return null;
    }

    private static ScalarValue evaluateLogical(
            BooleanLogicalKind kind, List<CompileTimeConstantGraph.Splat> inputs) {
        int requiredInputs = kind == BooleanLogicalKind.NOT ? 1 : 2;
        if (inputs.size() != requiredInputs
                || inputs.stream().anyMatch(input -> input.value().dataType() != DataType.BOOL)) {
            return null;
        }
        boolean left = inputs.getFirst().value().booleanValue();
        return switch (kind) {
            case NOT -> ScalarValue.bool(!left);
            case AND -> ScalarValue.bool(left && inputs.get(1).value().booleanValue());
            case OR -> ScalarValue.bool(left || inputs.get(1).value().booleanValue());
        };
    }

    private static ScalarValue evaluateBinaryArithmetic(
            BinaryArithmeticKind kind,
            List<CompileTimeConstantGraph.Splat> inputs,
            DataType outputType) {
        if (inputs.size() != 2
                || !inputs.get(0).value().dataType().isIntegral()
                || !inputs.get(1).value().dataType().isIntegral()
                || !outputType.isIntegral()) {
            return null;
        }
        if (outputType == DataType.INT32) {
            if (inputs.get(0).value().dataType() != DataType.INT32
                    || inputs.get(1).value().dataType() != DataType.INT32) {
                return null;
            }
            int left = inputs.get(0).value().int32Value();
            int right = inputs.get(1).value().int32Value();
            return switch (kind) {
                case ADD -> ScalarValue.int32(left + right);
                case SUB -> ScalarValue.int32(left - right);
                case MUL -> ScalarValue.int32(left * right);
                case MIN -> ScalarValue.int32(Math.min(left, right));
                case MAX -> ScalarValue.int32(Math.max(left, right));
                case DIV, POW -> null;
            };
        }
        if (outputType != DataType.INT64) {
            return null;
        }
        long left = integralLong(inputs.get(0).value());
        long right = integralLong(inputs.get(1).value());
        return switch (kind) {
            case ADD -> ScalarValue.int64(left + right);
            case SUB -> ScalarValue.int64(left - right);
            case MUL -> ScalarValue.int64(left * right);
            case MIN -> ScalarValue.int64(Math.min(left, right));
            case MAX -> ScalarValue.int64(Math.max(left, right));
            case DIV, POW -> null;
        };
    }

    private static ScalarValue evaluateComparison(
            BinaryComparisonKind kind,
            List<CompileTimeConstantGraph.Splat> inputs,
            DataType outputType) {
        if (inputs.size() != 2
                || outputType != DataType.BOOL
                || !inputs.get(0).value().dataType().isIntegral()
                || !inputs.get(1).value().dataType().isIntegral()) {
            return null;
        }
        long left = integralLong(inputs.get(0).value());
        long right = integralLong(inputs.get(1).value());
        return ScalarValue.bool(switch (kind) {
            case GREATER_THAN -> left > right;
            case GREATER_OR_EQUAL -> left >= right;
            case LESS_THAN -> left < right;
            case LESS_OR_EQUAL -> left <= right;
            case EQUAL -> left == right;
            case NOT_EQUAL -> left != right;
        });
    }

    private static ScalarValue evaluateScalar(
            ScalarElementwiseKind kind,
            List<CompileTimeConstantGraph.Splat> inputs,
            ScalarValue right,
            DataType outputType) {
        if (inputs.size() != 1
                || !outputType.isIntegral()
                || inputs.getFirst().value().dataType() != outputType
                || right.dataType() != outputType) {
            return null;
        }
        if (outputType == DataType.INT32) {
            int leftValue = inputs.getFirst().value().int32Value();
            int rightValue = right.int32Value();
            return switch (kind) {
                case ADD -> ScalarValue.int32(leftValue + rightValue);
                case SUB -> ScalarValue.int32(leftValue - rightValue);
                case MUL -> ScalarValue.int32(leftValue * rightValue);
                case MIN -> ScalarValue.int32(Math.min(leftValue, rightValue));
                case MAX -> ScalarValue.int32(Math.max(leftValue, rightValue));
                case DIV, POW, CLAMP -> null;
            };
        }
        long leftValue = inputs.getFirst().value().int64Value();
        long rightValue = right.int64Value();
        return switch (kind) {
            case ADD -> ScalarValue.int64(leftValue + rightValue);
            case SUB -> ScalarValue.int64(leftValue - rightValue);
            case MUL -> ScalarValue.int64(leftValue * rightValue);
            case MIN -> ScalarValue.int64(Math.min(leftValue, rightValue));
            case MAX -> ScalarValue.int64(Math.max(leftValue, rightValue));
            case DIV, POW, CLAMP -> null;
        };
    }

    private static long integralLong(ScalarValue value) {
        return value.dataType() == DataType.INT32 ? value.int32Value() : value.int64Value();
    }

    private static Rebuild rebuild(
            CompileTimeConstantGraph source,
            Map<NodeId, CompileTimeConstantGraph.Splat> folded,
            Map<ValueId, GraphValue> originalValues) {
        CompiledGraphModel graph = source.graph();
        Map<ValueId, ValueId> remapping = new HashMap<>();
        Map<ValueId, CompileTimeConstantGraph.Splat> constants = new HashMap<>();
        List<GraphValue> values = new ArrayList<>(graph.values().size());
        List<ValueId> inputs = new ArrayList<>(graph.inputs().size() + folded.size());
        long nextValueId = 0;

        for (ValueId input : graph.inputs()) {
            ValueId rebuilt = new ValueId(nextValueId++);
            remapping.put(input, rebuilt);
            inputs.add(rebuilt);
            values.add(new GraphValue(rebuilt, originalValues.get(input).descriptor()));
            CompileTimeConstantGraph.Splat splat = source.constants().get(input);
            if (splat != null) {
                constants.put(rebuilt, splat);
            }
        }
        for (CompiledNode node : graph.nodes()) {
            CompileTimeConstantGraph.Splat splat = folded.get(node.id());
            if (splat == null) {
                continue;
            }
            ValueId originalOutput = node.outputs().getFirst();
            ValueId synthetic = new ValueId(nextValueId++);
            remapping.put(originalOutput, synthetic);
            inputs.add(synthetic);
            values.add(new GraphValue(synthetic, originalValues.get(originalOutput).descriptor()));
            constants.put(synthetic, splat);
        }

        List<CompiledNode> nodes = new ArrayList<>(graph.nodes().size() - folded.size());
        List<NodeId> sourceNodeIds =
                new ArrayList<>(graph.nodes().size() - folded.size());
        Map<NodeId, GraphPhase> phases = new LinkedHashMap<>();
        long nextNodeId = 0;
        for (CompiledNode node : graph.nodes()) {
            if (folded.containsKey(node.id())) {
                continue;
            }
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
            sourceNodeIds.add(node.id());
            phases.put(rebuiltNode, graph.nodePhases().get(node.id()));
        }

        CompiledGraphModel rebuilt = new CompiledGraphModel(
                values, nodes, inputs, remap(graph.outputs(), remapping), phases);
        return new Rebuild(
                new CompileTimeConstantGraph(rebuilt, constants),
                List.copyOf(sourceNodeIds));
    }

    private static Map<ValueId, GraphValue> valuesById(CompiledGraphModel graph) {
        Map<ValueId, GraphValue> result = new HashMap<>();
        for (GraphValue value : graph.values()) {
            result.put(value.id(), value);
        }
        return result;
    }

    private static List<ValueId> remap(
            List<ValueId> original, Map<ValueId, ValueId> remapping) {
        List<ValueId> result = new ArrayList<>(original.size());
        for (ValueId value : original) {
            result.add(remapping.get(value));
        }
        return result;
    }

    /**
     * Folded graph/source facts and their matching derivative metadata.
     *
     * @param constantGraph non-null folded graph and source facts
     * @param derivatives non-null matching metadata
     */
    record Result(
            CompileTimeConstantGraph constantGraph,
            DerivativeGraphMetadata derivatives) {}

    private record Rebuild(
            CompileTimeConstantGraph constantGraph,
            List<NodeId> sourceNodeIds) {}
}
