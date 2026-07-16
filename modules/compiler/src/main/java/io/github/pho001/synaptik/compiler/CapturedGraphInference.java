package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.normalization.*;
import io.github.pho001.synaptik.model.operation.ordering.*;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.random.*;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Performs binding-free semantic verification of a structurally closed captured graph.
 *
 * <p>The pass independently derives every operation occurrence's complete output descriptors,
 * compares them with the stored graph descriptors, and retains only Shape obligations that the
 * current immutable model facts cannot prove or disprove. It neither rewrites the graph nor
 * reconstructs public Tensor expressions.
 */
final class CapturedGraphInference {
    private CapturedGraphInference() {}

    /**
     * Infers and validates every operation occurrence in deterministic graph order.
     *
     * @param graph non-null structurally valid captured graph, retained by exact reference
     * @return an immutable successful result that retains the exact {@code graph} reference and
     *     contains only unresolved constraints; never {@code null}
     * @throws NullPointerException if {@code graph} is null
     * @throws IllegalArgumentException if an occurrence violates its semantic descriptor contract
     */
    static ValidatedGraph inferAndValidate(CompiledGraphModel graph) {
        return inferAndValidate(CompileTimeConstantGraph.withoutConstants(graph));
    }

    /**
     * Validates a graph while retaining its already validated immutable constant-source roles.
     *
     * @param constantGraph non-null immutable structural graph and exact source facts
     * @return an immutable successful result retaining the exact sidecar and unresolved graph
     *     constraints; never {@code null}
     * @throws NullPointerException if {@code constantGraph} is null
     * @throws IllegalArgumentException if graph inference or descriptor validation fails
     */
    static ValidatedGraph inferAndValidate(CompileTimeConstantGraph constantGraph) {
        Objects.requireNonNull(constantGraph, "constantGraph");
        CompiledGraphModel graph = constantGraph.graph();
        Map<ValueId, GraphValue> values = new HashMap<>();
        for (GraphValue value : graph.values()) values.put(value.id(), value);
        List<DeferredGraphConstraint> deferred = new ArrayList<>();
        for (int index = 0; index < graph.nodes().size(); index++) {
            CompiledNode node = graph.nodes().get(index);
            List<TensorDescriptor> inputs = descriptors(node.inputs(), values);
            List<TensorDescriptor> stored = descriptors(node.outputs(), values);
            String context = context(index, node);
            InferenceResult result;
            try {
                result = dispatch(node.operation(), inputs, stored.size(), context);
            } catch (ArithmeticException exception) {
                throw new IllegalArgumentException(
                        context + "descriptor derivation failed: " + exception.getMessage(), exception);
            } catch (IllegalArgumentException exception) {
                if (exception.getMessage() != null && exception.getMessage().startsWith(context)) throw exception;
                throw new IllegalArgumentException(context + exception.getMessage(), exception);
            }
            if (result.outputs().size() != stored.size()) {
                throw new IllegalArgumentException(context + "expected output count "
                        + result.outputs().size() + ", stored=" + stored.size());
            }
            for (int output = 0; output < stored.size(); output++) {
                TensorDescriptor expected = result.outputs().get(output);
                if (!expected.equals(stored.get(output))) {
                    throw new IllegalArgumentException(context + "output[" + output + "] "
                            + node.outputs().get(output) + " expected=" + expected
                            + ", stored=" + stored.get(output));
                }
            }
            for (ConstraintRequest request : result.constraints()) {
                ProofStatus proof = GraphPredicateProof.evaluate(request.predicate());
                if (proof == ProofStatus.DISPROVEN) {
                    throw new IllegalArgumentException(context + "constraint " + request.subject()
                            + " failed: " + request.predicate());
                }
                if (proof == ProofStatus.DEFERRED) {
                    deferred.add(new DeferredGraphConstraint(node.id(), request.subject(), request.predicate()));
                }
            }
        }
        return new ValidatedGraph(constantGraph, deferred);
    }

    private static List<TensorDescriptor> descriptors(List<ValueId> ids, Map<ValueId, GraphValue> values) {
        List<TensorDescriptor> result = new ArrayList<>(ids.size());
        for (ValueId id : ids) result.add(values.get(id).descriptor());
        return result;
    }

    private static String context(int index, CompiledNode node) {
        OperationKind kind = node.operation().kind();
        return "nodes[" + index + "] " + node.id() + " " + kind.getClass().getName()
                + "." + kind.name() + ": ";
    }

    private static InferenceResult dispatch(
            Operation operation, List<TensorDescriptor> inputs, int outputCount, String context) {
        OperationKind kind = operation.kind();
        if (kind instanceof BinaryArithmeticKind || kind instanceof ScalarElementwiseKind
                || kind instanceof UnaryElementwiseKind || kind instanceof BinaryComparisonKind
                || kind instanceof BooleanLogicalKind || kind instanceof FloatingClassificationKind
                || kind instanceof WhereSelectionKind || kind instanceof CastKind) {
            return ElementwiseInference.infer(operation, inputs);
        }
        if (kind instanceof AggregateReductionKind || kind instanceof CumulativeScanKind
                || kind instanceof SoftmaxKind || kind instanceof LayerNormKind
                || kind instanceof RmsNormKind || kind instanceof BatchNormKind
                || kind instanceof OrderingKind || kind instanceof TopKKind) {
            return ReductionNormalizationInference.infer(operation, inputs);
        }
        if (kind instanceof SelectKind || kind instanceof AxisGatherKind || kind instanceof AxisScatterKind
                || kind instanceof GatherNdKind || kind instanceof ScatterNdKind || kind instanceof OneHotKind) {
            return IndexingInference.infer(operation, inputs);
        }
        if (kind instanceof ContiguousKind || kind instanceof ShapeTransformKind
                || kind instanceof AxisTransformKind || kind instanceof SliceKind
                || kind instanceof PadKind || kind instanceof TileKind
                || kind instanceof TensorCompositionKind || kind instanceof WindowTransformKind) {
            return LayoutInference.infer(operation, inputs);
        }
        if (kind instanceof MatmulKind || kind instanceof ScaledDotProductAttentionKind
                || kind instanceof Conv2dKind || kind instanceof Pool2dKind || kind instanceof LossKind
                || kind instanceof GraphRngKind || kind instanceof DropoutKind) {
            return StructuredOperationInference.infer(operation, inputs, outputCount);
        }
        throw new IllegalArgumentException(context + "unsupported operation kind");
    }

    /**
     * Associates one family-derived predicate with the semantic role reported on failure or
     * deferral.
     *
     * @param subject non-null semantic role text used in deterministic diagnostics
     * @param predicate non-null binding-free predicate to evaluate
     */
    record ConstraintRequest(String subject, GraphPredicate predicate) {
        /**
         * Creates an occurrence-local candidate constraint.
         *
         * @param subject non-null semantic role text
         * @param predicate non-null predicate to evaluate
         * @throws NullPointerException if either argument is {@code null}
         */
        ConstraintRequest { Objects.requireNonNull(subject); Objects.requireNonNull(predicate); }
    }

    /**
     * Holds the complete derived output descriptors and ordered candidate constraints for one
     * operation occurrence.
     *
     * @param outputs non-null derived descriptors in operation output order; snapshot on creation
     * @param constraints non-null candidate constraints in deterministic rule order; snapshot on
     *     creation
     */
    record InferenceResult(List<TensorDescriptor> outputs, List<ConstraintRequest> constraints) {
        /**
         * Creates an immutable inference result by snapshotting both ordered lists.
         *
         * @param outputs non-null output descriptors without null elements
         * @param constraints non-null candidate constraints without null elements
         * @throws NullPointerException if a list or contained element is {@code null}
         */
        InferenceResult {
            outputs = List.copyOf(outputs); constraints = List.copyOf(constraints);
        }
        /**
         * Creates an unconstrained result from the supplied ordered output descriptors.
         *
         * @param outputs non-null output descriptor array without null elements
         * @return an immutable result whose constraint list is empty; never {@code null}
         * @throws NullPointerException if {@code outputs} or an element is {@code null}
         */
        static InferenceResult of(TensorDescriptor... outputs) {
            return new InferenceResult(List.of(outputs), List.of());
        }

        /**
         * Creates a result with ordered output descriptors and ordered candidate constraints.
         *
         * @param outputs non-null ordered output descriptors; snapshot on creation
         * @param constraints non-null constraint array without null elements
         * @return the immutable derived result; never {@code null}
         * @throws NullPointerException if an argument or contained element is {@code null}
         */
        static InferenceResult constrained(List<TensorDescriptor> outputs, ConstraintRequest... constraints) {
            return new InferenceResult(outputs, List.of(constraints));
        }
    }
}
