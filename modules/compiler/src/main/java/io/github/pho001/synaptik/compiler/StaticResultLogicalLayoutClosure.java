package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormKind;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormKind;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Closes canonical logical layouts for proven materialized results and explicit splat inputs.
 *
 * <p>Model expression construction deliberately leaves most operation results layout-unresolved.
 * After final graph optimization has fixed the surviving topology, the allowlisted computation
 * families below may choose canonical contiguous logical result geometry because their outputs
 * are materialized values rather than storage views. {@link ContiguousKind#CONTIGUOUS} explicitly
 * requests the same geometry. Affine/view operations such as reshape, expand, permute, squeeze,
 * slice extraction, and scalar select are deliberately excluded: their output offset and strides
 * must be derived from the input or remain unresolved. Unknown future operation kinds therefore
 * fail closed instead of silently acquiring geometry.</p>
 *
 * <p>An explicit compile-time splat graph input may also use canonical geometry because it has no
 * caller binding whose descriptor must remain exact. Caller-bindable inputs, dynamic shapes, and
 * already resolved layouts are preserved. This pass selects no physical buffer, representation,
 * backend, route, or executable.</p>
 */
final class StaticResultLogicalLayoutClosure {
    /** Prevents construction of this stateless transformation owner. */
    private StaticResultLogicalLayoutClosure() {}

    /**
     * Closes eligible descriptors without changing graph structure or sidecar membership.
     *
     * <p>If no descriptor qualifies, the exact input is returned. Otherwise only changed
     * {@link GraphValue} objects and graph-owning immutable sidecars are rebuilt. Value and
     * boundary order, IDs, exact node and operation references, phases, constants, bindable Tensor
     * identities, constraints, and derivative-order values are preserved.</p>
     *
     * @param validatedGraph non-null final optimized, validated graph state after specialized
     *     descriptor closures; it is not mutated
     * @return the exact input when unchanged, otherwise a non-null validated result containing
     *     canonical logical layouts for every proven eligible value
     * @throws NullPointerException if {@code validatedGraph} is {@code null}
     * @throws IllegalArgumentException if checked canonical layout arithmetic overflows; the
     *     failure identifies the affected node output or compile-time splat input and retains the
     *     arithmetic failure as its cause
     */
    static ValidatedGraph close(ValidatedGraph validatedGraph) {
        java.util.Objects.requireNonNull(validatedGraph, "validatedGraph");

        CompiledGraphModel graph = validatedGraph.graph();
        Map<ValueId, GraphValue> selectedValues = new HashMap<>();
        for (GraphValue value : graph.values()) {
            selectedValues.put(value.id(), value);
        }
        boolean changed = false;

        for (ValueId valueId : validatedGraph.constants().keySet()) {
            changed |= closeValue(selectedValues, valueId,
                    "compile-time splat input " + valueId);
        }
        for (CompiledNode node : graph.nodes()) {
            if (!hasCanonicalContiguousResults(node.operation())) {
                continue;
            }
            for (int outputIndex = 0; outputIndex < node.outputs().size(); outputIndex++) {
                ValueId valueId = node.outputs().get(outputIndex);
                changed |= closeValue(selectedValues, valueId,
                        node.id() + " output[" + outputIndex + "] " + valueId);
            }
        }

        if (!changed) {
            return validatedGraph;
        }

        List<GraphValue> values = new ArrayList<>(graph.values().size());
        for (GraphValue value : graph.values()) {
            values.add(selectedValues.get(value.id()));
        }
        CompiledGraphModel rebuilt = new CompiledGraphModel(
                values,
                graph.nodes(),
                graph.inputs(),
                graph.outputs(),
                graph.nodePhases());
        CompileTimeConstantGraph constantGraph = new CompileTimeConstantGraph(
                rebuilt,
                validatedGraph.constantGraph().constants(),
                validatedGraph.constantGraph().bindableTensorIds());
        DerivativeGraphMetadata derivatives = new DerivativeGraphMetadata(
                rebuilt, validatedGraph.derivatives().derivativeOrderByNode());
        return new ValidatedGraph(constantGraph, validatedGraph.constraints(), derivatives);
    }
    /**
     * Returns whether every result of the exact operation family is a newly materialized logical
     * value whose unresolved static geometry may be chosen canonical contiguous.
     *
     * <p>The list is intentionally explicit. Structural views are absent; the functional
     * {@link SliceKind#SLICE_UPDATE} result is included separately from slice extraction. Adding a
     * new Model operation kind does not broaden this proof automatically.</p>
     */
    private static boolean hasCanonicalContiguousResults(Operation operation) {
        OperationKind kind = operation.kind();
        return kind instanceof ScaledDotProductAttentionKind
                || kind instanceof BinaryArithmeticKind
                || kind instanceof CastKind
                || kind instanceof FloatingClassificationKind
                || kind instanceof BinaryComparisonKind
                || kind instanceof BooleanLogicalKind
                || kind instanceof ScalarElementwiseKind
                || kind instanceof WhereSelectionKind
                || kind instanceof UnaryElementwiseKind
                || kind instanceof AxisGatherKind
                || kind instanceof AxisScatterKind
                || kind instanceof GatherNdKind
                || kind instanceof OneHotKind
                || kind instanceof ScatterNdKind
                || kind == ContiguousKind.CONTIGUOUS
                || kind instanceof PadKind
                || kind == SliceKind.SLICE_UPDATE
                || kind instanceof TensorCompositionKind
                || kind instanceof TileKind
                || kind instanceof WindowTransformKind
                || kind instanceof MatmulKind
                || kind instanceof LossKind
                || kind instanceof BatchNormKind
                || kind instanceof LayerNormKind
                || kind instanceof RmsNormKind
                || kind instanceof SoftmaxKind
                || kind instanceof OrderingKind
                || kind instanceof TopKKind
                || kind instanceof Pool2dKind
                || kind instanceof Pool3dKind
                || kind instanceof DropoutKind
                || kind instanceof GraphRngKind
                || kind instanceof RecurrentScanKind
                || kind instanceof AggregateReductionKind
                || kind instanceof CumulativeScanKind;
    }

    private static boolean closeValue(
            Map<ValueId, GraphValue> selectedValues, ValueId valueId, String context) {
        GraphValue value = selectedValues.get(valueId);
        TensorDescriptor descriptor = value.descriptor();
        if (!descriptor.shape().isFullyStatic() || descriptor.layout().isPresent()) {
            return false;
        }

        LayoutDescriptor layout;
        try {
            layout = LayoutDescriptor.contiguous(descriptor.shape());
        } catch (ArithmeticException failure) {
            throw new IllegalArgumentException(
                    "cannot close static result logical layout for " + context, failure);
        }
        TensorDescriptor closed = new TensorDescriptor(
                descriptor.dataType(),
                descriptor.shape(),
                Optional.of(layout),
                descriptor.requiresGrad());
        selectedValues.put(valueId, new GraphValue(valueId, closed));
        return true;
    }
}
