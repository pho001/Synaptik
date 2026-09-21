package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Closes final logical layouts required by exact unary {@code NEG} occurrences.
 *
 * <p>For an exact one-input, one-output {@link UnaryElementwiseKind#NEG} occurrence with
 * {@link NoOperationAttrs#INSTANCE}, the pass assigns canonical contiguous logical layout to a
 * fully static unresolved output. It applies the same closure to the direct input only when that
 * value is both a graph input and an explicitly registered compile-time splat. Other operations,
 * indirect constants, caller-bindable inputs, dynamic shapes, and resolved descriptors remain
 * unchanged. This is a bounded final Compiler descriptor rule, not general elementwise layout
 * inference or physical/backend state.</p>
 */
final class NegLogicalLayoutClosure {
    /** Prevents construction of this stateless transformation owner. */
    private NegLogicalLayoutClosure() {}

    /**
     * Closes eligible direct splat-input and result descriptors without changing graph structure.
     *
     * <p>If no descriptor qualifies, the exact input object is returned. Otherwise only changed
     * {@link GraphValue} objects are replaced. Value and boundary order, IDs, exact node and
     * operation references, phases, constants, bindable Tensor identities, constraints, and
     * derivative-order values are preserved. Derivative metadata is rebuilt only to own the
     * rebuilt exact graph reference.</p>
     *
     * @param validatedGraph non-null final optimized, validated, published-constant-closed, and
     *     convolution-closed graph state; it is not mutated
     * @return the exact input if no descriptor changes, otherwise a non-null validated result
     *     containing only eligible canonical logical-layout replacements
     * @throws NullPointerException if {@code validatedGraph} is {@code null}
     * @throws IllegalArgumentException if checked canonical-layout arithmetic overflows; the
     *     failure identifies the NEG node, value, and input or output role and retains the
     *     arithmetic failure as its cause
     */
    static ValidatedGraph close(ValidatedGraph validatedGraph) {
        java.util.Objects.requireNonNull(validatedGraph, "validatedGraph");

        CompiledGraphModel graph = validatedGraph.graph();
        Map<ValueId, GraphValue> selectedValues = new HashMap<>();
        for (GraphValue value : graph.values()) {
            selectedValues.put(value.id(), value);
        }
        Set<ValueId> graphInputs = new HashSet<>(graph.inputs());
        boolean changed = false;

        for (CompiledNode node : graph.nodes()) {
            if (node.operation().kind() != UnaryElementwiseKind.NEG
                    || node.operation().attrs() != NoOperationAttrs.INSTANCE
                    || node.inputs().size() != 1
                    || node.outputs().size() != 1) {
                continue;
            }

            ValueId inputId = node.inputs().getFirst();
            if (graphInputs.contains(inputId)
                    && validatedGraph.constants().containsKey(inputId)) {
                changed |= closeValue(selectedValues, node, inputId, "input");
            }
            changed |= closeValue(
                    selectedValues, node, node.outputs().getFirst(), "output");
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

    private static boolean closeValue(
            Map<ValueId, GraphValue> selectedValues,
            CompiledNode node,
            ValueId valueId,
            String role) {
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
                    "cannot close NEG logical layout for " + node.id() + " " + role + " "
                            + valueId,
                    failure);
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
