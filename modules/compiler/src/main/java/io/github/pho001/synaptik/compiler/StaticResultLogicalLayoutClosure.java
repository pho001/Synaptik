package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Applies shared computed logical-layout completion and closes explicit splat inputs.
 *
 * <p>Model expression construction deliberately leaves most operation results layout-unresolved.
 * After final graph optimization fixes the surviving topology and independent inference validates
 * ordinary semantics, {@link ScanBody#closeOrdinaryLayouts} owns the materialized-result
 * classification, exact family view recipes and qualified noncontiguous reshape materialization.
 * This pass supplies current operands in topological order and preserves the captured graph and
 * its sidecars. Unknown and unproved layouts remain unresolved.
 *
 * <p>An explicit compile-time splat graph input may also use canonical geometry because it has no
 * caller binding whose descriptor must remain exact. Caller-bindable inputs, dynamic shapes, and
 * already resolved layouts are preserved. This pass selects no physical buffer, representation,
 * backend, route, or executable.
 */
final class StaticResultLogicalLayoutClosure {
    /** Prevents construction of this stateless transformation owner. */
    private StaticResultLogicalLayoutClosure() {}

    /**
     * Closes eligible descriptors without changing graph structure or sidecar membership.
     *
     * <p>If no descriptor qualifies, the exact input is returned. Otherwise only changed {@link
     * GraphValue} objects and graph-owning immutable sidecars are rebuilt. Value and boundary order,
     * IDs, exact node and operation references, phases, constants, bindable Tensor identities,
     * constraints, derivative-order values, and containing-Scan activity plans are preserved.
     * Local body graph references and structural availability/merge/temporal meanings are unchanged.
     *
     * @param validatedGraph non-null final optimized, validated graph state after specialized
     *     descriptor closures; it is not mutated
     * @return the exact input when unchanged, otherwise a non-null validated result containing
     *     canonical materialized-result layouts and exact family-derived views where proved
     * @throws NullPointerException if {@code validatedGraph} is {@code null}
     * @throws IllegalArgumentException if checked canonical or affine layout arithmetic overflows;
     *     the failure identifies the affected node and canonical outputs or compile-time splat
     *     input and retains the arithmetic failure as its cause
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
            changed |= closeComputedResults(selectedValues, node);
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
                rebuilt, validatedGraph.derivatives().derivativeOrderByNode(),
                validatedGraph.derivatives().scanActivityByNode());
        return new ValidatedGraph(constantGraph, validatedGraph.constraints(), derivatives);
    }

    /**
     * Applies the shared Model decision to current operands and complete canonical results.
     * Only descriptor replacements are retained; graph and sidecar orchestration stays here.
     * @param selectedValues current call-local descriptor table
     * @param node exact validated node in topological order
     * @return whether any result acquired proved geometry
     * @throws IllegalArgumentException if checked geometry overflows, retaining node/result context
     */
    private static boolean closeComputedResults(
            Map<ValueId, GraphValue> selectedValues, CompiledNode node) {
        boolean eligibleOutput = false;
        for (ValueId id : node.outputs()) {
            TensorDescriptor output = selectedValues.get(id).descriptor();
            if (output.shape().isFullyStatic() && output.layout().isEmpty()) {
                eligibleOutput = true;
                break;
            }
        }
        if (!eligibleOutput) return false;
        List<TensorDescriptor> inputs = new ArrayList<>(node.inputs().size());
        for (ValueId id : node.inputs()) inputs.add(selectedValues.get(id).descriptor());
        List<TensorDescriptor> outputs = new ArrayList<>(node.outputs().size());
        for (ValueId id : node.outputs()) outputs.add(selectedValues.get(id).descriptor());
        List<TensorDescriptor> closed;
        try {
            closed = ScanBody.closeOrdinaryLayouts(node.operation(), inputs, outputs);
        } catch (ArithmeticException failure) {
            throw new IllegalArgumentException("cannot close static result logical layouts for "
                    + node.id() + " outputs " + node.outputs(), failure);
        }
        boolean changed = false;
        for (int index = 0; index < outputs.size(); index++) {
            if (closed.get(index) == outputs.get(index)) continue;
            ValueId id = node.outputs().get(index);
            selectedValues.put(id, new GraphValue(id, closed.get(index)));
            changed = true;
        }
        return changed;
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
