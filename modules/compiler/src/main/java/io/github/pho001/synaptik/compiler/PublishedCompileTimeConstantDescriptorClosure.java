package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * Closes unresolved logical layouts for final source-only published constant splats.
 *
 * <p>The final optimized graph alone determines eligibility. An eligible value is simultaneously
 * a graph input, explicit compile-time splat, graph output, and absent from every node input; its
 * shape is fully static and its layout remains unresolved. Closure supplies only the Model's
 * canonical contiguous logical layout. It does not add graph structure, caller bindings,
 * physical geometry, allocation, backend selection, preparation, or execution state.</p>
 */
final class PublishedCompileTimeConstantDescriptorClosure {
    private PublishedCompileTimeConstantDescriptorClosure() {}

    /**
     * Closes every eligible descriptor while preserving all other validated graph state.
     *
     * <p>If no descriptor is eligible, the exact input result is returned. Otherwise the rebuilt
     * graph retains value encounter order and IDs, every exact node reference, both ordered
     * boundaries, phases, constant splats, bindable Tensor identities, deferred constraints, and
     * derivative orders. Derivative metadata is rebuilt solely so that it owns the rebuilt exact
     * graph reference.</p>
     *
     * @param validatedGraph non-null final optimized and validated graph state; it is not mutated
     * @return the exact input when no descriptor changes, or a non-null validated result whose
     *     eligible descriptors have canonical contiguous logical layouts
     * @throws NullPointerException if {@code validatedGraph} is {@code null}
     * @throws IllegalArgumentException if canonical stride or referenced-span arithmetic
     *     overflows; the failure identifies the eligible value and retains the arithmetic cause
     */
    static ValidatedGraph close(ValidatedGraph validatedGraph) {
        java.util.Objects.requireNonNull(validatedGraph, "validatedGraph");

        CompiledGraphModel graph = validatedGraph.graph();
        Set<ValueId> inputs = new HashSet<>(graph.inputs());
        Set<ValueId> outputs = new HashSet<>(graph.outputs());
        Set<ValueId> consumed = new HashSet<>();
        for (CompiledNode node : graph.nodes()) {
            consumed.addAll(node.inputs());
        }

        List<GraphValue> values = new ArrayList<>(graph.values().size());
        boolean changed = false;
        for (GraphValue value : graph.values()) {
            TensorDescriptor descriptor = value.descriptor();
            if (!inputs.contains(value.id())
                    || !validatedGraph.constants().containsKey(value.id())
                    || !outputs.contains(value.id())
                    || consumed.contains(value.id())
                    || !descriptor.shape().isFullyStatic()
                    || descriptor.layout().isPresent()) {
                values.add(value);
                continue;
            }

            try {
                TensorDescriptor closed = new TensorDescriptor(
                        descriptor.dataType(),
                        descriptor.shape(),
                        Optional.of(LayoutDescriptor.contiguous(descriptor.shape())),
                        descriptor.requiresGrad());
                values.add(new GraphValue(value.id(), closed));
                changed = true;
            } catch (ArithmeticException failure) {
                throw new IllegalArgumentException(
                        "cannot close published compile-time constant descriptor for "
                                + value.id(),
                        failure);
            }
        }
        if (!changed) {
            return validatedGraph;
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
}
