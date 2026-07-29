package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.ForwardPublicationBinding;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Associates ordered forward and gradient publication bindings with one exact compiled graph.
 *
 * <p>The two binding lists have distinct meanings. A forward binding identifies the requested
 * forward Tensor and its final graph value. A gradient binding identifies a differentiation
 * target and that target's final gradient value; it does not add gradient state to Tensor.
 * Membership is snapshotted while exact immutable binding references are retained.</p>
 *
 * <p>This output-only plan proves graph membership and boundary order. It contains no publication
 * policy, alias or copy decision, storage, delivery behavior, prepared state, or runtime state.</p>
 */
public final class PublicationPlan {
    private final CompiledGraphModel graph;
    private final List<ForwardPublicationBinding> forwardBindings;
    private final List<GradientPublicationBinding> gradientBindings;

    /**
     * Creates and validates publication bindings for one exact final graph.
     *
     * @param graph non-null final graph retained by exact reference
     * @param forwardBindings non-null ordered forward bindings to snapshot; Tensor and value IDs
     *     must each be unique and the values must equal the graph-output prefix
     * @param gradientBindings non-null ordered target-to-gradient bindings to snapshot; target IDs
     *     must be unique within each derivative order, while gradient values may repeat or equal
     *     forward values
     * @throws NullPointerException if a top-level argument or list element is {@code null}
     * @throws IllegalArgumentException if a binding violates graph membership, uniqueness, prefix,
     *     or complete boundary rules
     */
    PublicationPlan(
            CompiledGraphModel graph,
            List<ForwardPublicationBinding> forwardBindings,
            List<GradientPublicationBinding> gradientBindings) {
        this.graph = Objects.requireNonNull(graph, "graph");
        Objects.requireNonNull(forwardBindings, "forwardBindings");
        Objects.requireNonNull(gradientBindings, "gradientBindings");

        Set<ValueId> graphOutputIds = new HashSet<>(graph.outputs());
        Set<TensorId> forwardTensorIds = new HashSet<>();
        Set<ValueId> forwardValueIds = new HashSet<>();
        for (int index = 0; index < forwardBindings.size(); index++) {
            ForwardPublicationBinding binding = Objects.requireNonNull(
                    forwardBindings.get(index), "forwardBindings[" + index + "]");
            if (!graphOutputIds.contains(binding.valueId())) {
                throw new IllegalArgumentException(
                        "forwardBindings[" + index + "] is not a graph output");
            }
            if (!forwardTensorIds.add(binding.tensorId())) {
                throw new IllegalArgumentException(
                        "forwardBindings[" + index + "] duplicates an earlier TensorId");
            }
            if (!forwardValueIds.add(binding.valueId())) {
                throw new IllegalArgumentException(
                        "forwardBindings[" + index + "] duplicates an earlier forward value");
            }
            if (index >= graph.outputs().size()
                    || !binding.valueId().equals(graph.outputs().get(index))) {
                throw new IllegalArgumentException(
                        "forwardBindings[" + index + "] does not match graph output prefix");
            }
        }

        Set<TensorId> gradientTensorIds = new HashSet<>();
        int derivativeOrder = 1;
        int targetIndex = 0;
        for (int index = 0; index < gradientBindings.size(); index++) {
            GradientPublicationBinding binding = Objects.requireNonNull(
                    gradientBindings.get(index), "gradientBindings[" + index + "]");
            if (!graphOutputIds.contains(binding.valueId())) {
                throw new IllegalArgumentException(
                        "gradientBindings[" + index + "] is not a graph output");
            }
            if (binding.derivativeOrder() != derivativeOrder) {
                if (derivativeOrder != 1 || binding.derivativeOrder() != 2) {
                    throw new IllegalArgumentException(
                            "gradientBindings must be ordered by derivative order");
                }
                derivativeOrder = 2;
                targetIndex = 0;
                gradientTensorIds.clear();
            }
            if (binding.targetIndex() != targetIndex++) {
                throw new IllegalArgumentException(
                        "gradientBindings[" + index
                                + "] targetIndex does not match stage-local order");
            }
            if (!gradientTensorIds.add(binding.target())) {
                throw new IllegalArgumentException(
                        "gradientBindings[" + index
                                + "] duplicates an earlier target in this derivative order");
            }
        }

        List<ValueId> expectedBoundary =
                new ArrayList<>(forwardBindings.size() + gradientBindings.size());
        expectedBoundary.addAll(
                forwardBindings.stream().map(ForwardPublicationBinding::valueId).toList());
        Set<ValueId> seenValues = new HashSet<>(expectedBoundary);
        for (GradientPublicationBinding gradientBinding : gradientBindings) {
            if (seenValues.add(gradientBinding.valueId())) {
                expectedBoundary.add(gradientBinding.valueId());
            }
        }
        if (!graph.outputs().equals(expectedBoundary)) {
            throw new IllegalArgumentException(
                    "graph output boundary does not match publication bindings");
        }

        this.forwardBindings = List.copyOf(forwardBindings);
        this.gradientBindings = List.copyOf(gradientBindings);
    }

    /**
     * Returns the owning final graph.
     *
     * @return the exact non-null graph reference supplied at construction
     */
    public CompiledGraphModel graph() {
        return graph;
    }

    /**
     * Returns ordered requested-forward publication bindings.
     *
     * @return immutable non-null membership snapshot retaining exact binding references
     */
    public List<ForwardPublicationBinding> forwardBindings() {
        return forwardBindings;
    }

    /**
     * Returns ordered differentiation-target-to-gradient publication bindings.
     *
     * @return immutable non-null membership snapshot retaining exact binding references
     */
    public List<GradientPublicationBinding> gradientBindings() {
        return gradientBindings;
    }
}
