package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.PublicationBinding;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Associates ordered forward and gradient publication roles with one exact compiled graph.
 *
 * <p>The two role lists have distinct meanings. A forward binding identifies the requested
 * forward Tensor and its final graph value. A gradient binding identifies a differentiation
 * target and that target's final gradient value; it does not add gradient state to Tensor.
 * Membership is snapshotted while exact immutable binding references are retained.</p>
 *
 * <p>This output-only plan proves graph membership and boundary order. It contains no publication
 * policy, alias or copy decision, storage, delivery behavior, prepared state, or runtime state.</p>
 */
public final class PublicationPlan {
    private final CompiledGraphModel graph;
    private final List<PublicationBinding> forwardOutputs;
    private final List<PublicationBinding> gradientResults;

    /**
     * Creates and validates publication roles for one exact final graph.
     *
     * @param graph non-null final graph retained by exact reference
     * @param forwardOutputs non-null ordered forward bindings to snapshot; Tensor and value IDs
     *     must each be unique and the values must equal the graph-output prefix
     * @param gradientResults non-null ordered target-to-gradient bindings to snapshot; target IDs
     *     must be unique, while gradient values may repeat or equal forward values
     * @throws NullPointerException if a top-level argument or list element is {@code null}
     * @throws IllegalArgumentException if a binding violates graph membership, uniqueness, prefix,
     *     or complete boundary rules
     */
    PublicationPlan(
            CompiledGraphModel graph,
            List<PublicationBinding> forwardOutputs,
            List<PublicationBinding> gradientResults) {
        this.graph = Objects.requireNonNull(graph, "graph");
        Objects.requireNonNull(forwardOutputs, "forwardOutputs");
        Objects.requireNonNull(gradientResults, "gradientResults");

        Set<ValueId> graphOutputIds = new HashSet<>(graph.outputs());
        Set<TensorId> forwardTensorIds = new HashSet<>();
        Set<ValueId> forwardValueIds = new HashSet<>();
        for (int index = 0; index < forwardOutputs.size(); index++) {
            PublicationBinding binding = Objects.requireNonNull(
                    forwardOutputs.get(index), "forwardOutputs[" + index + "]");
            if (!graphOutputIds.contains(binding.valueId())) {
                throw new IllegalArgumentException(
                        "forwardOutputs[" + index + "] is not a graph output");
            }
            if (!forwardTensorIds.add(binding.tensorId())) {
                throw new IllegalArgumentException(
                        "forwardOutputs[" + index + "] duplicates an earlier TensorId");
            }
            if (!forwardValueIds.add(binding.valueId())) {
                throw new IllegalArgumentException(
                        "forwardOutputs[" + index + "] duplicates an earlier forward value");
            }
            if (index >= graph.outputs().size()
                    || !binding.valueId().equals(graph.outputs().get(index))) {
                throw new IllegalArgumentException(
                        "forwardOutputs[" + index + "] does not match graph output prefix");
            }
        }

        Set<TensorId> gradientTensorIds = new HashSet<>();
        for (int index = 0; index < gradientResults.size(); index++) {
            PublicationBinding binding = Objects.requireNonNull(
                    gradientResults.get(index), "gradientResults[" + index + "]");
            if (!graphOutputIds.contains(binding.valueId())) {
                throw new IllegalArgumentException(
                        "gradientResults[" + index + "] is not a graph output");
            }
            if (!gradientTensorIds.add(binding.tensorId())) {
                throw new IllegalArgumentException(
                        "gradientResults[" + index + "] duplicates an earlier TensorId");
            }
        }

        List<ValueId> expectedBoundary =
                new ArrayList<>(forwardOutputs.size() + gradientResults.size());
        expectedBoundary.addAll(
                forwardOutputs.stream().map(PublicationBinding::valueId).toList());
        Set<ValueId> seenValues = new HashSet<>(expectedBoundary);
        for (PublicationBinding gradientResult : gradientResults) {
            if (seenValues.add(gradientResult.valueId())) {
                expectedBoundary.add(gradientResult.valueId());
            }
        }
        if (!graph.outputs().equals(expectedBoundary)) {
            throw new IllegalArgumentException(
                    "graph output boundary does not match publication roles");
        }

        this.forwardOutputs = List.copyOf(forwardOutputs);
        this.gradientResults = List.copyOf(gradientResults);
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
    public List<PublicationBinding> forwardOutputs() {
        return forwardOutputs;
    }

    /**
     * Returns ordered differentiation-target-to-gradient publication bindings.
     *
     * @return immutable non-null membership snapshot retaining exact binding references
     */
    public List<PublicationBinding> gradientResults() {
        return gradientResults;
    }
}
