package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Classifies compile graph inputs as caller-bindable inputs or exact logical constants.
 *
 * <p>The typed bindable list, its compatibility value-ID projection, and the constant list are
 * immutable membership snapshots retaining exact identity and scalar references. This
 * output-only plan describes logical source roles only; it contains no Tensor, descriptor, dense
 * payload, storage, backend value, materialization instruction, or physical allocation.</p>
 */
public final class CompileConstantPlan {
    private final List<BindableInput> bindableInputBindings;
    private final List<ValueId> bindableInputs;
    private final List<ConstantSource> constantSources;

    /**
     * Creates an immutable disjoint source-role classification.
     *
     * @param bindableInputBindings non-null ordered caller Tensor-to-graph-input bindings to
     *     snapshot; elements, Tensor IDs, and value IDs must be non-null and unique
     * @param constantSources non-null ordered logical constant sources to snapshot; elements and
     *     their value IDs must be unique and disjoint from bindable inputs
     * @throws NullPointerException if a top-level argument or list element is {@code null}
     * @throws IllegalArgumentException if a Tensor ID or value ID repeats within the bindable
     *     list, or a value ID repeats within or across the two source-role lists
     */
    CompileConstantPlan(
            List<BindableInput> bindableInputBindings,
            List<ConstantSource> constantSources) {
        Objects.requireNonNull(bindableInputBindings, "bindableInputBindings");
        Objects.requireNonNull(constantSources, "constantSources");

        Set<ValueId> sourceIds = new HashSet<>();
        Set<TensorId> tensorIds = new HashSet<>();
        for (int index = 0; index < bindableInputBindings.size(); index++) {
            BindableInput input = Objects.requireNonNull(
                    bindableInputBindings.get(index),
                    "bindableInputBindings[" + index + "]");
            if (!tensorIds.add(input.tensorId())) {
                throw new IllegalArgumentException(
                        "bindableInputBindings[" + index + "] repeats " + input.tensorId());
            }
            if (!sourceIds.add(input.valueId())) {
                throw new IllegalArgumentException(
                        "bindableInputBindings[" + index + "] repeats " + input.valueId());
            }
        }
        for (int index = 0; index < constantSources.size(); index++) {
            ConstantSource source = Objects.requireNonNull(
                    constantSources.get(index), "constantSources[" + index + "]");
            if (!sourceIds.add(source.valueId())) {
                throw new IllegalArgumentException(
                        "constantSources[" + index + "] overlaps or duplicates "
                                + source.valueId());
            }
        }

        this.bindableInputBindings = List.copyOf(bindableInputBindings);
        this.bindableInputs = this.bindableInputBindings.stream()
                .map(BindableInput::valueId)
                .toList();
        this.constantSources = List.copyOf(constantSources);
    }

    /**
     * Returns the exact logical caller Tensor identity for every caller-bindable graph input.
     *
     * @return immutable non-null ordered membership snapshot in final graph-input order,
     *     retaining exact Tensor-ID and value-ID references
     */
    public List<BindableInput> bindableInputBindings() {
        return bindableInputBindings;
    }

    /**
     * Returns graph inputs whose values remain caller-bindable.
     *
     * @return immutable non-null ordered projection of {@link #bindableInputBindings()}, retaining
     *     exact value-ID references
     */
    public List<ValueId> bindableInputs() {
        return bindableInputs;
    }

    /**
     * Returns graph inputs fixed to exact logical scalar splats.
     *
     * @return immutable non-null ordered membership snapshot retaining exact source references
     */
    public List<ConstantSource> constantSources() {
        return constantSources;
    }

    /**
     * Associates one logical caller Tensor with its exact final caller-bindable graph input.
     *
     * <p>The standalone value validates identity components but does not by itself prove graph
     * membership. The owning validated compile artifacts establish membership and ordering.</p>
     *
     * @param tensorId non-null immutable logical caller Tensor identity
     * @param valueId non-null exact final graph-input identity
     */
    public record BindableInput(TensorId tensorId, ValueId valueId) {
        /**
         * Validates one caller-bindable identity association.
         *
         * @param tensorId non-null immutable logical caller Tensor identity
         * @param valueId non-null exact final graph-input identity
         * @throws NullPointerException if either component is {@code null}, checked in component
         *     order
         */
        public BindableInput {
            Objects.requireNonNull(tensorId, "tensorId");
            Objects.requireNonNull(valueId, "valueId");
        }
    }

    /**
     * Associates one graph input with the exact scalar repeated at every logical coordinate.
     *
     * @param valueId non-null exact graph-input identity
     * @param value non-null exact immutable typed scalar reference
     */
    public record ConstantSource(ValueId valueId, ScalarValue value) {
        /**
         * Validates one logical constant source.
         *
         * @param valueId non-null exact graph-input identity
         * @param value non-null exact immutable typed scalar
         * @throws NullPointerException if either component is {@code null}
         */
        public ConstantSource {
            Objects.requireNonNull(valueId, "valueId");
            Objects.requireNonNull(value, "value");
        }
    }
}
