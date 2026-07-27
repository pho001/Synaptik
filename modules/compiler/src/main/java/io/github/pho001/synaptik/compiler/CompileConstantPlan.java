package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Classifies compile graph inputs as caller-bindable inputs or exact logical constants.
 *
 * <p>Both lists are immutable membership snapshots retaining exact identifier and scalar
 * references. This output-only plan describes logical source roles only; it contains no Tensor,
 * dense payload, storage, backend value, materialization instruction, or physical allocation.</p>
 */
public final class CompileConstantPlan {
    private final List<ValueId> bindableInputs;
    private final List<ConstantSource> constantSources;

    /**
     * Creates an immutable disjoint source-role classification.
     *
     * @param bindableInputs non-null ordered bindable input IDs to snapshot; elements must be
     *     non-null and unique
     * @param constantSources non-null ordered logical constant sources to snapshot; elements and
     *     their value IDs must be unique and disjoint from bindable inputs
     * @throws NullPointerException if a top-level argument or list element is {@code null}
     * @throws IllegalArgumentException if an ID repeats within or across the two role lists
     */
    CompileConstantPlan(
            List<ValueId> bindableInputs,
            List<ConstantSource> constantSources) {
        Objects.requireNonNull(bindableInputs, "bindableInputs");
        Objects.requireNonNull(constantSources, "constantSources");

        Set<ValueId> sourceIds = new HashSet<>();
        for (int index = 0; index < bindableInputs.size(); index++) {
            ValueId input = Objects.requireNonNull(
                    bindableInputs.get(index), "bindableInputs[" + index + "]");
            if (!sourceIds.add(input)) {
                throw new IllegalArgumentException(
                        "bindableInputs[" + index + "] duplicates " + input);
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

        this.bindableInputs = List.copyOf(bindableInputs);
        this.constantSources = List.copyOf(constantSources);
    }

    /**
     * Returns graph inputs whose values remain caller-bindable.
     *
     * @return immutable non-null ordered membership snapshot retaining exact value-ID references
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
