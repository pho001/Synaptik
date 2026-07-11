package io.github.pho001.synaptik.model.tensor;

import java.util.Objects;

/**
 * Values and logical input indices produced by one top-K occurrence.
 *
 * <p>This shallowly immutable carrier retains both exact Tensor wrappers and uses ordinary record
 * value semantics. The carrier itself does not discover siblings, reconstruct outputs, manage
 * aliases, or validate descriptor, producer, provenance, Shape, identity, or storage agreement.
 * Tensor construction supplies values at producer slot zero and INT64 indices at slot one.</p>
 *
 * @param values non-null selected-values Tensor retained by exact reference
 * @param indices non-null INT64 logical-indices Tensor retained by exact reference
 */
public record TopKResult(Tensor values, Tensor indices) {
    /**
     * Creates a result retaining both exact wrappers.
     *
     * @param values non-null values wrapper
     * @param indices non-null indices wrapper
     * @throws NullPointerException if either component is null, checked in component order
     */
    public TopKResult {
        values = Objects.requireNonNull(values, "values");
        indices = Objects.requireNonNull(indices, "indices");
    }
}
