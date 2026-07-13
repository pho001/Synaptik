package io.github.pho001.synaptik.model.operation.loss;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;
import java.util.Optional;

/**
 * Carries the normalized class axis, explicit reduction, and optional exact integral ignore index
 * for index-target categorical cross-entropy directly from logits.
 *
 * <p>The optional ignore value is intrinsic metadata rather than a Tensor input. A matching target
 * is ignored before bounds checking or logits evaluation, and mean reduction divides by the
 * number of non-ignored targets. An empty or all-ignored domain therefore has positive-zero sum
 * and NaN mean. This record retains no Shape, count, denominator, target value, executable
 * algorithm, gradient, graph, compiler, backend, runtime, or training state.</p>
 *
 * @param axis normalized non-negative logits class axis
 * @param reduction non-null explicit reduction over the non-class target domain
 * @param ignoreIndex non-null optional containing an exact INT32 or INT64 value when present
 */
public record IndexCategoricalCrossEntropyWithLogitsAttrs(
        int axis, LossReduction reduction, Optional<ScalarValue> ignoreIndex)
        implements OperationAttrs {
    /**
     * Creates immutable index-target categorical-cross-entropy attributes.
     *
     * @param axis normalized non-negative logits class axis to retain
     * @param reduction non-null exact reduction value to retain
     * @param ignoreIndex non-null optional exact integral ignore value to retain
     * @throws IllegalArgumentException if {@code axis} is negative or a present ignore value is
     *     not INT32 or INT64
     * @throws NullPointerException if {@code reduction} or {@code ignoreIndex} is null, checked in
     *     component order
     */
    public IndexCategoricalCrossEntropyWithLogitsAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
        Objects.requireNonNull(reduction, "reduction");
        Objects.requireNonNull(ignoreIndex, "ignoreIndex");
        if (ignoreIndex.isPresent()) {
            DataType dataType = ignoreIndex.orElseThrow().dataType();
            if (dataType != DataType.INT32 && dataType != DataType.INT64) {
                throw new IllegalArgumentException(
                        "ignoreIndex must have data type INT32 or INT64, but was " + dataType);
            }
        }
    }

}
