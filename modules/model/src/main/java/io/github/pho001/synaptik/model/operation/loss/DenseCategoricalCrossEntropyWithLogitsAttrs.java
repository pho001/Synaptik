package io.github.pho001.synaptik.model.operation.loss;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;

/**
 * Carries the normalized class axis and explicit reduction for dense-target categorical
 * cross-entropy directly from logits.
 *
 * <p>The axis is already normalized against the logits rank and is therefore non-negative. The
 * reduction applies to losses after the class axis is removed: {@link LossReduction#NONE} retains
 * one loss per non-class coordinate, {@link LossReduction#SUM} sums those losses, and
 * {@link LossReduction#MEAN} divides that sum by the non-class sample count. The denominator is
 * not the class count, logits element count, positive-target count, or target-weight sum.</p>
 *
 * <p>The record stores only intrinsic loss meaning. It retains no Tensor, Shape, class extent,
 * sample count, target value, denominator, numerical algorithm, gradient, graph, compiler,
 * backend, runtime, or training state. In particular, it does not prove the later dynamic
 * obligation that the sample domain is empty or the class extent is positive.</p>
 *
 * @param axis normalized non-negative class axis of the logits and exact-shape dense target
 * @param reduction non-null explicit reduction over the non-class sample domain
 */
public record DenseCategoricalCrossEntropyWithLogitsAttrs(
        int axis, LossReduction reduction) implements OperationAttrs {
    /**
     * Creates immutable dense categorical-cross-entropy attributes.
     *
     * @param axis normalized non-negative class axis to retain
     * @param reduction non-null exact reduction value to retain
     * @throws IllegalArgumentException if {@code axis} is negative
     * @throws NullPointerException if {@code reduction} is null, with message {@code reduction}
     */
    public DenseCategoricalCrossEntropyWithLogitsAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
        Objects.requireNonNull(reduction, "reduction");
    }

    /**
     * Returns the normalized class axis.
     *
     * @return stored non-negative logits axis
     */
    @Override
    public int axis() {
        return axis;
    }

    /**
     * Returns the explicit sample-domain reduction.
     *
     * @return exact non-null reduction supplied at construction
     */
    @Override
    public LossReduction reduction() {
        return reduction;
    }
}
