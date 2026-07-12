package io.github.pho001.synaptik.model.operation.loss;

/**
 * Selects how a loss operation reduces its complete logical element domain.
 *
 * <p>This immutable vocabulary is configuration used by family-specific attributes. It stores no
 * axis, denominator, mask, default, parser, tensor, or executable behavior.</p>
 */
public enum LossReduction {
    /** Retains one loss value at every logical prediction coordinate. */
    NONE,

    /** Sums all logical loss values into one scalar; an empty domain produces positive zero. */
    SUM,

    /**
     * Divides the complete logical loss sum by the full logical element count; a scalar has count
     * one and an empty domain produces NaN.
     */
    MEAN
}
