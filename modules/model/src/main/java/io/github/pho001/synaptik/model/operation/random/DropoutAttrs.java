package io.github.pho001.synaptik.model.operation.random;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Immutable semantic attributes for one explicit-state inverted-dropout occurrence.
 *
 * <p>The probability is the fraction of logical input elements to drop. It must be finite and
 * numerically in {@code [0.0, 1.0)}. Both signed zero values are valid, mean zero dropout, and are
 * retained without normalization. This value contains no Tensor or state and selects no random
 * algorithm, execution route, sampling behavior, or gradient rule.</p>
 *
 * @param probability finite drop probability in {@code [0.0, 1.0)}
 */
public record DropoutAttrs(double probability) implements OperationAttrs {
    /**
     * Creates validated dropout attributes.
     *
     * @param probability finite drop probability in {@code [0.0, 1.0)}; signed zero is retained
     * @throws IllegalArgumentException if {@code probability} is non-finite, negative, or at least
     *     one
     */
    public DropoutAttrs {
        if (!Double.isFinite(probability) || probability < 0.0d || probability >= 1.0d) {
            throw new IllegalArgumentException(
                    "probability must be finite and in [0.0, 1.0): " + probability);
        }
    }
}
