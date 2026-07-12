package io.github.pho001.synaptik.model.operation.loss;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;

/**
 * Carries the explicit reduction for one mean-squared-error loss occurrence.
 *
 * <p>The record stores only the exact non-null reduction value. It contains no input, output
 * shape, denominator, data type, numerical algorithm, gradient, graph, backend, or runtime state.</p>
 *
 * @param reduction non-null explicit reduction applied to the complete squared-error domain
 */
public record MeanSquaredErrorAttrs(LossReduction reduction) implements OperationAttrs {
    /**
     * Creates immutable mean-squared-error attributes.
     *
     * @param reduction non-null exact reduction value to retain
     * @throws NullPointerException if {@code reduction} is null, with message {@code reduction}
     */
    public MeanSquaredErrorAttrs {
        Objects.requireNonNull(reduction, "reduction");
    }

    /**
     * Returns the requested complete-domain loss reduction.
     *
     * @return the exact non-null reduction supplied at construction
     */
    @Override
    public LossReduction reduction() {
        return reduction;
    }
}
