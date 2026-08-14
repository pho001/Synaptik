package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.Objects;

/**
 * Carries both explicit recurrent states produced by one
 * {@link LstmCell#forward(Tensor, Tensor, Tensor)} call.
 *
 * <p>The components retain the exact next-hidden and next-cell Tensor references constructed by
 * the cell. Creating this value performs no descriptor validation, Tensor expression creation,
 * identifier allocation, mutation, or execution.</p>
 *
 * @param nextHidden the non-null exact next-hidden Tensor, retained without copying
 * @param nextCell the non-null exact next-cell Tensor, retained without copying
 */
public record LstmCellForwardResult(Tensor nextHidden, Tensor nextCell) {
    /**
     * Creates a result from the exact two caller-supplied state references.
     *
     * @param nextHidden the non-null exact next-hidden Tensor, retained without copying
     * @param nextCell the non-null exact next-cell Tensor, retained without copying
     * @throws NullPointerException if {@code nextHidden} or {@code nextCell} is null, checked in
     *     that order with the component name as the message
     */
    public LstmCellForwardResult {
        Objects.requireNonNull(nextHidden, "nextHidden");
        Objects.requireNonNull(nextCell, "nextCell");
    }
}
