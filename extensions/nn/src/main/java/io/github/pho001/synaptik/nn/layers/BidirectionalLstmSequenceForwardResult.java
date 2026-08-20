package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.List;
import java.util.Objects;

/**
 * Carries compact merged hidden outputs and four separate final states from one statically packed
 * bidirectional LSTM composition.
 *
 * <p>Output element {@code t} contains only original rows whose construction-time length exceeds
 * {@code t}, in ascending original batch order. Its final axis contains forward hidden features
 * followed by backward hidden features realigned to the same original time. Each final-state
 * Tensor contains one row per original batch entry. Intermediate cell states are not a public
 * output sequence. This value snapshots only the list structure: it retains exact Tensor
 * references, copies or evaluates no Tensor, and carries no lengths, mask, module state, or
 * runtime execution state.</p>
 *
 * @param packedOutputs non-null ordered compact hidden-output list with no null element;
 *     structurally snapshotted while retaining every exact Tensor reference
 * @param forwardFinalHidden non-null exact forward final hidden state in original batch order, or
 *     the exact forward initial hidden state when no step is active
 * @param forwardFinalCell non-null exact forward final cell state in original batch order, or the
 *     exact forward initial cell state when no step is active
 * @param backwardFinalHidden non-null exact backward final hidden state in original batch order,
 *     or the exact backward initial hidden state when no step is active
 * @param backwardFinalCell non-null exact backward final cell state in original batch order, or
 *     the exact backward initial cell state when no step is active
 */
public record BidirectionalLstmSequenceForwardResult(
        List<Tensor> packedOutputs,
        Tensor forwardFinalHidden,
        Tensor forwardFinalCell,
        Tensor backwardFinalHidden,
        Tensor backwardFinalCell) {
    /**
     * Creates an immutable structural snapshot from exact Tensor references.
     *
     * @param packedOutputs non-null ordered list with no null element; its structure is copied
     * @param forwardFinalHidden non-null exact forward final hidden reference to retain
     * @param forwardFinalCell non-null exact forward final cell reference to retain
     * @param backwardFinalHidden non-null exact backward final hidden reference to retain
     * @param backwardFinalCell non-null exact backward final cell reference to retain
     * @throws NullPointerException if {@code packedOutputs}, its first null element, or a state
     *     component is null, checked in record declaration order
     */
    public BidirectionalLstmSequenceForwardResult {
        Objects.requireNonNull(packedOutputs, "packedOutputs");
        packedOutputs = List.copyOf(packedOutputs);
        Objects.requireNonNull(forwardFinalHidden, "forwardFinalHidden");
        Objects.requireNonNull(forwardFinalCell, "forwardFinalCell");
        Objects.requireNonNull(backwardFinalHidden, "backwardFinalHidden");
        Objects.requireNonNull(backwardFinalCell, "backwardFinalCell");
    }
}
