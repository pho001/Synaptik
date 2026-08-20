package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.List;
import java.util.Objects;

/**
 * Carries compact merged outputs and separate final hidden states from one statically packed
 * bidirectional vanilla-RNN composition.
 *
 * <p>Output element {@code t} contains only original rows whose construction-time length exceeds
 * {@code t}, in ascending original batch order. Its final axis contains forward features followed
 * by backward features realigned to the same original time. Both final-state Tensors contain one
 * row per original batch entry. This value snapshots only the list structure: it retains exact
 * Tensor references, copies or evaluates no Tensor, and carries no lengths, mask, module state,
 * or runtime execution state.</p>
 *
 * @param packedOutputs non-null ordered compact-output list with no null element; structurally
 *     snapshotted while retaining every exact Tensor reference
 * @param forwardFinalHidden non-null exact forward final hidden state in original batch order,
 *     or the exact forward initial state when no step is active
 * @param backwardFinalHidden non-null exact backward final hidden state in original batch order,
 *     or the exact backward initial state when no step is active
 */
public record BidirectionalRnnSequenceForwardResult(
        List<Tensor> packedOutputs,
        Tensor forwardFinalHidden,
        Tensor backwardFinalHidden) {
    /**
     * Creates an immutable structural snapshot from exact Tensor references.
     *
     * @param packedOutputs non-null ordered list with no null element; its structure is copied
     * @param forwardFinalHidden non-null exact forward final hidden reference to retain
     * @param backwardFinalHidden non-null exact backward final hidden reference to retain
     * @throws NullPointerException if {@code packedOutputs}, its first null element,
     *     {@code forwardFinalHidden}, or {@code backwardFinalHidden} is null, checked in that
     *     order
     */
    public BidirectionalRnnSequenceForwardResult {
        Objects.requireNonNull(packedOutputs, "packedOutputs");
        packedOutputs = List.copyOf(packedOutputs);
        Objects.requireNonNull(forwardFinalHidden, "forwardFinalHidden");
        Objects.requireNonNull(backwardFinalHidden, "backwardFinalHidden");
    }
}
