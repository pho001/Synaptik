package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Carries the compact outputs of one statically packed vanilla-RNN sequence and its restored
 * final hidden state.
 *
 * <p>The output list has one exact {@link RnnCell} result per represented time step. Its element
 * at time {@code t} contains only rows whose construction-time length exceeds {@code t}, in their
 * original relative batch order. {@code finalHidden} contains one row per original batch entry,
 * restored to original batch order. This value snapshots only the list structure; it neither
 * copies nor evaluates any Tensor, and it carries no lengths, mask, or runtime execution state.</p>
 *
 * @param packedOutputs non-null ordered list with no null element; structurally snapshotted while
 *     retaining every exact Tensor reference
 * @param finalHidden non-null exact final-hidden Tensor produced by sequence construction, or the
 *     exact initial-hidden Tensor when no time step is active
 */
public record RnnSequenceForwardResult(List<Tensor> packedOutputs, Tensor finalHidden) {
    /**
     * Creates an immutable result snapshot.
     *
     * @param packedOutputs non-null ordered list with no null element; its structure is copied and
     *     its exact Tensor elements are retained
     * @param finalHidden non-null exact restored final-hidden Tensor reference to retain
     * @throws NullPointerException if {@code packedOutputs}, its first null element, or
     *     {@code finalHidden} is null, checked in that order
     */
    public RnnSequenceForwardResult {
        Objects.requireNonNull(packedOutputs, "packedOutputs");
        List<Tensor> snapshot = new ArrayList<>(packedOutputs.size());
        int index = 0;
        for (Tensor output : packedOutputs) {
            snapshot.add(Objects.requireNonNull(output, "packedOutputs[" + index + "]"));
            index++;
        }
        packedOutputs = List.copyOf(snapshot);
        finalHidden = Objects.requireNonNull(finalHidden, "finalHidden");
    }
}
