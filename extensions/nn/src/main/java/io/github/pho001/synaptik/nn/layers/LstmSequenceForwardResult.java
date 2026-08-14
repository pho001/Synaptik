package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Carries the compact hidden outputs of one statically packed LSTM sequence and both restored
 * recurrent states.
 *
 * <p>The output at time {@code t} contains only rows whose construction-time length exceeds
 * {@code t}, in stable original relative order. {@code finalHidden} and {@code finalCell} each
 * contain one row per original batch entry in original order. Intermediate cell states are not a
 * public output sequence. This value snapshots only the output-list structure; it neither copies
 * nor evaluates a Tensor and carries no lengths, mask, or runtime execution state.</p>
 *
 * @param packedOutputs non-null ordered hidden-output list with no null element; structurally
 *     snapshotted while retaining every exact Tensor reference
 * @param finalHidden non-null exact restored final-hidden Tensor, or the exact initial-hidden
 *     Tensor when no time step is active
 * @param finalCell non-null exact restored final-cell Tensor, or the exact initial-cell Tensor
 *     when no time step is active
 */
public record LstmSequenceForwardResult(
        List<Tensor> packedOutputs, Tensor finalHidden, Tensor finalCell) {
    /**
     * Creates an immutable result snapshot from exact Tensor references.
     *
     * @param packedOutputs non-null ordered hidden-output list with no null element; its structure
     *     is copied and its exact Tensor elements are retained
     * @param finalHidden non-null exact restored final-hidden Tensor reference to retain
     * @param finalCell non-null exact restored final-cell Tensor reference to retain
     * @throws NullPointerException if {@code packedOutputs}, its first null element,
     *     {@code finalHidden}, or {@code finalCell} is null, checked in that order
     */
    public LstmSequenceForwardResult {
        Objects.requireNonNull(packedOutputs, "packedOutputs");
        List<Tensor> snapshot = new ArrayList<>(packedOutputs.size());
        int index = 0;
        for (Tensor output : packedOutputs) {
            snapshot.add(Objects.requireNonNull(output, "packedOutputs[" + index + "]"));
            index++;
        }
        packedOutputs = List.copyOf(snapshot);
        finalHidden = Objects.requireNonNull(finalHidden, "finalHidden");
        finalCell = Objects.requireNonNull(finalCell, "finalCell");
    }
}
