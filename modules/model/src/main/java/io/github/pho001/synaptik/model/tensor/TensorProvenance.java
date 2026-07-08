package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.operation.Operation;
import java.util.List;
import java.util.Objects;

/**
 * Immutable association between one public tensor and one indexed producer output.
 *
 * <p>Provenance retains one exact {@link TensorProducer} reference and a zero-based output index.
 * The operation, ordered immutable inputs, and exact selected output descriptor are derived from
 * that producer. Single-output expressions use output index zero.</p>
 *
 * <p>This record is not graph membership, an intermediate-representation node, or executable
 * behavior. Record equality and hashing compare the producer through its ordinary object identity
 * and the output index. Thus two positions of one producer differ, and structurally equal but
 * separately invoked producers remain distinct.</p>
 *
 * @param producer the exact non-null immutable expression producer reference to retain
 * @param outputIndex the zero-based producer output position
 */
public record TensorProvenance(TensorProducer producer, int outputIndex) {
    /**
     * Creates immutable expression-origin metadata.
     *
     * <p>Validation checks the producer reference, then the lower and upper output-index bounds.
     * The exact producer reference is retained without traversal, inference, or graph capture.</p>
     *
     * @param producer the exact non-null immutable expression producer reference to retain
     * @param outputIndex the zero-based producer output position
     * @throws NullPointerException if {@code producer} is null, with message {@code producer}
     * @throws IllegalArgumentException if {@code outputIndex} is negative or is not less than the
     *     producer's output count
     */
    public TensorProvenance {
        Objects.requireNonNull(producer, "producer");
        if (outputIndex < 0) {
            throw new IllegalArgumentException(
                    "outputIndex must be non-negative: " + outputIndex);
        }
        if (outputIndex >= producer.outputCount()) {
            throw new IllegalArgumentException(
                    "outputIndex " + outputIndex
                            + " is outside available output count " + producer.outputCount());
        }
    }

    /**
     * Returns the semantic operation that produced the tensor carrying this provenance.
     *
     * @return the exact non-null immutable operation reference retained by {@link #producer()}
     */
    public Operation operation() {
        return producer.operation();
    }

    /**
     * Returns the ordered immutable input-tensor snapshot.
     *
     * <p>The list is non-null and unmodifiable, preserves empty, repeated, and ordered positions,
     * and contains the exact identity-bearing tensor references supplied at construction. Its
     * container identity is not part of the contract.</p>
     *
     * @return the non-null immutable ordered snapshot of exact input-tensor references retained by
     *     {@link #producer()}
     */
    public List<Tensor> inputs() {
        return producer.inputs();
    }

    /**
     * Returns the descriptor selected by this provenance position.
     *
     * @return the exact non-null descriptor reference at {@link #outputIndex()} in the producer's
     *     immutable ordered descriptor snapshot
     */
    public TensorDescriptor outputDescriptor() {
        return producer.outputDescriptors().get(outputIndex);
    }
}
