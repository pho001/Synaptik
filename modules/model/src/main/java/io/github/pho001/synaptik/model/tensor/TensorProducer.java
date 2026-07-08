package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.operation.Operation;
import java.util.List;
import java.util.Objects;

/**
 * Immutable identity of one public tensor-expression occurrence before compiler capture.
 *
 * <p>A producer retains the exact backend-independent operation reference, an immutable ordered
 * snapshot of exact input tensor references, and an immutable ordered snapshot of exact output
 * descriptor references. Repeated references and encounter order remain significant. The
 * operation's selected signature validates the final input and output counts. Ordinary object
 * identity distinguishes separately invoked occurrences even when all retained values are
 * structurally equal; identity hash codes are not stable producer identifiers.</p>
 *
 * <p>Output descriptors describe result positions without retaining result tensors. The reference
 * direction is therefore result tensor to provenance to producer, with no producer/result cycle.
 * This pre-capture identity is neither a tensor nor compiled graph state and owns no graph-local
 * node or value identity, storage, compiler state, backend metadata, or runtime behavior.</p>
 */
public final class TensorProducer {
    private final Operation operation;
    private final List<Tensor> inputs;
    private final List<TensorDescriptor> outputDescriptors;

    /**
     * Creates one validated expression producer.
     *
     * @param operation exact non-null immutable operation reference to retain
     * @param inputs non-null ordered input tensor references to snapshot; empty and repeated
     *     positions are retained when accepted by the operation signature
     * @param outputDescriptors non-null, non-empty ordered output descriptor references to
     *     snapshot; repeated exact references are permitted
     * @throws NullPointerException if {@code operation}, {@code inputs}, or
     *     {@code outputDescriptors} is null, checked in that order; or if an input or output
     *     descriptor element is null, reported by its zero-based indexed position
     * @throws IllegalArgumentException if {@code outputDescriptors} is empty, or if the final
     *     input or output count is outside the inclusive range accepted by the operation's
     *     selected signature
     */
    TensorProducer(
            Operation operation,
            List<Tensor> inputs,
            List<TensorDescriptor> outputDescriptors) {
        this.operation = Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(outputDescriptors, "outputDescriptors");

        for (int index = 0; index < inputs.size(); index++) {
            Objects.requireNonNull(inputs.get(index), "inputs[" + index + "]");
        }
        List<Tensor> inputSnapshot = List.copyOf(inputs);

        if (outputDescriptors.isEmpty()) {
            throw new IllegalArgumentException("outputDescriptors must not be empty");
        }
        for (int index = 0; index < outputDescriptors.size(); index++) {
            Objects.requireNonNull(
                    outputDescriptors.get(index), "outputDescriptors[" + index + "]");
        }
        List<TensorDescriptor> outputDescriptorSnapshot = List.copyOf(outputDescriptors);

        operation.signature().validateOccurrence(
                inputSnapshot.size(), outputDescriptorSnapshot.size());
        this.inputs = inputSnapshot;
        this.outputDescriptors = outputDescriptorSnapshot;
    }

    /**
     * Returns this occurrence's semantic operation.
     *
     * @return the exact non-null operation reference supplied at construction
     */
    public Operation operation() {
        return operation;
    }

    /**
     * Returns this occurrence's ordered input positions.
     *
     * @return the immutable ordered snapshot containing the exact input tensor references
     */
    public List<Tensor> inputs() {
        return inputs;
    }

    /**
     * Returns this occurrence's ordered output descriptors.
     *
     * @return the immutable non-empty ordered snapshot containing the exact descriptor references
     */
    public List<TensorDescriptor> outputDescriptors() {
        return outputDescriptors;
    }

    /**
     * Returns the number of output positions described by this producer.
     *
     * @return the positive size derived from {@link #outputDescriptors()}
     */
    public int outputCount() {
        return outputDescriptors.size();
    }
}
