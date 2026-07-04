package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.operation.Operation;
import java.util.List;
import java.util.Objects;

/**
 * Immutable expression-origin metadata for one public tensor.
 *
 * <p>Provenance retains the exact backend-independent operation reference and an immutable,
 * ordered snapshot of the exact input-tensor references. Empty inputs are valid for zero-input
 * semantic origins, repeated tensor references preserve distinct ordered roles, and later
 * mutation of the caller's list cannot change this value.</p>
 *
 * <p>This record is not graph membership, producer-occurrence identity, an intermediate-
 * representation node, or executable behavior. It does not validate operation arity, descriptor
 * compatibility, cycles, or graph-wide structure. Record equality and hashing compare the
 * operation value and ordered tensor references using their ordinary equality; because
 * {@link Tensor} uses object identity, equal tensor identifiers do not make different input
 * objects equal. Generated text is diagnostic only and is not serialization or graph identity.</p>
 *
 * @param operation the exact non-null immutable semantic operation reference to retain
 * @param inputs the non-null ordered input list to snapshot; elements must be non-null, while an
 *     empty list and repeated tensor references are permitted
 */
public record TensorProvenance(Operation operation, List<Tensor> inputs) {
    /**
     * Creates immutable expression-origin metadata.
     *
     * <p>Validation checks {@code operation}, then {@code inputs}, then input elements in ascending
     * index order. Only after every element is validated is the list snapshotted with
     * {@link List#copyOf(java.util.Collection)}. The exact operation and tensor-element references
     * are retained without traversal, inference, semantic validation, or graph capture.</p>
     *
     * @param operation the exact non-null immutable semantic operation reference to retain
     * @param inputs the non-null ordered input list to snapshot; empty and repeated inputs are
     *     valid, and no list-container identity is retained
     * @throws NullPointerException if {@code operation} is null, with message {@code operation};
     *     if {@code inputs} is null, with message {@code inputs}; or if an element is null, with
     *     its zero-based indexed message such as {@code inputs[2]}
     */
    public TensorProvenance {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        for (int index = 0; index < inputs.size(); index++) {
            Objects.requireNonNull(inputs.get(index), "inputs[" + index + "]");
        }
        inputs = List.copyOf(inputs);
    }

    /**
     * Returns the semantic operation that produced the tensor carrying this provenance.
     *
     * @return the exact non-null immutable operation reference supplied at construction
     */
    @Override
    public Operation operation() {
        return operation;
    }

    /**
     * Returns the ordered immutable input-tensor snapshot.
     *
     * <p>The list is non-null and unmodifiable, preserves empty, repeated, and ordered positions,
     * and contains the exact identity-bearing tensor references supplied at construction. Its
     * container identity is not part of the contract.</p>
     *
     * @return the non-null immutable ordered snapshot of exact input-tensor references
     */
    @Override
    public List<Tensor> inputs() {
        return inputs;
    }
}
