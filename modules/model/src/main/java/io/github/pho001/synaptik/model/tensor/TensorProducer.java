package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.operation.Operation;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Supplier;

/**
 * Immutable identity of one public tensor-expression occurrence before compiler capture.
 *
 * <p>A producer retains the exact backend-independent operation reference, an immutable ordered
 * snapshot of exact input tensor references, and an immutable ordered snapshot of exact output
 * descriptor references. It also owns one immutable canonical output tensor for every described
 * position. Repeated references and encounter order remain significant. The operation's selected
 * signature validates the final input and output counts. Ordinary object identity distinguishes
 * separately invoked occurrences even when all retained values are structurally equal; identity
 * hash codes are not stable producer identifiers.</p>
 *
 * <p>Each canonical output retains indexed provenance back to this producer, intentionally
 * forming one immutable {@code Tensor -> TensorProvenance -> TensorProducer -> outputs -> Tensor}
 * cycle. Construction validates the complete occurrence before requesting its first output
 * identity, creates wrappers in ascending output order, and assigns the final output snapshot
 * before any part of the occurrence is returned. Final state therefore supports safe publication
 * of the completed occurrence. Retaining any output may retain every sibling output through this
 * producer; the cycle owns no external resource and remains eligible for ordinary garbage
 * collection when the complete occurrence is unreachable.</p>
 *
 * <p>This pre-capture identity is neither a tensor nor compiled graph state and owns no
 * graph-local node or value identity, gradient or backward state, storage, compiler state,
 * backend metadata, execution state, or runtime behavior.</p>
 */
public final class TensorProducer {
    private final Operation operation;
    private final List<Tensor> inputs;
    private final List<TensorDescriptor> outputDescriptors;
    private final List<Tensor> outputs;

    /**
     * Creates one validated expression producer and its canonical unlabeled output tensors.
     *
     * <p>This package-owned construction seam uses {@link TensorFactory}'s JVM-wide identity
     * allocator so direct tensor-package construction and factory construction share one unique
     * identity sequence. All operation, list, element, non-empty-output, and signature-count
     * validation completes before the first identity is requested. Successful output wrappers are
     * created in ascending index order with the exact slot descriptors, no label or storage, and
     * indexed provenance back to this producer. A later allocation failure does not roll back
     * identifiers already consumed, but no partially constructed occurrence is returned.</p>
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
     * @throws IllegalStateException if the package identity allocator is exhausted; identifiers
     *     allocated for earlier output positions remain consumed
     */
    TensorProducer(
            Operation operation,
            List<Tensor> inputs,
            List<TensorDescriptor> outputDescriptors) {
        this(
                operation,
                inputs,
                outputDescriptors,
                Optional.empty(),
                TensorFactory::nextTensorId);
    }

    /**
     * Creates one validated expression producer and all of its canonical output tensors.
     *
     * <p>Operation, container, element, non-empty-output, and signature-count validation precedes
     * the first call to {@code tensorIdSupplier}. The producer's operation, input snapshot, and
     * descriptor snapshot are already assigned while wrappers are created in ascending output
     * order. Each wrapper receives the supplied identity, exact slot descriptor, the current label
     * policy, no storage, and indexed provenance back to this producer. The supplier is used only
     * during construction and is not retained. No wrapper escapes before the final immutable
     * output snapshot is assigned.</p>
     *
     * @param operation exact non-null immutable operation reference to retain
     * @param inputs non-null ordered input tensor references to snapshot
     * @param outputDescriptors non-null, non-empty ordered output descriptor references to
     *     snapshot
     * @param firstOutputLabel non-null optional label for output zero; other outputs are unlabeled
     * @param tensorIdSupplier non-null source of one fresh identity per output position
     * @throws NullPointerException if a required reference, list element, or supplied identity is
     *     null
     * @throws IllegalArgumentException if output descriptors are empty, occurrence counts are
     *     invalid, or the first output label is blank
     * @throws IllegalStateException if the identity source is exhausted; identifiers already
     *     supplied for earlier output positions are not rolled back
     */
    TensorProducer(
            Operation operation,
            List<Tensor> inputs,
            List<TensorDescriptor> outputDescriptors,
            Optional<String> firstOutputLabel,
            Supplier<TensorId> tensorIdSupplier) {
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

        Objects.requireNonNull(firstOutputLabel, "firstOutputLabel");
        Objects.requireNonNull(tensorIdSupplier, "tensorIdSupplier");
        List<Tensor> outputSnapshot = new ArrayList<>(outputDescriptorSnapshot.size());
        for (int outputIndex = 0;
                outputIndex < outputDescriptorSnapshot.size();
                outputIndex++) {
            outputSnapshot.add(new Tensor(
                    tensorIdSupplier.get(),
                    outputDescriptorSnapshot.get(outputIndex),
                    outputIndex == 0 ? firstOutputLabel : Optional.empty(),
                    Optional.of(new TensorProvenance(this, outputIndex)),
                    Optional.empty()));
        }
        this.outputs = List.copyOf(outputSnapshot);
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

    /**
     * Returns the canonical tensor for one output position.
     *
     * <p>This is an indexed read of retained final state. It performs no allocation, traversal,
     * inference, wrapper reconstruction, or mutation.</p>
     *
     * @param outputIndex zero-based output position, from zero (inclusive) to
     *     {@link #outputCount()} (exclusive)
     * @return the exact non-null canonical tensor owned by this producer at {@code outputIndex}
     * @throws IndexOutOfBoundsException if {@code outputIndex} is negative or is greater than or
     *     equal to the available output count; the message reports both the requested index and
     *     the available count
     */
    public Tensor output(int outputIndex) {
        int count = outputCount();
        if (outputIndex < 0 || outputIndex >= count) {
            throw new IndexOutOfBoundsException(
                    "outputIndex " + outputIndex + " is outside output count " + count);
        }
        return outputs.get(outputIndex);
    }
}
