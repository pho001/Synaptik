package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated storage-free floating-classification Tensor expressions.
 *
 * <p>This package-private boundary accepts only floating input data types and produces fixed BOOL
 * descriptors with the exact input shape, unresolved layout, and disabled gradient eligibility.
 * It records one parameterless classification producer without inspecting input storage or
 * values, calculating classifications, inserting casts, capturing a graph, defining gradients,
 * or executing work. Every successful result is fresh, unlabeled, storage-free, and has
 * provenance output index zero over the exact input reference.</p>
 */
final class TensorFloatingClassifications {
    /** Prevents instantiation because expression construction is stateless and package-local. */
    private TensorFloatingClassifications() {
    }

    /**
     * Creates one fresh derived Tensor for a floating classification request.
     *
     * <p>The result describes future elementwise classification values; this method neither reads
     * an attached host-storage value nor makes the classification result available eagerly.</p>
     *
     * @param input non-null floating Tensor retained by exact reference in result provenance
     * @param kind non-null parameterless classification kind retained in the result operation
     * @return the non-null exact fresh BOOL Tensor returned by the central factory
     * @throws NullPointerException if {@code input} or {@code kind} is null, checked in that order
     *     with the parameter name as the message
     * @throws IllegalArgumentException if the input data type is not floating, with the exact
     *     rejected data type in the message
     * @throws IllegalStateException if Tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor apply(Tensor input, FloatingClassificationKind kind) {
        return applyChecked(input, kind, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null floating Tensor retained by exact reference in result provenance
     * @param kind non-null parameterless classification kind retained in the result operation
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input} or {@code kind} is null, checked in that order
     *     with the parameter name as the message
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if the input data type is not floating, with the exact
     *     rejected data type in the message
     * @throws IllegalStateException if Tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor applyChecked(Tensor input, FloatingClassificationKind kind,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return applyDescriptors(
                input == null ? null : input.descriptor(),
                kind,
                descriptorCheck,
                input);
    }

    /**
     * Completes ordinary checked semantics synchronously without creating Tensor identities.
     * The same descriptor guard core serves real construction; the sink is never retained.
     * @param operation non-null ordinary operation selecting this family and its attributes
     * @param inputs non-null ordered operand descriptors, including repeated occurrences
     * @param outputCount exact canonical sibling count
     * @param sink non-null consumer called exactly once with checked semantics and all outputs
     * @throws NullPointerException if a required reference is null
     * @throws IllegalArgumentException if cardinality or ordinary semantic guards reject the request
     */
    static void describeChecked(Operation operation, List<TensorDescriptor> inputs, int outputCount,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> sink) {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(sink, "sink");
        if (inputs.size() != 1 || outputCount != 1) {
            throw new IllegalArgumentException("invalid ordinary descriptor cardinality");
        }
        for (TensorDescriptor input : inputs) {
            Objects.requireNonNull(input, "input descriptor");
        }
        applyDescriptors(inputs.get(0), (FloatingClassificationKind) operation.kind(), sink, null);
    }

    /**
     * Runs the shared descriptor guards; provenance references are used only for real construction.
     * @param input descriptor supplying ordinary operand metadata
     * @param kind ordinary checked construction parameter
     * @param descriptorCheck synchronous, unretained sink; null selects real construction
     * @param originalInput exact real operand, or null for descriptor completion
     * @return the real result, or null after synchronous descriptor completion
     */
    private static Tensor applyDescriptors(
            TensorDescriptor input,
            FloatingClassificationKind kind,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalInput) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");

        DataType dataType = input.dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "input must be a floating data type, but was " + dataType);
        }

        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.BOOL,
                input.shape(),
                Optional.empty(),
                false);
        Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(originalInput));
    }
}
