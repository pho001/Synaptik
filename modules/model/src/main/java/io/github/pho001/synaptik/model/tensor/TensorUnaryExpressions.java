package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated storage-free unary elementwise tensor expressions.
 *
 * <p>This package-private boundary owns the common deterministic construction order for all
 * nineteen parameterless unary elementwise kinds. It validates input and kind presence, accepts
 * only a floating input data type, retains the exact data type and shape, records unresolved
 * result layout and unchanged gradient eligibility, and delegates the exact sole producer input
 * once to the central Tensor factory. It does not inspect values or storage, evaluate
 * mathematics, validate numerical domains, decompose first-class transforms, choose an accuracy
 * policy, canonicalize expressions, insert casts, create gradient rules, or capture a graph.</p>
 */
final class TensorUnaryExpressions {
    /** Prevents instantiation because expression construction is stateless and package-local. */
    private TensorUnaryExpressions() {
    }

    /**
     * Creates one fresh derived tensor for a unary elementwise request.
     *
     * <p>Validation and construction occur in this exact order: null-check {@code input} and
     * {@code kind}; read and validate the input descriptor's floating data type; create one
     * unresolved-layout descriptor with the exact input data type, shape reference, and gradient
     * eligibility; create one operation from the exact supplied kind and
     * {@code NoOperationAttrs.INSTANCE}; and delegate the exact sole producer input once to
     * {@link TensorFactory#createDerived(TensorDescriptor, Optional, Operation, List)} with no
     * label. Failures before the final delegation allocate no Tensor identity. A successful call
     * returns the factory's exact fresh, unlabeled, storage-free result; the input and all of its
     * metadata and storage remain unchanged.</p>
     *
     * @param input non-null floating tensor retained by exact reference in result provenance
     * @param kind non-null parameterless unary semantic kind retained in the result operation
     * @return the non-null exact fresh derived tensor returned by the central factory
     * @throws NullPointerException if {@code input} or {@code kind} is null, checked in that order
     *     with the parameter name as the message
     * @throws IllegalArgumentException if the input data type is not floating, with the exact
     *     rejected data type in the message
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor apply(Tensor input, UnaryElementwiseKind kind) {
        return applyChecked(input, kind, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null floating tensor retained by exact reference in result provenance
     * @param kind non-null parameterless unary semantic kind retained in the result operation
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input} or {@code kind} is null, checked in that order
     *     with the parameter name as the message
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if the input data type is not floating, with the exact
     *     rejected data type in the message
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor applyChecked(Tensor input, UnaryElementwiseKind kind,
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
        applyDescriptors(inputs.get(0), (UnaryElementwiseKind) operation.kind(), sink, null);
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
            UnaryElementwiseKind kind,
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
                dataType,
                input.shape(),
                Optional.empty(),
                input.requiresGrad());
        Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(originalInput));
    }
}
