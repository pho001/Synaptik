package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs storage-free contiguous-layout request expressions from local Tensor metadata.
 *
 * <p>This package-private boundary preserves the exact input Shape, DataType, and gradient-
 * eligibility value. It resolves newly created canonical row-major geometry only when the Shape
 * is fully static and otherwise leaves layout unresolved. It never inspects input layout, label,
 * provenance, host storage, liveness, or values and performs no storage allocation, copy,
 * canonicalization, graph capture, materialization planning, lowering, or execution.</p>
 */
final class TensorContiguousExpressions {
    /** Prevents instantiation because contiguous-expression construction owns no state. */
    private TensorContiguousExpressions() {
    }

    /**
     * Creates one fresh contiguous-layout request from the supplied input.
     *
     * <p>The input is null-checked before metadata access. Its descriptor and exact Shape are read,
     * then the Shape is tested once for complete static resolution. A static Shape produces one
     * new {@link LayoutDescriptor#contiguous(Shape)} value; a dynamic Shape produces unresolved
     * layout. One result descriptor and exact parameterless operation are then passed with the
     * sole producer input to one factory call, which creates provenance and the derived Tensor.</p>
     *
     * <p>The returned Tensor retains the exact input Shape, DataType, and gradient-eligibility
     * value, has no label or storage, and records exact {@link ContiguousKind#CONTIGUOUS} and
     * {@link NoOperationAttrs#INSTANCE} semantics with ordered input {@code [input]}. A valid call
     * always returns a fresh identity, including for an already-contiguous or nested request.</p>
     *
     * @param input non-null Tensor retained by exact reference as the sole provenance input; it is
     *     not mutated and none of its storage or value state is read
     * @return the non-null fresh Tensor returned by the single derived-factory invocation
     * @throws NullPointerException if {@code input} is null, with message {@code input}; no Tensor
     *     identity is consumed
     * @throws ArithmeticException if canonical layout stride or referenced-span arithmetic
     *     overflows for a fully static Shape; this occurs before Tensor identity allocation
     * @throws IllegalStateException if tensor identifier space is exhausted after descriptor,
     *     operation, and provenance construction
     */
    static Tensor apply(Tensor input) {
        return applyChecked(input, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null Tensor retained by exact reference as the sole provenance input; it is
     *     not mutated and none of its storage or value state is read
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input} is null, with message {@code input}; no Tensor
     *     identity is consumed
     * @throws ArithmeticException if canonical layout stride or referenced-span arithmetic
     *     overflows for a fully static Shape; this occurs before Tensor identity allocation
     * @throws IllegalStateException if tensor identifier space is exhausted after descriptor,
     *     operation, and provenance construction
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata
     */
    static Tensor applyChecked(Tensor input,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return applyDescriptors(
                input == null ? null : input.descriptor(),
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
        applyDescriptors(inputs.get(0), sink, null);
    }

    /**
     * Runs the shared descriptor guards; provenance references are used only for real construction.
     * @param input descriptor supplying ordinary operand metadata
     * @param descriptorCheck synchronous, unretained sink; null selects real construction
     * @param originalInput exact real operand, or null for descriptor completion
     * @return the real result, or null after synchronous descriptor completion
     */
    private static Tensor applyDescriptors(
            TensorDescriptor input,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalInput) {
        Objects.requireNonNull(input, "input");
        TensorDescriptor inputDescriptor = input;
        Shape shape = inputDescriptor.shape();
        Optional<LayoutDescriptor> layout = shape.isFullyStatic()
                ? Optional.of(LayoutDescriptor.contiguous(shape))
                : Optional.empty();
        TensorDescriptor descriptor = new TensorDescriptor(
                inputDescriptor.dataType(),
                shape,
                layout,
                inputDescriptor.requiresGrad());
        Operation operation = new Operation(
                ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(originalInput));
    }
}
