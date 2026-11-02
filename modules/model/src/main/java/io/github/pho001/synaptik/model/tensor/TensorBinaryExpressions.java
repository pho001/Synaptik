package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated storage-free binary arithmetic tensor expressions.
 *
 * <p>This package-private boundary owns the common deterministic construction order for all seven
 * parameterless binary arithmetic kinds. It validates operand and kind presence, promotes
 * same-category floating or signed-integral data types, rejects integral division and power,
 * derives only locally provable right-aligned broadcast shape, records an unresolved result
 * layout and gradient-eligibility OR, and delegates exact ordered producer
 * inputs once to the central Tensor factory. It does not inspect or mutate storage,
 * execute arithmetic, canonicalize expressions, insert casts, create gradient rules, or capture a
 * graph.</p>
 */
final class TensorBinaryExpressions {
    /** Prevents instantiation because expression construction is stateless and package-local. */
    private TensorBinaryExpressions() {
    }

    /**
     * Creates one fresh derived tensor for an ordered binary arithmetic request.
     *
     * <p>Validation and construction occur in this exact order: null-check {@code left},
     * {@code right}, and {@code kind}; promote the two descriptor data types; reject integral
     * {@code DIV} or {@code POW}; broadcast the two descriptor shapes; create one
     * unresolved-layout descriptor with input gradient eligibility
     * combined by logical OR; create one operation from the exact supplied kind and
     * {@code NoOperationAttrs.INSTANCE}; and delegate ordered exact producer inputs
     * {@code [left, right]} once to
     * {@link TensorFactory#createDerived(TensorDescriptor, Optional, Operation, List)} with no
     * label. Failures before the final delegation allocate no Tensor identity. A successful call
     * returns the factory's exact fresh, unlabeled, storage-free result; neither input nor its
     * metadata or storage is mutated.</p>
     *
     * @param left non-null ordered left operand retained by exact reference in provenance
     * @param right non-null ordered right operand retained by exact reference in provenance
     * @param kind non-null parameterless binary arithmetic semantic kind retained in the result
     *     operation
     * @return the non-null exact fresh derived tensor returned by the central factory
     * @throws NullPointerException if {@code left}, {@code right}, or {@code kind} is null,
     *     checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if an operand is boolean, the operands cross the floating
     *     and integral categories, an integral request is {@code DIV} or {@code POW}, or the
     *     shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor apply(Tensor left, Tensor right, BinaryArithmeticKind kind) {
        return applyChecked(left, right, kind, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param left non-null ordered left operand retained by exact reference in provenance
     * @param right non-null ordered right operand retained by exact reference in provenance
     * @param kind non-null parameterless binary arithmetic semantic kind retained in the result
     *     operation
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code left}, {@code right}, or {@code kind} is null,
     *     checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if an operand is boolean, the operands cross the floating
     *     and integral categories, an integral request is {@code DIV} or {@code POW}, or the
     *     shapes cannot be broadcast under the local shape contract
     * @throws IllegalStateException if tensor identifier space is exhausted after local model
     *     values have been constructed
     */
    static Tensor applyChecked(Tensor left, Tensor right, BinaryArithmeticKind kind,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return applyDescriptors(
                left == null ? null : left.descriptor(),
                right == null ? null : right.descriptor(),
                kind,
                descriptorCheck,
                left,
                right);
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
        if (inputs.size() != 2 || outputCount != 1) {
            throw new IllegalArgumentException("invalid ordinary descriptor cardinality");
        }
        for (TensorDescriptor input : inputs) {
            Objects.requireNonNull(input, "input descriptor");
        }
        applyDescriptors(inputs.get(0), inputs.get(1), (BinaryArithmeticKind) operation.kind(), sink, null, null);
    }

    /**
     * Runs the shared descriptor guards; provenance references are used only for real construction.
     * @param left descriptor supplying ordinary operand metadata
     * @param right descriptor supplying ordinary operand metadata
     * @param kind ordinary checked construction parameter
     * @param descriptorCheck synchronous, unretained sink; null selects real construction
     * @param originalLeft exact real operand, or null for descriptor completion
     * @param originalRight exact real operand, or null for descriptor completion
     * @return the real result, or null after synchronous descriptor completion
     */
    private static Tensor applyDescriptors(
            TensorDescriptor left,
            TensorDescriptor right,
            BinaryArithmeticKind kind,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalLeft,
            Tensor originalRight) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        Objects.requireNonNull(kind, "kind");

        DataType dataType = DataTypePromotion.promoteNumeric(
                left.dataType(), right.dataType());
        if (dataType.isIntegral()
                && (kind == BinaryArithmeticKind.DIV || kind == BinaryArithmeticKind.POW)) {
            throw new IllegalArgumentException(kind + " does not support integral data types");
        }
        Shape shape = ShapeBroadcast.broadcast(
                left.shape(), right.shape());
        TensorDescriptor descriptor = new TensorDescriptor(
                dataType,
                shape,
                Optional.empty(),
                left.requiresGrad() || right.requiresGrad());
        Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(originalLeft, originalRight));
    }
}
