package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free softmax and log-softmax expressions.
 *
 * <p>A normalization slice contains positions that differ only along one axis while all other
 * coordinates remain fixed. For {@code [1, 2, 3]}, ideal softmax probabilities are approximately
 * {@code [0.09003057, 0.24472847, 0.66524096]} and sum to one. Ideal log-softmax values are
 * approximately {@code [-2.40760596, -1.40760596, -0.40760596]}; exponentiating them reconstructs
 * the softmax probabilities. Both meanings preserve every logical input position and Shape.</p>
 *
 * <p>This package-private boundary accepts only floating input metadata, normalizes one axis, and
 * records the requested first-class kind with exact one-input provenance. It does not inspect
 * values or storage, calculate probabilities, select a numerical algorithm, preserve resolved
 * layout, decompose an operation, define gradients, capture a graph, or provide compiler, backend,
 * runtime, or execution behavior.</p>
 */
final class TensorSoftmaxExpressions {
    /** Prevents instantiation because softmax expression construction owns no state. */
    private TensorSoftmaxExpressions() {
    }

    /**
     * Validates local input metadata and creates one fresh normalization expression.
     *
     * <p>Validation null-checks {@code input} and then {@code kind}, requires FLOAT64, FLOAT32,
     * BFLOAT16, FLOAT16 or OFP8 input, reads the exact input Shape, normalizes {@code axis} exactly
     * once, and creates one {@link SoftmaxAttrs} before common construction. Type rejection
     * precedes axis validation, and every failure before factory delegation consumes no Tensor
     * identity.</p>
     *
     * @param input non-null floating Tensor retained as the sole provenance input
     * @param kind non-null exact {@link SoftmaxKind#SOFTMAX} or
     *     {@link SoftmaxKind#LOG_SOFTMAX} identity retained by the result operation
     * @param axis positive or negative input axis accepted by {@link Shape#normalizeAxis(int)}
     * @return a non-null fresh unresolved-layout, unlabeled, storage-free Tensor preserving exact
     *     Shape, type, gradient eligibility, requested kind, and one-input provenance
     * @throws NullPointerException if {@code input} is null, with message {@code input}, or if
     *     {@code kind} is null, with message {@code kind}; input validation occurs first
     * @throws IllegalArgumentException if the input is not floating, with message
     *     {@code input must have a floating data type, but was <dataType>}
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input Shape, including
     *     every axis for scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    static Tensor apply(Tensor input, SoftmaxKind kind, int axis) {
        return applyChecked(input, kind, axis, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null floating Tensor retained as the sole provenance input
     * @param kind non-null exact {@link SoftmaxKind#SOFTMAX} or
     *     {@link SoftmaxKind#LOG_SOFTMAX} identity retained by the result operation
     * @param axis positive or negative input axis accepted by {@link Shape#normalizeAxis(int)}
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input} is null, with message {@code input}, or if
     *     {@code kind} is null, with message {@code kind}; input validation occurs first
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if the input is not floating, with message
     *     {@code input must have a floating data type, but was <dataType>}
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input Shape, including
     *     every axis for scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    static Tensor applyChecked(Tensor input, SoftmaxKind kind, int axis,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        return applyDescriptors(
                input.descriptor(), kind, axis, descriptorCheck, input);
    }

    /**
     * Runs the shared descriptor guards and completes metadata before real provenance is used.
     * @param input immutable operand metadata
     * @param kind ordinary construction argument, validated in existing order
     * @param axis ordinary construction argument, validated in existing order
     * @param descriptorCheck optional synchronous, unretained metadata completion; null constructs a Tensor
     * @param inputTensor real provenance, unused when metadata completion is supplied
     * @return null after metadata completion, otherwise the ordinary result
     */
    private static Tensor applyDescriptors(
            TensorDescriptor input,
            SoftmaxKind kind,
            int axis,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor inputTensor) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        validateFloatingInput(input);
        Shape inputShape = input.shape();
        int normalizedAxis = inputShape.normalizeAxis(axis);
        SoftmaxAttrs attrs = new SoftmaxAttrs(normalizedAxis);
        return createDescriptors(input, kind, inputShape, attrs, descriptorCheck, inputTensor);
    }

    /**
     * Requires one of the four current floating input data types without conversion or promotion.
     *
     * @param input non-null descriptor supplying the data type
     * @throws IllegalArgumentException if the input type is not floating, with message
     *     {@code input must have a floating data type, but was <dataType>}
     */
    private static void validateFloatingInput(TensorDescriptor input) {
        DataType dataType = input.dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "input must have a floating data type, but was " + dataType);
        }
    }


    /**
     * Runs the shared descriptor guards and completes metadata before real provenance is used.
     * @param input immutable operand metadata
     * @param kind ordinary construction argument, validated in existing order
     * @param shape ordinary construction argument, validated in existing order
     * @param attrs ordinary construction argument, validated in existing order
     * @param descriptorCheck optional synchronous, unretained metadata completion; null constructs a Tensor
     * @param inputTensor real provenance, unused when metadata completion is supplied
     * @return null after metadata completion, otherwise the ordinary result
     */
    private static Tensor createDescriptors(
            TensorDescriptor input,
            SoftmaxKind kind,
            Shape shape,
            SoftmaxAttrs attrs,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor inputTensor) {
        TensorDescriptor descriptor = new TensorDescriptor(
                input.dataType(),
                shape,
                Optional.empty(),
                input.requiresGrad());
        Operation operation = new Operation(kind, attrs);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(inputTensor));
    }

    /**
     * Completes one admitted ordinary occurrence synchronously using the real construction guards.
     * No Tensor or identity is created and the callback is not retained. The owning facade checks
     * occurrence cardinality and compares the returned canonical operation with the supplied one.
     *
     * @param operation non-null ordinary operation admitted to this family by the owning selector
     * @param inputs non-null immutable operand metadata in canonical input order
     * @param outputCount complete canonical output count, already validated by the owning facade
     * @param sink non-null synchronous consumer of the canonical operation and all output metadata
     * @throws IllegalArgumentException if ordinary semantic guards or the consumer reject metadata
     */
    static void describeChecked(Operation operation, List<TensorDescriptor> inputs, int outputCount,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> sink) {
        Objects.requireNonNull(sink, "sink");
        SoftmaxAttrs attrs = (SoftmaxAttrs) operation.attrs();
        applyDescriptors(inputs.get(0), (SoftmaxKind) operation.kind(), attrs.axis(), sink, null);
    }
}
