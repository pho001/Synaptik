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
     * <p>Validation null-checks {@code input} and then {@code kind}, requires FLOAT64, FLOAT32, or
     * BFLOAT16 input, reads the exact input Shape, normalizes {@code axis} exactly once, and creates
     * one {@link SoftmaxAttrs} before common construction. Type rejection precedes axis validation,
     * and every failure before factory delegation consumes no Tensor identity.</p>
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
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        validateFloatingInput(input);
        Shape inputShape = input.descriptor().shape();
        int normalizedAxis = inputShape.normalizeAxis(axis);
        SoftmaxAttrs attrs = new SoftmaxAttrs(normalizedAxis);
        return create(input, kind, inputShape, attrs);
    }

    /**
     * Requires one of the three current floating input data types without conversion or promotion.
     *
     * @param input non-null Tensor whose immutable descriptor supplies the data type
     * @throws IllegalArgumentException if the input type is not floating, with message
     *     {@code input must have a floating data type, but was <dataType>}
     */
    private static void validateFloatingInput(Tensor input) {
        DataType dataType = input.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "input must have a floating data type, but was " + dataType);
        }
    }

    /**
     * Constructs one exact descriptor and operation, then delegates one derived Tensor identity.
     *
     * <p>The descriptor retains the exact input data type, supplied Shape reference, and unchanged
     * gradient eligibility while leaving layout unresolved. The operation retains the exact kind
     * and attributes references, and the central factory receives exact ordered producer input
     * {@code [input]} exactly once with no label or storage. The factory creates index-zero
     * provenance. Construction reads no input label, provenance, layout geometry, storage, Shape
     * dimensions, element count, or values.</p>
     *
     * @param input validated Tensor retained as the exact sole provenance input
     * @param kind exact non-null normalization kind retained by the operation
     * @param shape exact non-null immutable input Shape retained by the result descriptor
     * @param attrs non-null normalized-axis attributes retained by exact reference
     * @return the non-null fresh unresolved-layout, unlabeled, storage-free Tensor returned by the
     *     central factory
     * @throws IllegalStateException if tensor identifier space is exhausted after descriptor,
     *     operation, and provenance construction
     */
    private static Tensor create(
            Tensor input, SoftmaxKind kind, Shape shape, SoftmaxAttrs attrs) {
        TensorDescriptor descriptor = new TensorDescriptor(
                input.descriptor().dataType(),
                shape,
                Optional.empty(),
                input.descriptor().requiresGrad());
        Operation operation = new Operation(kind, attrs);
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(input));
    }
}
