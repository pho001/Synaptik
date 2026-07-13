package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free, one-axis cumulative-scan expressions.
 *
 * <p>A cumulative scan preserves logical shape while accumulating addition or multiplication
 * along one axis. Inclusive mode includes the current position; exclusive mode omits it and gives
 * the first traversed position the selected kind's identity. Reverse mode changes traversal
 * direction without reversing output positions.</p>
 *
 * <p>This package-private boundary accepts exactly FLOAT64, FLOAT32, BFLOAT16, INT32, and INT64
 * inputs, normalizes one axis, preserves the exact input Shape, data type, and gradient-eligibility
 * metadata, and records exact one-input provenance. It does not inspect or accumulate values,
 * preserve resolved layout,
 * allocate storage, define gradient rules, capture a graph, or provide compiler, runtime,
 * backend, or execution behavior.</p>
 */
final class TensorCumulativeScanExpressions {
    /** Prevents instantiation because cumulative-scan expression construction owns no state. */
    private TensorCumulativeScanExpressions() {
    }

    /**
     * Validates local input metadata and creates one fresh cumulative-scan expression.
     *
     * <p>Validation checks {@code input}, then {@code kind}, then accepts exactly floating or
     * integral data types, reads the exact input Shape, normalizes {@code axis} exactly once, and
     * constructs one {@link CumulativeScanAttrs}. Failures before factory delegation consume no
     * Tensor identity.</p>
     *
     * @param input non-null FLOAT64, FLOAT32, BFLOAT16, INT32, or INT64 Tensor retained as the
     *     sole provenance input
     * @param kind non-null cumulative arithmetic kind retained by the operation
     * @param axis positive or negative input axis accepted by {@link Shape#normalizeAxis(int)}
     * @param exclusive {@code true} to omit each current position from its traversed prefix, or
     *     {@code false} to include it
     * @param reverse {@code true} to traverse from the axis end without changing output order, or
     *     {@code false} to traverse forward
     * @return the non-null fresh storage-free expression with exact Shape, type, gradient
     *     eligibility, normalized attributes, and one-input provenance
     * @throws NullPointerException if {@code input} or {@code kind} is null, with the matching
     *     parameter name as message; input validation occurs first
     * @throws IllegalArgumentException if the input is not floating or integral, with message
     *     {@code input must have a numeric data type, but was <dataType>}
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input Shape, including
     *     every axis when the input is scalar
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    static Tensor apply(
            Tensor input,
            CumulativeScanKind kind,
            int axis,
            boolean exclusive,
            boolean reverse) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        validateNumericInput(input);
        Shape inputShape = input.descriptor().shape();
        int normalizedAxis = inputShape.normalizeAxis(axis);
        CumulativeScanAttrs attrs =
                new CumulativeScanAttrs(normalizedAxis, exclusive, reverse);
        return create(input, inputShape, kind, attrs);
    }

    /**
     * Accepts exactly floating or integral input metadata without conversion or promotion.
     *
     * @param input non-null Tensor whose immutable descriptor supplies the data type
     * @throws IllegalArgumentException if the input type is neither floating nor integral, with
     *     message {@code input must have a numeric data type, but was <dataType>}
     */
    private static void validateNumericInput(Tensor input) {
        DataType dataType = input.descriptor().dataType();
        if (!dataType.isFloating() && !dataType.isIntegral()) {
            throw new IllegalArgumentException(
                    "input must have a numeric data type, but was " + dataType);
        }
    }

    /**
     * Constructs one exact descriptor and operation, then delegates one derived Tensor identity.
     *
     * @param input validated Tensor retained as the exact sole provenance input
     * @param shape exact non-null immutable input Shape retained by the result descriptor
     * @param kind non-null exact scan kind retained by the operation
     * @param attrs non-null normalized attributes retained by exact reference in the operation
     * @return the non-null fresh unresolved-layout, unlabeled, storage-free Tensor returned by the
     *     central factory
     * @throws IllegalStateException if tensor identifier space is exhausted after descriptor,
     *     operation, and provenance construction
     */
    private static Tensor create(
            Tensor input,
            Shape shape,
            CumulativeScanKind kind,
            CumulativeScanAttrs attrs) {
        TensorDescriptor descriptor = new TensorDescriptor(
                input.descriptor().dataType(),
                shape,
                Optional.empty(),
                input.descriptor().requiresGrad());
        Operation operation = new Operation(kind, attrs);
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(input));
    }
}
