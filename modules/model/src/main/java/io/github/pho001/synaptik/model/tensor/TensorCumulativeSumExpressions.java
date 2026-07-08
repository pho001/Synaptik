package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.scan.CumulativeSumAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeSumKind;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free, one-axis cumulative-sum expressions.
 *
 * <p>A cumulative sum is a shape-preserving scan: it associates every output position with a
 * prefix of positions along one axis. Inclusive mode includes the current position; exclusive
 * mode omits it and gives the first traversed position an additive-zero boundary. Forward mode
 * traverses from the axis beginning, while reverse mode traverses from its end without reversing
 * output order. For logical input {@code [1, 2, 3]}, inclusive forward produces
 * {@code [1, 3, 6]} from {@code 1}, {@code 1 + 2}, and {@code 1 + 2 + 3}. Exclusive forward
 * produces {@code [0, 1, 3]} from the empty prefix, {@code 1}, and {@code 1 + 2}. Inclusive
 * reverse produces {@code [6, 5, 3]} from {@code 1 + 2 + 3}, {@code 2 + 3}, and {@code 3}.
 * Exclusive reverse produces {@code [5, 3, 0]} from {@code 2 + 3}, {@code 3}, and the empty
 * reverse prefix.</p>
 *
 * <p>This package-private boundary validates only local model metadata. It accepts floating and
 * integral input types, normalizes one axis, preserves the exact input Shape, data type, and
 * gradient-eligibility metadata, and records exact one-input provenance. It does not inspect or
 * accumulate values, preserve resolved layout, allocate storage, define numerical or gradient
 * rules, capture a graph, or provide compiler, runtime, backend, or execution behavior.</p>
 */
final class TensorCumulativeSumExpressions {
    /** Prevents instantiation because cumulative-sum expression construction owns no state. */
    private TensorCumulativeSumExpressions() {
    }

    /**
     * Validates local input metadata and creates one fresh cumulative-sum expression.
     *
     * <p>Validation null-checks {@code input}, accepts exactly floating or integral data types,
     * reads the exact input Shape, normalizes {@code axis} exactly once, and constructs one
     * {@link CumulativeSumAttrs} before delegating to common construction. BOOL rejection occurs
     * before axis validation. Failures before factory delegation consume no Tensor identity.</p>
     *
     * @param input non-null floating or integral Tensor retained as the sole provenance input
     * @param axis positive or negative input axis accepted by {@link Shape#normalizeAxis(int)}
     * @param exclusive {@code true} to omit each current position from its traversed prefix, or
     *     {@code false} to include it
     * @param reverse {@code true} to traverse from the axis end without changing output order, or
     *     {@code false} to traverse forward
     * @return the non-null fresh storage-free expression with exact Shape, type, gradient
     *     eligibility, normalized attributes, and one-input provenance
     * @throws NullPointerException if {@code input} is null, with message {@code input}
     * @throws IllegalArgumentException if the input is not floating or integral, with message
     *     {@code input must have a numeric data type, but was <dataType>}
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input Shape, including
     *     every axis when the input is scalar
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    static Tensor apply(Tensor input, int axis, boolean exclusive, boolean reverse) {
        Objects.requireNonNull(input, "input");
        validateNumericInput(input);
        Shape inputShape = input.descriptor().shape();
        int normalizedAxis = inputShape.normalizeAxis(axis);
        CumulativeSumAttrs attrs =
                new CumulativeSumAttrs(normalizedAxis, exclusive, reverse);
        return create(input, inputShape, attrs);
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
     * <p>Construction creates an unresolved-layout descriptor from the exact input data type,
     * supplied exact Shape, and unchanged gradient eligibility. It then creates one
     * {@link CumulativeSumKind#CUM_SUM} operation retaining {@code attrs} and delegates exact
     * ordered producer input {@code [input]} once to the central derived factory with no label.
     * The factory creates index-zero provenance. This method reads no label, existing provenance,
     * layout, storage, shape dimensions, element count, or values.</p>
     *
     * @param input validated Tensor retained as the exact sole provenance input
     * @param shape exact non-null immutable input Shape retained by the result descriptor
     * @param attrs non-null normalized attributes retained by exact reference in the operation
     * @return the non-null fresh unresolved-layout, unlabeled, storage-free Tensor returned by the
     *     central factory
     * @throws IllegalStateException if tensor identifier space is exhausted after descriptor,
     *     operation, and provenance construction
     */
    private static Tensor create(Tensor input, Shape shape, CumulativeSumAttrs attrs) {
        TensorDescriptor descriptor = new TensorDescriptor(
                input.descriptor().dataType(),
                shape,
                Optional.empty(),
                input.descriptor().requiresGrad());
        Operation operation = new Operation(CumulativeSumKind.CUM_SUM, attrs);
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(input));
    }
}
