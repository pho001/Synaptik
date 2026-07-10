package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaAttrs;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free, single-axis arg-min and arg-max expressions.
 *
 * <p>Both families accept FLOAT64, FLOAT32, BFLOAT16, INT32, or INT64 input and produce a fresh
 * fixed-INT64, false-gradient result. Integral candidates use signed order. Floating candidates
 * prefer NaN over every non-NaN for both extrema directions, treat multiple NaNs as ties, order
 * negative zero below positive zero, and order infinities normally. The explicit policy then
 * selects the smallest or largest logical coordinate among equal candidates. A statically empty
 * selected axis is invalid because it has no logical index; an empty unselected axis and an
 * unbound selected extent remain structurally valid.</p>
 *
 * <p>Construction records metadata only. It does not read values or storage, select an index,
 * implement an algorithm, execute an operation, create gradient rules, capture a graph, or
 * provide compiler, runtime, or backend behavior.</p>
 */
final class TensorArgExtremaExpressions {
    /** Prevents instantiation because arg-extrema construction is stateless. */
    private TensorArgExtremaExpressions() {
    }

    /**
     * Validates and creates one fresh fixed-INT64 arg-extrema expression.
     *
     * <p>Validation checks input, kind, policy, supported kind, numeric input, axis, and static
     * selected-axis non-emptiness in that order. It normalizes the axis once, derives the result
     * Shape, constructs exact shared attributes, and delegates once to the central factory.
     * Every local failure consumes no Tensor identity.</p>
     *
     * @param input non-null floating or integral tensor retained as the sole producer input
     * @param kind non-null {@code ARG_MIN} or {@code ARG_MAX} kind
     * @param axis positive or negative input axis accepted by {@link Shape#normalizeAxis(int)}
     * @param keepDimensions whether to retain the selected result axis with extent one
     * @param tiePolicy non-null explicit logical-index tie policy
     * @return a non-null fresh unlabeled, storage-free INT64 expression with false gradient
     *     eligibility, unresolved layout, the requested result Shape, exact one-input producer
     *     provenance, and output index zero
     * @throws NullPointerException if {@code input}, {@code kind}, or {@code tiePolicy} is null,
     *     checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if {@code kind} is neither arg-extrema kind, with message
     *     {@code kind must be ARG_MIN or ARG_MAX, but was <kind>}; if the input type is BOOL, with
     *     message {@code input must have a numeric data type, but was BOOL}; or if the normalized
     *     selected axis has static extent zero, with message {@code arg-extrema reduction axis
     *     must be non-empty, but axis <normalizedAxis> has static extent 0}
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input Shape, including
     *     every axis for a scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after local metadata
     *     construction
     */
    static Tensor apply(
            Tensor input,
            AggregateReductionKind kind,
            int axis,
            boolean keepDimensions,
            ArgExtremaTiePolicy tiePolicy) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(tiePolicy, "tiePolicy");
        validateKind(kind);
        validateNumericInput(input);

        Shape inputShape = input.descriptor().shape();
        int normalizedAxis = inputShape.normalizeAxis(axis);
        if (inputShape.dimensions().get(normalizedAxis).staticSize().orElse(-1L) == 0L) {
            throw new IllegalArgumentException(
                    "arg-extrema reduction axis must be non-empty, but axis "
                            + normalizedAxis
                            + " has static extent 0");
        }
        Shape shape = reduceShape(inputShape, normalizedAxis, keepDimensions);
        ArgExtremaAttrs attrs =
                new ArgExtremaAttrs(normalizedAxis, keepDimensions, tiePolicy);
        return create(input, kind, shape, attrs);
    }

    /**
     * Restricts construction to the two arg-extrema kinds.
     *
     * @param kind non-null aggregate kind already checked by the package-private entry
     * @throws IllegalArgumentException if {@code kind} is neither arg-extrema kind, with message
     *     {@code kind must be ARG_MIN or ARG_MAX, but was <kind>}
     */
    private static void validateKind(AggregateReductionKind kind) {
        if (kind != AggregateReductionKind.ARG_MIN
                && kind != AggregateReductionKind.ARG_MAX) {
            throw new IllegalArgumentException(
                    "kind must be ARG_MIN or ARG_MAX, but was " + kind);
        }
    }

    /**
     * Requires floating or signed-integral input without promotion or conversion.
     *
     * @param input non-null tensor whose descriptor supplies the exact input type
     * @throws IllegalArgumentException if the input type is BOOL, with message
     *     {@code input must have a numeric data type, but was BOOL}
     */
    private static void validateNumericInput(Tensor input) {
        DataType type = input.descriptor().dataType();
        if (!type.isFloating() && !type.isIntegral()) {
            throw new IllegalArgumentException(
                    "input must have a numeric data type, but was " + type);
        }
    }

    /**
     * Derives an axis-removing or retained-axis Shape while preserving unaffected dimensions.
     *
     * <p>Retention allocates one same-rank dimension array and replaces only the selected
     * position with a new {@link StaticDimension} of extent one. Removal allocates one
     * rank-minus-one array and preserves all nonselected references in order. One
     * {@link Shape#ofDimensions(Dimension...)} call completes either branch; rank-one removal
     * returns the canonical scalar Shape.</p>
     *
     * @param inputShape non-null immutable input Shape whose axis is already normalized
     * @param normalizedAxis valid non-negative selected position in {@code inputShape}
     * @param keepDimensions {@code true} to retain the selected position with extent one, or
     *     {@code false} to remove it
     * @return the non-null derived immutable Shape, retaining every unaffected exact Dimension
     *     reference
     */
    private static Shape reduceShape(
            Shape inputShape, int normalizedAxis, boolean keepDimensions) {
        List<Dimension> dimensions = inputShape.dimensions();
        if (keepDimensions) {
            Dimension[] result = new Dimension[dimensions.size()];
            for (int index = 0; index < dimensions.size(); index++) {
                result[index] = dimensions.get(index);
            }
            result[normalizedAxis] = new StaticDimension(1);
            return Shape.ofDimensions(result);
        }
        Dimension[] result = new Dimension[dimensions.size() - 1];
        for (int inputIndex = 0, outputIndex = 0; inputIndex < dimensions.size(); inputIndex++) {
            if (inputIndex != normalizedAxis) {
                result[outputIndex++] = dimensions.get(inputIndex);
            }
        }
        return Shape.ofDimensions(result);
    }

    /**
     * Creates the fixed descriptor, exact operation, and one-output provenance.
     *
     * <p>The method creates one unresolved, non-differentiable INT64 descriptor and one Operation
     * retaining the exact kind and attributes, then makes one central factory call with no label
     * and exact ordered input {@code [input]}. The factory creates a fresh producer and
     * index-zero provenance. This method performs no further validation and accesses no value or
     * storage.</p>
     *
     * @param input validated tensor retained as the sole producer input
     * @param kind validated exact arg-extrema kind
     * @param shape non-null derived result Shape
     * @param attrs non-null exact normalized attributes
     * @return the non-null fresh unlabeled, storage-free Tensor returned by the central factory,
     *     with the exact descriptor, operation, ordered input, and output index zero
     * @throws IllegalStateException if tensor identifier space is exhausted after descriptor,
     *     operation, producer, and provenance construction
     */
    private static Tensor create(
            Tensor input,
            AggregateReductionKind kind,
            Shape shape,
            ArgExtremaAttrs attrs) {
        TensorDescriptor descriptor =
                new TensorDescriptor(DataType.INT64, shape, Optional.empty(), false);
        Operation operation = new Operation(kind, attrs);
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(input));
    }
}
