package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgMaxAttrs;
import io.github.pho001.synaptik.model.operation.reduction.ArgMaxTiePolicy;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free, single-axis arg-max expressions.
 *
 * <p>This package-private boundary owns deterministic null and numeric-input validation, one
 * positive-or-negative axis normalization, structural result-shape derivation, fixed INT64
 * descriptor construction, exact {@link ArgMaxAttrs}, and one-input provenance. The caller's
 * policy is always explicit at this boundary; public convenience overloads supply
 * {@link ArgMaxTiePolicy#FIRST_INDEX} before delegation.</p>
 *
 * <p>Construction is eager only for immutable expression metadata. Floating and integral inputs
 * are accepted, BOOL is rejected, and static zero extents and dynamic dimensions remain valid
 * structural inputs. The helper reads no values or storage, compares no maxima, selects no index,
 * and defines no NaN, signed-zero, infinity, equality, empty-axis, gradient, graph, compiler,
 * runtime, backend, or execution behavior. It does not modify the ordinary aggregate-reduction
 * helper because arg-max has no full form and has distinct attributes and result facts.</p>
 */
final class TensorArgMaxExpressions {
    /** Prevents instantiation because single-axis arg-max construction owns no state. */
    private TensorArgMaxExpressions() {
    }

    /**
     * Validates and creates one fresh fixed-INT64 arg-max expression.
     *
     * <p>Validation checks {@code input}, then {@code tiePolicy}, then numeric input eligibility.
     * It reads the exact immutable input Shape, normalizes {@code axis} exactly once, derives a
     * removal or retained-axis Shape, constructs one {@link ArgMaxAttrs}, and delegates to common
     * construction. Every failure before common construction consumes no Tensor identity.</p>
     *
     * @param input non-null floating or integral tensor retained as the sole provenance input
     * @param axis positive or negative input axis accepted by {@link Shape#normalizeAxis(int)}
     * @param keepDimensions {@code true} to retain the selected result axis with extent one, or
     *     {@code false} to remove it
     * @param tiePolicy non-null explicit policy retained by exact enum reference in attributes
     * @return the non-null fresh storage-free INT64 result with false gradient eligibility,
     *     unresolved layout, no label, and exact one-input provenance
     * @throws NullPointerException if {@code input} or {@code tiePolicy} is null, checked in that
     *     order with the parameter name as the message
     * @throws IllegalArgumentException if the input is not floating or integral, with message
     *     {@code input must have a numeric data type, but was <dataType>}
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input Shape, including
     *     every axis when the input is scalar
     * @throws IllegalStateException if tensor identifier space is exhausted after the local Shape
     *     and attributes have been constructed
     */
    static Tensor apply(
            Tensor input, int axis, boolean keepDimensions, ArgMaxTiePolicy tiePolicy) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(tiePolicy, "tiePolicy");
        validateNumericInput(input);
        Shape inputShape = input.descriptor().shape();
        int normalizedAxis = inputShape.normalizeAxis(axis);
        Shape shape = reduceShape(inputShape, normalizedAxis, keepDimensions);
        ArgMaxAttrs attrs = new ArgMaxAttrs(normalizedAxis, keepDimensions, tiePolicy);
        return create(input, shape, attrs);
    }

    /**
     * Validates floating or integral input without conversion, promotion, or storage access.
     *
     * @param input non-null tensor whose immutable descriptor supplies the input type
     * @throws IllegalArgumentException if the input type is neither floating nor integral, with
     *     message {@code input must have a numeric data type, but was <dataType>}
     */
    private static void validateNumericInput(Tensor input) {
        DataType type = input.descriptor().dataType();
        if (!type.isFloating() && !type.isIntegral()) {
            throw new IllegalArgumentException(
                    "input must have a numeric data type, but was " + type);
        }
    }

    /**
     * Derives a removal or retained-axis Shape while preserving unaffected references.
     *
     * <p>Retention allocates one same-rank array, copies every input Dimension reference, and
     * replaces only the normalized position with a new {@link StaticDimension} of extent one.
     * Removal allocates one rank-minus-one array and copies every nonselected reference in order.
     * One {@link Shape#ofDimensions(Dimension...)} call completes either branch; an empty result
     * array yields the canonical scalar Shape.</p>
     *
     * @param inputShape non-null immutable shape whose axis is already normalized
     * @param normalizedAxis valid non-negative selected position in {@code inputShape}
     * @param keepDimensions {@code true} to replace the selected result position with extent one,
     *     or {@code false} to remove it
     * @return a non-null immutable locally derived Shape retaining every unaffected exact input
     *     Dimension reference
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
     * Constructs the fixed result descriptor, exact operation/provenance, and one identity.
     *
     * <p>The method creates, in order, one unresolved non-differentiable INT64 descriptor, one
     * {@link AggregateReductionKind#ARG_MAX} Operation retaining the exact attributes reference,
     * and one {@link TensorFactory#createDerived(TensorDescriptor, Optional, Operation, List)}
     * call with ordered producer input {@code [input]} and no label. The factory creates the
     * producer and index-zero provenance. This method performs no further validation and accesses
     * no values or storage.</p>
     *
     * @param input validated tensor retained as the exact sole provenance input
     * @param shape non-null canonical or locally derived result shape
     * @param attrs non-null exact normalized attributes retained by the operation
     * @return the non-null fresh INT64, non-differentiable, unresolved-layout, unlabeled,
     *     storage-free Tensor returned by the central factory
     * @throws IllegalStateException if tensor identifier space is exhausted after descriptor,
     *     operation, and provenance construction
     */
    private static Tensor create(Tensor input, Shape shape, ArgMaxAttrs attrs) {
        TensorDescriptor descriptor =
                new TensorDescriptor(DataType.INT64, shape, Optional.empty(), false);
        Operation operation = new Operation(AggregateReductionKind.ARG_MAX, attrs);
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(input));
    }
}
