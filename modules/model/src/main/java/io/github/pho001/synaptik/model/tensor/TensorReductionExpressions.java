package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated storage-free sum, mean, product, minimum, and maximum tensor
 * expressions.
 *
 * <p>This package-private boundary owns deterministic validation, single-axis normalization,
 * structural result-shape derivation, typed operation construction, and exact one-input
 * provenance for the supported numeric aggregate kinds. Full forms produce the canonical scalar
 * shape and use {@link NoOperationAttrs#INSTANCE}. Axis forms either remove the selected dimension
 * or replace it with extent one and carry {@link AxisReductionAttrs}. Every unaffected immutable
 * {@link Dimension} reference is retained.</p>
 *
 * <p>Construction is eager only for expression metadata. The helper accepts dynamic and zero
 * extents without reading an element count, and it does not inspect values or storage, aggregate
 * data, select accumulation or empty-domain behavior, create gradient rules, capture a graph, or
 * provide compiler, runtime, or backend behavior. Product, minimum, and maximum preserve an
 * existing gradient request as eligibility metadata without claiming that a gradient rule
 * exists. Aggregate minimum and maximum remain distinct from equally named elementwise binary
 * kinds; this helper defines no NaN, signed-zero, tie, or empty-domain policy.</p>
 */
final class TensorReductionExpressions {
    /** Prevents instantiation because numeric reduction construction is stateless. */
    private TensorReductionExpressions() {
    }

    /**
     * Creates one fresh expression that reduces every axis of a floating tensor.
     *
     * <p>Validation occurs in this exact order: null-check {@code input}, null-check {@code kind},
     * accept only {@code SUM}, {@code MEAN}, {@code PROD}, {@code MIN}, or {@code MAX}, then
     * require a floating input. The method then delegates to common construction with {@link
     * NoOperationAttrs#INSTANCE} and the canonical scalar shape. Failures before common factory
     * delegation consume no Tensor identity.</p>
     *
     * @param input non-null floating tensor retained by exact reference as the sole provenance
     *     input; no values, storage, element count, or shape extents are inspected
     * @param kind non-null aggregate kind; must be {@link AggregateReductionKind#SUM},
     *     {@link AggregateReductionKind#MEAN}, {@link AggregateReductionKind#PROD}, {@link
     *     AggregateReductionKind#MIN}, or {@link AggregateReductionKind#MAX}
     * @return the non-null exact fresh scalar Tensor returned by the central factory, with
     *     unchanged type and gradient eligibility and no label, resolved layout, or storage
     * @throws NullPointerException if {@code input} or {@code kind} is null, checked in that order
     *     with the parameter name as the message
     * @throws IllegalArgumentException if {@code kind} is unsupported by this helper or the input
     *     data type is not floating, checked in that order with the documented exact message
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     model values have been constructed
     */
    static Tensor applyFull(Tensor input, AggregateReductionKind kind) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        validateKind(kind);
        validateFloatingInput(input);
        return create(input, kind, NoOperationAttrs.INSTANCE, Shape.scalar());
    }

    /**
     * Creates one fresh expression that reduces a normalized single input axis.
     *
     * <p>Validation null-checks {@code input} and {@code kind}, validates the supported kind, and
     * validates floating input before axis handling. The input Shape then normalizes the caller
     * axis exactly once. Result-shape derivation either removes that normalized position or
     * replaces it with a new {@link StaticDimension} of extent one, retaining every unaffected
     * dimension reference. One {@link AxisReductionAttrs} carries the normalized axis and exact
     * retention flag into common construction.</p>
     *
     * @param input non-null floating tensor retained by exact reference as the sole provenance
     *     input; its metadata, storage association, and values remain unchanged
     * @param kind non-null aggregate kind; must be {@link AggregateReductionKind#SUM},
     *     {@link AggregateReductionKind#MEAN}, {@link AggregateReductionKind#PROD}, {@link
     *     AggregateReductionKind#MIN}, or {@link AggregateReductionKind#MAX}
     * @param axis positive or negative input axis accepted by {@link Shape#normalizeAxis(int)}
     * @param keepDimensions {@code true} to replace the selected result dimension with extent one,
     *     or {@code false} to remove it
     * @return the non-null exact fresh derived Tensor returned by the central factory, with the
     *     locally derived shape, unchanged type and gradient eligibility, and no label, resolved
     *     layout, or storage
     * @throws NullPointerException if {@code input} or {@code kind} is null, checked in that order
     *     with the parameter name as the message
     * @throws IllegalArgumentException if {@code kind} is unsupported by this helper or the input
     *     data type is not floating, checked in that order before axis validation
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input Shape, including
     *     every axis when the input is scalar
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     model values have been constructed
     */
    static Tensor applyAxis(
            Tensor input,
            AggregateReductionKind kind,
            int axis,
            boolean keepDimensions) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        validateKind(kind);
        validateFloatingInput(input);

        Shape inputShape = input.descriptor().shape();
        int normalizedAxis = inputShape.normalizeAxis(axis);
        Shape shape = reduceShape(inputShape, normalizedAxis, keepDimensions);
        AxisReductionAttrs attrs = new AxisReductionAttrs(normalizedAxis, keepDimensions);
        return create(input, kind, attrs, shape);
    }

    /**
     * Restricts this helper to its five floating numeric aggregate kinds.
     *
     * @param kind non-null reduction kind already checked by the package-private entry
     * @throws IllegalArgumentException if {@code kind} is not {@code SUM}, {@code MEAN},
     *     {@code PROD}, {@code MIN}, or {@code MAX}, with the exact rejected enum value in the
     *     message
     */
    private static void validateKind(AggregateReductionKind kind) {
        if (kind != AggregateReductionKind.SUM
                && kind != AggregateReductionKind.MEAN
                && kind != AggregateReductionKind.PROD
                && kind != AggregateReductionKind.MIN
                && kind != AggregateReductionKind.MAX) {
            throw new IllegalArgumentException(
                    "kind must be SUM, MEAN, PROD, MIN, or MAX, but was " + kind);
        }
    }

    /**
     * Enforces the floating-only input boundary without conversion or promotion.
     *
     * @param input non-null Tensor whose descriptor is read only for its immutable data type
     * @throws IllegalArgumentException if the input data type is not floating, with the exact
     *     rejected data type in the message
     */
    private static void validateFloatingInput(Tensor input) {
        DataType dataType = input.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "input must be a floating data type, but was " + dataType);
        }
    }

    /**
     * Derives a single-axis result Shape while retaining unaffected Dimension references.
     *
     * <p>Retention allocates one same-rank array, copies every input reference, and replaces only
     * the selected position with a new extent-one dimension. Removal allocates one rank-minus-one
     * array and copies every nonselected reference in original order. Shape construction occurs
     * exactly once; an empty result array produces the canonical scalar Shape.</p>
     *
     * @param inputShape non-null immutable input Shape whose axis was already normalized
     * @param normalizedAxis non-negative valid position in {@code inputShape}
     * @param keepDimensions whether to retain the selected result position with extent one
     * @return a non-null immutable locally derived Shape; unaffected dimensions are the exact
     *     input references
     */
    private static Shape reduceShape(
            Shape inputShape, int normalizedAxis, boolean keepDimensions) {
        List<Dimension> inputDimensions = inputShape.dimensions();
        if (keepDimensions) {
            Dimension[] outputDimensions = new Dimension[inputDimensions.size()];
            for (int index = 0; index < inputDimensions.size(); index++) {
                outputDimensions[index] = inputDimensions.get(index);
            }
            outputDimensions[normalizedAxis] = new StaticDimension(1);
            return Shape.ofDimensions(outputDimensions);
        }

        Dimension[] outputDimensions = new Dimension[inputDimensions.size() - 1];
        int outputIndex = 0;
        for (int inputIndex = 0; inputIndex < inputDimensions.size(); inputIndex++) {
            if (inputIndex != normalizedAxis) {
                outputDimensions[outputIndex++] = inputDimensions.get(inputIndex);
            }
        }
        return Shape.ofDimensions(outputDimensions);
    }

    /**
     * Constructs exact descriptor, operation, provenance, and derived Tensor metadata.
     *
     * <p>The method creates, in order, one unresolved descriptor with input type and gradient
     * eligibility, one Operation retaining the exact kind and attributes reference, one
     * provenance value retaining exact ordered input {@code [input]}, and one central
     * {@link TensorFactory#createDerived(TensorDescriptor, Optional, TensorProvenance)} call with
     * no label. It performs no additional semantic validation and does not access storage or
     * values.</p>
     *
     * @param input validated floating tensor retained as the sole provenance input
     * @param kind validated numeric aggregate kind retained exactly in the Operation
     * @param attrs non-null exact immutable full- or axis-reduction attributes reference
     * @param shape non-null canonical or locally derived result Shape
     * @return the non-null exact fresh, unlabeled, storage-free Tensor returned by the factory
     * @throws IllegalStateException if tensor identifier space is exhausted after descriptor,
     *     operation, and provenance construction
     */
    private static Tensor create(
            Tensor input,
            AggregateReductionKind kind,
            OperationAttrs attrs,
            Shape shape) {
        TensorDescriptor descriptor = new TensorDescriptor(
                input.descriptor().dataType(),
                shape,
                Optional.empty(),
                input.descriptor().requiresGrad());
        Operation operation = new Operation(kind, attrs);
        TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
        return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
    }
}
