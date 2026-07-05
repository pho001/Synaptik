package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free masked sum and mean tensor expressions.
 *
 * <p>A mask dimension aligns to one distinct input axis, in mask-dimension order. Equal
 * immutable dimensions are compatible, and a static mask dimension of extent one may align to
 * any input dimension. Omitted input axes are implicit broadcast dimensions. For example, a mask
 * shaped {@code [batch, time]} aligns through mapping {@code [0, 1]} to input
 * {@code [batch, time, features]}: mask axis zero maps to input axis zero, mask axis one maps to
 * input axis one, and input axis two is omitted and therefore broadcast.</p>
 *
 * <p>When several mappings are valid, one containing the reduction axis is preferred. Remaining
 * ambiguity is resolved by minimum total positional displacement and then lexicographically.
 * Resolution uses bounded dynamic-programming states over mask and input positions rather than a
 * retained candidate collection. The selected input axis is removed from the result while every
 * unaffected {@link Dimension} reference is retained.</p>
 *
 * <p>This helper constructs metadata only. It does not inspect values or storage, materialize or
 * reshape a mask, aggregate or divide values, create gradient rules, capture a graph, or provide
 * compiler, runtime, or backend behavior.</p>
 */
final class TensorMaskedReductionExpressions {
    /** Prevents instantiation because masked-reduction construction is stateless. */
    private TensorMaskedReductionExpressions() {
    }

    /**
     * Creates one fresh masked, axis-removing sum or mean expression.
     *
     * <p>Validation checks {@code input}, {@code mask}, and {@code kind} for null in that order,
     * then validates the kind, floating input type, and exact BOOL mask type. The input Shape
     * normalizes {@code axis} exactly once before mask rank and alignment are checked. Every
     * failure before the final factory call consumes no Tensor identity.</p>
     *
     * <p>The result retains the input data type and gradient eligibility, has an unresolved
     * layout and no label or host storage, and records one operation with exact ordered
     * provenance {@code [input, mask]}. Neither input is mutated or retained in attributes.</p>
     *
     * @param input non-null floating tensor whose exact type and gradient eligibility are retained
     * @param mask non-null exact BOOL tensor aligned structurally without reading its values or
     *     storage
     * @param kind non-null reduction kind; must be {@link AggregateReductionKind#SUM} or
     *     {@link AggregateReductionKind#MEAN}
     * @param axis input axis in the inclusive range {@code [-rank, rank - 1]}; negative values
     *     count from the final axis
     * @return the non-null exact fresh Tensor returned by the central factory, with the selected
     *     axis removed, unresolved layout, no label or storage, and ordered two-input provenance
     * @throws NullPointerException if {@code input}, {@code mask}, or {@code kind} is null, checked
     *     in that order with the parameter name as the message
     * @throws IllegalArgumentException if the kind, input type, mask type, mask rank, or local
     *     Shape alignment is invalid, checked in the documented order with the documented message
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input Shape, including
     *     every axis for a scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after all local
     *     immutable expression metadata has been constructed
     */
    static Tensor apply(
            Tensor input,
            Tensor mask,
            AggregateReductionKind kind,
            int axis) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(mask, "mask");
        Objects.requireNonNull(kind, "kind");
        if (kind != AggregateReductionKind.SUM && kind != AggregateReductionKind.MEAN) {
            throw new IllegalArgumentException("kind must be SUM or MEAN, but was " + kind);
        }

        DataType inputDataType = input.descriptor().dataType();
        if (!inputDataType.isFloating()) {
            throw new IllegalArgumentException(
                    "input must be a floating data type, but was " + inputDataType);
        }

        DataType maskDataType = mask.descriptor().dataType();
        if (maskDataType != DataType.BOOL) {
            throw new IllegalArgumentException(
                    "mask must have BOOL data type, but was " + maskDataType);
        }

        Shape inputShape = input.descriptor().shape();
        Shape maskShape = mask.descriptor().shape();
        int normalizedAxis = inputShape.normalizeAxis(axis);
        if (maskShape.rank() > inputShape.rank()) {
            throw new IllegalArgumentException(
                    "mask rank must not exceed input rank: mask="
                            + maskShape.rank()
                            + ", input="
                            + inputShape.rank());
        }

        List<Integer> mapping = resolveMapping(inputShape, maskShape, normalizedAxis);
        if (mapping == null) {
            throw new IllegalArgumentException(
                    "mask shape "
                            + maskShape
                            + " cannot be aligned to input shape "
                            + inputShape
                            + " for reduction axis "
                            + normalizedAxis);
        }

        Shape resultShape = reduceShape(inputShape, normalizedAxis);
        return create(input, mask, kind, normalizedAxis, mapping, resultShape);
    }

    /**
     * Resolves the preferred ordered injective mask-to-input axis mapping.
     *
     * <p>The dynamic-programming table is indexed by the next mask and input positions and by
     * whether the remaining suffix must cover the reduction axis. Each state chooses between
     * skipping an input axis and aligning the current mask dimension. Thus at most
     * {@code (maskRank + 1) * (inputRank + 1) * 2} states are retained, never all complete
     * candidates. Reconstruction considers alignment before skipping when both retain the same
     * optimum, which selects the lexicographically smallest mapping.</p>
     *
     * @param inputShape non-null immutable input Shape
     * @param maskShape non-null immutable mask Shape whose rank does not exceed the input rank
     * @param normalizedAxis valid non-negative input reduction axis
     * @return an immutable preferred mapping, including an empty list for a scalar mask, or
     *     {@code null} when no compatible ordered injection exists
     */
    private static List<Integer> resolveMapping(
            Shape inputShape, Shape maskShape, int normalizedAxis) {
        List<Dimension> inputDimensions = inputShape.dimensions();
        List<Dimension> maskDimensions = maskShape.dimensions();
        int inputRank = inputDimensions.size();
        int maskRank = maskDimensions.size();
        long unreachable = Long.MAX_VALUE;
        long[][][] minimumDisplacement =
                new long[maskRank + 1][inputRank + 1][2];

        for (int inputIndex = 0; inputIndex <= inputRank; inputIndex++) {
            minimumDisplacement[maskRank][inputIndex][0] = 0;
            minimumDisplacement[maskRank][inputIndex][1] = unreachable;
        }
        for (int maskIndex = maskRank - 1; maskIndex >= 0; maskIndex--) {
            minimumDisplacement[maskIndex][inputRank][0] = unreachable;
            minimumDisplacement[maskIndex][inputRank][1] = unreachable;
            for (int inputIndex = inputRank - 1; inputIndex >= 0; inputIndex--) {
                for (int mustCover = 0; mustCover <= 1; mustCover++) {
                    long best = minimumDisplacement[maskIndex][inputIndex + 1][mustCover];
                    if (compatible(maskDimensions.get(maskIndex), inputDimensions.get(inputIndex))) {
                        int remainingMustCover =
                                mustCover == 1 && inputIndex != normalizedAxis ? 1 : 0;
                        long suffix = minimumDisplacement[maskIndex + 1][inputIndex + 1]
                                [remainingMustCover];
                        if (suffix != unreachable) {
                            long aligned = Math.abs((long) inputIndex - maskIndex) + suffix;
                            if (aligned < best) {
                                best = aligned;
                            }
                        }
                    }
                    minimumDisplacement[maskIndex][inputIndex][mustCover] = best;
                }
            }
        }

        int mustCover = minimumDisplacement[0][0][1] != unreachable ? 1 : 0;
        if (minimumDisplacement[0][0][mustCover] == unreachable) {
            return null;
        }

        List<Integer> mapping = new ArrayList<>(maskRank);
        int maskIndex = 0;
        int inputIndex = 0;
        while (maskIndex < maskRank) {
            long optimum = minimumDisplacement[maskIndex][inputIndex][mustCover];
            if (compatible(maskDimensions.get(maskIndex), inputDimensions.get(inputIndex))) {
                int remainingMustCover =
                        mustCover == 1 && inputIndex != normalizedAxis ? 1 : 0;
                long suffix = minimumDisplacement[maskIndex + 1][inputIndex + 1]
                        [remainingMustCover];
                if (suffix != unreachable
                        && Math.abs((long) inputIndex - maskIndex) + suffix == optimum) {
                    mapping.add(inputIndex);
                    maskIndex++;
                    inputIndex++;
                    mustCover = remainingMustCover;
                    continue;
                }
            }
            inputIndex++;
        }
        return List.copyOf(mapping);
    }

    /**
     * Tests one locally provable mask-dimension alignment.
     *
     * <p>Immutable value equality proves matching static extents or canonical dynamic symbols. A
     * static mask singleton is additionally compatible with every static, zero, or dynamic input
     * dimension. No other dynamic/static relationship or symbolic constraint is inferred.</p>
     *
     * @param maskDimension non-null immutable mask dimension
     * @param inputDimension non-null immutable input dimension
     * @return {@code true} exactly when the dimensions are equal or the mask dimension is a
     *     static singleton
     */
    private static boolean compatible(
            Dimension maskDimension, Dimension inputDimension) {
        return maskDimension.equals(inputDimension)
                || maskDimension instanceof StaticDimension staticDimension
                        && staticDimension.size() == 1;
    }

    /**
     * Removes one normalized input axis while retaining every unaffected Dimension reference.
     *
     * @param inputShape non-null immutable input Shape
     * @param normalizedAxis valid non-negative input-axis position to remove
     * @return a non-null immutable result Shape; rank-one removal returns the canonical scalar
     *     Shape through the single {@link Shape#ofDimensions(Dimension...)} call
     */
    private static Shape reduceShape(Shape inputShape, int normalizedAxis) {
        List<Dimension> inputDimensions = inputShape.dimensions();
        Dimension[] resultDimensions = new Dimension[inputDimensions.size() - 1];
        int resultIndex = 0;
        for (int inputIndex = 0; inputIndex < inputDimensions.size(); inputIndex++) {
            if (inputIndex != normalizedAxis) {
                resultDimensions[resultIndex++] = inputDimensions.get(inputIndex);
            }
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Constructs the exact masked-reduction descriptor, operation, provenance, and Tensor.
     *
     * <p>Construction creates, in order, one {@link MaskedReductionAttrs}, one unresolved
     * descriptor retaining input type and gradient eligibility, one Operation retaining the exact
     * kind and attributes reference, one provenance value retaining exact ordered references
     * {@code [input, mask]}, and one central factory call with no label. No storage, values, prior
     * provenance, or graph state are inspected.</p>
     *
     * @param input validated floating tensor supplying result type and gradient eligibility
     * @param mask validated BOOL tensor retained as provenance input one
     * @param kind validated SUM or MEAN kind retained exactly
     * @param normalizedAxis valid normalized input axis retained in attributes
     * @param mapping non-null immutable preferred mask-dimension-to-input-axis mapping
     * @param resultShape non-null locally derived axis-removing result Shape
     * @return the non-null exact fresh, unlabeled, storage-free Tensor returned by the factory
     * @throws IllegalStateException if tensor identifier space is exhausted after descriptor,
     *     operation, and provenance construction
     */
    private static Tensor create(
            Tensor input,
            Tensor mask,
            AggregateReductionKind kind,
            int normalizedAxis,
            List<Integer> mapping,
            Shape resultShape) {
        MaskedReductionAttrs attrs = new MaskedReductionAttrs(normalizedAxis, mapping);
        TensorDescriptor descriptor = new TensorDescriptor(
                input.descriptor().dataType(),
                resultShape,
                Optional.empty(),
                input.descriptor().requiresGrad());
        Operation operation = new Operation(kind, attrs);
        TensorProvenance provenance = new TensorProvenance(operation, List.of(input, mask));
        return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
    }
}
