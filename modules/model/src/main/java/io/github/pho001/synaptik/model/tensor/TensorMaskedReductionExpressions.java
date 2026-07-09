package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free masked sum and mean tensor expressions.
 *
 * <p>The BOOL mask uses the model's ordinary right-aligned broadcasting contract, and that
 * broadcast must produce exactly the input Shape. Callers use visible rank-editing expressions
 * such as {@link Tensor#expandDims(int)} when the intended axes are not already expressed by
 * right alignment. This helper neither guesses axis placement nor inserts a hidden transform.</p>
 *
 * <p>A false mask position excludes the corresponding input value before aggregation, including
 * NaN and infinity. A slice with no selected values produces floating zero for sum and NaN for
 * mean. The selected input axis is removed from the result while every unaffected
 * {@link Dimension} reference is retained.</p>
 *
 * <p>This helper constructs one operation and its metadata only. It does not inspect values or
 * storage, materialize a mask, aggregate or divide values, create gradient rules, capture a graph,
 * or provide compiler, runtime, or backend behavior.</p>
 */
final class TensorMaskedReductionExpressions {
    /** Prevents instantiation because masked-reduction construction is stateless. */
    private TensorMaskedReductionExpressions() {
    }

    /**
     * Creates one fresh masked, axis-removing sum or mean expression.
     *
     * <p>Validation checks {@code input}, {@code mask}, and {@code kind} for null in that order,
     * followed by exact kind, floating input, and BOOL mask checks. The input Shape then normalizes
     * {@code axis} exactly once. One ordinary broadcast is computed and must equal the input Shape;
     * an incompatible aligned pair retains {@link ShapeBroadcast}'s diagnostic, while a compatible
     * broadcast that enlarges the input domain receives the directional masked-reduction
     * diagnostic. Every validation failure occurs before expression metadata or Tensor identity
     * construction.</p>
     *
     * <p>The result retains the input data type and gradient eligibility, has unresolved layout
     * and no label or storage, and records one operation with exact ordered provenance
     * {@code [input, mask]}. Neither input is mutated. Mask eligibility does not contribute to the
     * result, and preserving input eligibility does not promise a gradient rule.</p>
     *
     * @param input non-null floating tensor whose exact type and gradient eligibility are retained
     * @param mask non-null exact BOOL tensor whose Shape must broadcast exactly to the input Shape;
     *     the caller retains ownership and the exact reference is stored only in provenance
     * @param kind non-null reduction kind; must be {@link AggregateReductionKind#SUM} or
     *     {@link AggregateReductionKind#MEAN}
     * @param axis input axis in {@code [-rank, rank - 1]}; negative values count from the end
     * @return the non-null exact fresh Tensor returned by the central factory, with the selected
     *     axis removed, unresolved layout, no label or storage, and two-input provenance
     * @throws NullPointerException if {@code input}, {@code mask}, or {@code kind} is null, checked
     *     in that order with the parameter name as the message
     * @throws IllegalArgumentException if kind or data-type validation fails, if Shape broadcasting
     *     is incompatible, or if the broadcast result does not equal the input Shape
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
        Shape broadcastShape = ShapeBroadcast.broadcast(inputShape, maskShape);
        if (!broadcastShape.equals(inputShape)) {
            throw new IllegalArgumentException(
                    "mask shape "
                            + maskShape
                            + " must broadcast exactly to input shape "
                            + inputShape
                            + ", but produced "
                            + broadcastShape);
        }

        Shape resultShape = reduceShape(inputShape, normalizedAxis);
        return create(input, mask, kind, normalizedAxis, resultShape);
    }

    /**
     * Removes one normalized input axis while retaining every unaffected Dimension reference.
     *
     * @param inputShape non-null immutable input Shape
     * @param normalizedAxis valid non-negative input-axis position to remove
     * @return a non-null immutable result Shape; rank-one removal returns canonical scalar Shape
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
     * Constructs the exact masked-reduction descriptor, operation, producer, and Tensor.
     *
     * <p>Construction creates, in order, one {@link MaskedReductionAttrs}, one unresolved
     * descriptor retaining input type and gradient eligibility, and one Operation retaining the
     * exact kind and attributes. One central factory call then creates a one-output producer with
     * ordered inputs {@code [input, mask]} and output-index-zero provenance. Identifier exhaustion
     * can occur only at that final boundary.</p>
     *
     * @param input validated floating tensor supplying result type and gradient eligibility
     * @param mask validated BOOL tensor retained as provenance input one
     * @param kind validated SUM or MEAN kind retained exactly
     * @param normalizedAxis valid normalized input axis retained in attributes
     * @param resultShape non-null locally derived axis-removing result Shape
     * @return the non-null exact fresh, unlabeled, unresolved, storage-free Tensor from the factory
     * @throws IllegalStateException if tensor identifier space is exhausted after descriptor and
     *     operation construction
     */
    private static Tensor create(
            Tensor input,
            Tensor mask,
            AggregateReductionKind kind,
            int normalizedAxis,
            Shape resultShape) {
        MaskedReductionAttrs attrs = new MaskedReductionAttrs(normalizedAxis);
        TensorDescriptor descriptor = new TensorDescriptor(
                input.descriptor().dataType(),
                resultShape,
                Optional.empty(),
                input.descriptor().requiresGrad());
        Operation operation = new Operation(kind, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(input, mask));
    }
}
