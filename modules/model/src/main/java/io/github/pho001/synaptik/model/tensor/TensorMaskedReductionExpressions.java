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
        return applyChecked(input, mask, kind, axis, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null floating tensor whose exact type and gradient eligibility are retained
     * @param mask non-null exact BOOL tensor whose Shape must broadcast exactly to the input Shape;
     *     the caller retains ownership and the exact reference is stored only in provenance
     * @param kind non-null reduction kind; must be {@link AggregateReductionKind#SUM} or
     *     {@link AggregateReductionKind#MEAN}
     * @param axis input axis in {@code [-rank, rank - 1]}; negative values count from the end
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input}, {@code mask}, or {@code kind} is null, checked
     *     in that order with the parameter name as the message
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if kind or data-type validation fails, if Shape broadcasting
     *     is incompatible, or if the broadcast result does not equal the input Shape
     * @throws IndexOutOfBoundsException if {@code axis} is invalid for the input Shape, including
     *     every axis for a scalar input
     * @throws IllegalStateException if tensor identifier space is exhausted after all local
     *     immutable expression metadata has been constructed
     */
    static Tensor applyChecked(
            Tensor input,
            Tensor mask,
            AggregateReductionKind kind,
            int axis,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(mask, "mask");
        Objects.requireNonNull(kind, "kind");
        return applyDescriptors(
                input.descriptor(), mask.descriptor(), kind, axis, descriptorCheck, input, mask);
    }

    /**
     * Runs the shared descriptor guards and completes metadata before real provenance is used.
     * @param input immutable operand metadata
     * @param mask immutable operand metadata
     * @param kind ordinary construction argument, validated in existing order
     * @param axis ordinary construction argument, validated in existing order
     * @param descriptorCheck optional synchronous, unretained metadata completion; null constructs a Tensor
     * @param inputTensor real provenance, unused when metadata completion is supplied
     * @param maskTensor real provenance, unused when metadata completion is supplied
     * @return null after metadata completion, otherwise the ordinary result
     */
    private static Tensor applyDescriptors(
            TensorDescriptor input,
            TensorDescriptor mask,
            AggregateReductionKind kind,
            int axis,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor inputTensor,
            Tensor maskTensor) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(mask, "mask");
        Objects.requireNonNull(kind, "kind");
        if (kind != AggregateReductionKind.SUM && kind != AggregateReductionKind.MEAN) {
            throw new IllegalArgumentException("kind must be SUM or MEAN, but was " + kind);
        }

        DataType inputDataType = input.dataType();
        if (!inputDataType.isFloating()) {
            throw new IllegalArgumentException(
                    "input must be a floating data type, but was " + inputDataType);
        }

        DataType maskDataType = mask.dataType();
        if (maskDataType != DataType.BOOL) {
            throw new IllegalArgumentException(
                    "mask must have BOOL data type, but was " + maskDataType);
        }

        Shape inputShape = input.shape();
        Shape maskShape = mask.shape();
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
        return createDescriptors(input, mask, kind, normalizedAxis, resultShape, descriptorCheck, inputTensor, maskTensor);
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
     * Runs the shared descriptor guards and completes metadata before real provenance is used.
     * @param input immutable operand metadata
     * @param mask immutable operand metadata
     * @param kind ordinary construction argument, validated in existing order
     * @param normalizedAxis ordinary construction argument, validated in existing order
     * @param resultShape ordinary construction argument, validated in existing order
     * @param descriptorCheck optional synchronous, unretained metadata completion; null constructs a Tensor
     * @param inputTensor real provenance, unused when metadata completion is supplied
     * @param maskTensor real provenance, unused when metadata completion is supplied
     * @return null after metadata completion, otherwise the ordinary result
     */
    private static Tensor createDescriptors(
            TensorDescriptor input,
            TensorDescriptor mask,
            AggregateReductionKind kind,
            int normalizedAxis,
            Shape resultShape,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor inputTensor,
            Tensor maskTensor) {
        MaskedReductionAttrs attrs = new MaskedReductionAttrs(normalizedAxis);
        TensorDescriptor descriptor = new TensorDescriptor(
                input.dataType(),
                resultShape,
                Optional.empty(),
                input.requiresGrad());
        Operation operation = new Operation(kind, attrs);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(inputTensor, maskTensor));
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
        MaskedReductionAttrs attrs = (MaskedReductionAttrs) operation.attrs();
        applyDescriptors(inputs.get(0), inputs.get(1), (AggregateReductionKind) operation.kind(),
                attrs.axis(), sink, null, null);
    }
}
