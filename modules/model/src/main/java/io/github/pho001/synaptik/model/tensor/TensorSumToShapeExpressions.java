package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated binding-aware sum-to-Shape Tensor expressions.
 *
 * <p>This package-private boundary owns the exact null, numeric-type, rank, and statically
 * provable right-aligned compatibility checks. It retains unresolved aligned pairs so later
 * binding validation can prove that each target extent is one or equals the source extent. It then
 * constructs one explicit {@link AggregateReductionKind#SUM} occurrence with exact target Shape,
 * unchanged input type and gradient eligibility, unresolved layout, and one-input index-zero
 * provenance.</p>
 *
 * <p>The helper reads no values, storage, or element count. It does not resolve axes, bind
 * dimensions, aggregate data, define gradient construction, capture a graph, or provide compiler,
 * runtime, backend, or execution behavior.</p>
 */
final class TensorSumToShapeExpressions {
    /** Prevents instantiation because expression construction is stateless. */
    private TensorSumToShapeExpressions() {
    }

    /**
     * Creates one fresh explicit SUM expression with the requested exact result Shape.
     *
     * <p>Validation null-checks {@code input} and {@code targetShape} in that order, rejects BOOL,
     * reads the exact input Shape, rejects a larger target rank, and then scans aligned target axes
     * in increasing order. A static pair is accepted exactly when the extents are equal or the
     * target extent is one; any pair involving an unresolved Dimension is deferred. Every local
     * failure precedes factory delegation and consumes no Tensor identifier.</p>
     *
     * @param input non-null numeric Tensor retained by exact reference as the sole provenance
     *     input; its metadata, storage association, values, and identity remain unchanged
     * @param targetShape non-null exact result Shape to retain in both attributes and descriptor;
     *     its rank must not exceed the input rank
     * @return the non-null fresh, unlabeled, storage-free Tensor with exact input type and gradient
     *     eligibility, exact target Shape, unresolved layout, and output-index-zero provenance
     * @throws NullPointerException if {@code input} or {@code targetShape} is null, checked in
     *     that order with the parameter name as the message
     * @throws IllegalArgumentException if the input is BOOL, the target rank exceeds the input
     *     rank, or the first fully static aligned pair is neither equal nor target-one
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    static Tensor apply(Tensor input, Shape targetShape) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(targetShape, "targetShape");
        validateInput(input);
        Shape inputShape = input.descriptor().shape();
        validateCompatibility(inputShape, targetShape);
        return create(input, targetShape);
    }

    /**
     * Requires one of the five current numeric input types without conversion or promotion.
     *
     * @param input non-null Tensor whose descriptor is read only for its immutable data type
     * @throws IllegalArgumentException if the input type is BOOL, with the exact SUM diagnostic
     */
    private static void validateInput(Tensor input) {
        DataType dataType = input.descriptor().dataType();
        if (!dataType.isFloating() && !dataType.isIntegral()) {
            throw new IllegalArgumentException(
                    "input must have a numeric data type for SUM, but was " + dataType);
        }
    }

    /**
     * Rejects only rank and aligned-dimension incompatibility provable from current Shapes.
     *
     * @param inputShape non-null exact source Shape already read after numeric-type validation
     * @param targetShape non-null exact result Shape supplied by the caller
     * @throws IllegalArgumentException if target rank exceeds input rank, or if the first static
     *     aligned pair in increasing target-axis order is unequal and target is not one
     */
    private static void validateCompatibility(Shape inputShape, Shape targetShape) {
        int inputRank = inputShape.rank();
        int targetRank = targetShape.rank();
        if (targetRank > inputRank) {
            throw new IllegalArgumentException(
                    "sumToShape target rank must not exceed input rank: input="
                            + inputRank + ", target=" + targetRank);
        }

        int leadingInputAxes = inputRank - targetRank;
        for (int targetAxis = 0; targetAxis < targetRank; targetAxis++) {
            int inputAxis = leadingInputAxes + targetAxis;
            Dimension inputDimension = inputShape.dimension(inputAxis);
            Dimension targetDimension = targetShape.dimension(targetAxis);
            if (inputDimension instanceof StaticDimension inputStatic
                    && targetDimension instanceof StaticDimension targetStatic
                    && inputStatic.size() != targetStatic.size()
                    && targetStatic.size() != 1) {
                throw new IllegalArgumentException(
                        "sumToShape incompatible dimension at target axis " + targetAxis
                                + " (input axis " + inputAxis + "): input=" + inputDimension
                                + ", target=" + targetDimension);
            }
        }
    }

    /**
     * Constructs exact attributes, descriptor, operation, producer, provenance, and result state.
     *
     * <p>Construction creates those values in the documented order. The descriptor retains the
     * exact target Shape and input data type/eligibility with unresolved layout. The central
     * factory snapshots ordered input {@code [input]}, selects output index zero, assigns one fresh
     * identity, and attaches no label or storage.</p>
     *
     * @param input validated numeric source retained as the sole provenance input
     * @param targetShape validated exact semantic result Shape retained unchanged
     * @return the non-null exact fresh Tensor returned by central derived construction
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    private static Tensor create(Tensor input, Shape targetShape) {
        SumToShapeAttrs attrs = new SumToShapeAttrs(targetShape);
        TensorDescriptor descriptor = new TensorDescriptor(
                input.descriptor().dataType(),
                targetShape,
                Optional.empty(),
                input.descriptor().requiresGrad());
        Operation operation = new Operation(AggregateReductionKind.SUM, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(input));
    }
}
