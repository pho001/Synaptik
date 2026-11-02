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
        return applyChecked(input, targetShape, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null numeric Tensor retained by exact reference as the sole provenance
     *     input; its metadata, storage association, values, and identity remain unchanged
     * @param targetShape non-null exact result Shape to retain in both attributes and descriptor;
     *     its rank must not exceed the input rank
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input} or {@code targetShape} is null, checked in
     *     that order with the parameter name as the message
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if the input is BOOL, the target rank exceeds the input
     *     rank, or the first fully static aligned pair is neither equal nor target-one
     * @throws IllegalStateException if tensor identifier space is exhausted after local immutable
     *     expression metadata has been constructed
     */
    static Tensor applyChecked(Tensor input, Shape targetShape,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(targetShape, "targetShape");
        return applyDescriptors(
                input.descriptor(), targetShape, descriptorCheck, input);
    }

    /**
     * Runs the shared descriptor guards and completes metadata before real provenance is used.
     * @param input immutable operand metadata
     * @param targetShape ordinary construction argument, validated in existing order
     * @param descriptorCheck optional synchronous, unretained metadata completion; null constructs a Tensor
     * @param inputTensor real provenance, unused when metadata completion is supplied
     * @return null after metadata completion, otherwise the ordinary result
     */
    private static Tensor applyDescriptors(
            TensorDescriptor input,
            Shape targetShape,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor inputTensor) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(targetShape, "targetShape");
        validateInput(input);
        Shape inputShape = input.shape();
        validateCompatibility(inputShape, targetShape);
        return createDescriptors(input, targetShape, descriptorCheck, inputTensor);
    }

    /**
     * Requires one of the five current numeric input types without conversion or promotion.
     *
     * @param input non-null descriptor inspected only for its immutable data type
     * @throws IllegalArgumentException if the input type is BOOL, with the exact SUM diagnostic
     */
    private static void validateInput(TensorDescriptor input) {
        DataType dataType = input.dataType();
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
     * Runs the shared descriptor guards and completes metadata before real provenance is used.
     * @param input immutable operand metadata
     * @param targetShape ordinary construction argument, validated in existing order
     * @param descriptorCheck optional synchronous, unretained metadata completion; null constructs a Tensor
     * @param inputTensor real provenance, unused when metadata completion is supplied
     * @return null after metadata completion, otherwise the ordinary result
     */
    private static Tensor createDescriptors(
            TensorDescriptor input,
            Shape targetShape,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor inputTensor) {
        SumToShapeAttrs attrs = new SumToShapeAttrs(targetShape);
        TensorDescriptor descriptor = new TensorDescriptor(
                input.dataType(),
                targetShape,
                Optional.empty(),
                input.requiresGrad());
        Operation operation = new Operation(AggregateReductionKind.SUM, attrs);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(inputTensor));
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
        SumToShapeAttrs attrs = (SumToShapeAttrs) operation.attrs();
        applyDescriptors(inputs.get(0), attrs.targetShape(), sink, null);
    }
}
