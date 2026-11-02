package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free constant-padding and per-axis tiling expressions.
 *
 * <p>Both operations require one width or repeat value per input axis and snapshot caller arrays
 * before constructing immutable semantic attributes. Each padded extent is the canonical
 * symbolic formula {@code (input + before) + after}, and each tiled extent is
 * {@code input * repeat}. Static values are folded, while zero/zero padding and repeat one retain
 * the exact input Dimension reference. Static values, symbolic coefficients, and symbolic offsets
 * use checked {@code long} arithmetic. Scalar empty requests and static zero extents are valid.</p>
 *
 * <p>Every result preserves the input DataType and gradient-eligibility value but deliberately
 * leaves layout unresolved. Padding introduces new positions and tiling repeats positions, so
 * neither result is described as an input view, even for an identity request. Construction
 * records exact one-input provenance without inspecting values or storage, converting typed
 * padding constants, defining gradients, binding or evaluating symbolic extents, capturing a graph,
 * selecting materialization, lowering to a backend, mapping ONNX, or executing work.</p>
 */
final class TensorPadTileExpressions {
    /** Prevents instantiation because pad/tile expression construction owns no state. */
    private TensorPadTileExpressions() {
    }

    /**
     * Validates one complete constant-padding request and creates a fresh PAD expression.
     *
     * <p>References are null-checked in parameter order, then each array length is compared with
     * the exact input rank. The arrays are cloned in parameter order before boxing, and
     * {@link PadAttrs} owns width validation and immutable snapshots. Exact constant/input data-
     * type equality is checked after width validation and before Shape arithmetic. All failures
     * before final factory delegation consume no Tensor identity.</p>
     *
     * @param input non-null Tensor retained as the exact sole provenance input
     * @param before non-null caller-owned before widths, exactly one per input axis
     * @param after non-null caller-owned after widths, exactly one per input axis
     * @param constantValue non-null exact typed padding constant; its data type must equal the
     *     input data type and it is retained by exact reference
     * @return a non-null fresh unlabeled, storage-free PAD expression with unresolved layout
     * @throws NullPointerException if {@code input}, {@code before}, {@code after}, or
     *     {@code constantValue} is null, checked in that order with the parameter name as the
     *     message
     * @throws IllegalArgumentException if an array length differs from input rank or a width is
     *     negative, or the constant and input data types differ
     * @throws ArithmeticException if checked static extent or symbolic-offset addition overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor pad(Tensor input, long[] before, long[] after, ScalarValue constantValue) {
        return padChecked(input, before, after, constantValue, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null Tensor retained as the exact sole provenance input
     * @param before non-null caller-owned before widths, exactly one per input axis
     * @param after non-null caller-owned after widths, exactly one per input axis
     * @param constantValue non-null exact typed padding constant; its data type must equal the
     *     input data type and it is retained by exact reference
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input}, {@code before}, {@code after}, or
     *     {@code constantValue} is null, checked in that order with the parameter name as the
     *     message
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if an array length differs from input rank or a width is
     *     negative, or the constant and input data types differ
     * @throws ArithmeticException if checked static extent or symbolic-offset addition overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor padChecked(Tensor input, long[] before, long[] after, ScalarValue constantValue,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return padDescriptors(
                input == null ? null : input.descriptor(),
                before,
                after,
                constantValue,
                descriptorCheck,
                input);
    }

    /**
     * Validates one complete-pattern repeat request and creates a fresh TILE expression.
     *
     * <p>References are null-checked in order, then the repeat count is compared with exact input
     * rank. The caller array is cloned before boxing, and {@link TileAttrs} owns positive-value
     * validation and its immutable snapshot. All failures before final factory delegation consume
     * no Tensor identity.</p>
     *
     * @param input non-null Tensor retained as the exact sole provenance input
     * @param repeats non-null caller-owned positive repeat counts, exactly one per input axis
     * @return a non-null fresh unlabeled, storage-free TILE expression with unresolved layout
     * @throws NullPointerException if {@code input} or {@code repeats} is null, with that parameter
     *     name as the message
     * @throws IllegalArgumentException if the array length differs from input rank or a repeat is
     *     non-positive
     * @throws ArithmeticException if checked static extent, symbolic coefficient, or symbolic
     *     offset multiplication overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor tile(Tensor input, long[] repeats) {
        return tileChecked(input, repeats, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input non-null Tensor retained as the exact sole provenance input
     * @param repeats non-null caller-owned positive repeat counts, exactly one per input axis
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input} or {@code repeats} is null, with that parameter
     *     name as the message
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if the array length differs from input rank or a repeat is
     *     non-positive
     * @throws ArithmeticException if checked static extent, symbolic coefficient, or symbolic
     *     offset multiplication overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor tileChecked(Tensor input, long[] repeats,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        return tileDescriptors(
                input == null ? null : input.descriptor(),
                repeats,
                descriptorCheck,
                input);
    }

    /**
     * Derives one checked same-rank padding Shape.
     *
     * <p>Each result extent is constructed as
     * {@code addConstant(addConstant(input, before), after)}. Static extent {@code 2} with widths
     * {@code 1} and {@code 3} therefore becomes static {@code 6}, while named extent {@code N}
     * becomes the canonical formula {@code N + 4}. Widths are applied in before-then-after order
     * for deterministic checked failure timing. Zero widths preserve the exact input Dimension
     * reference, and empty attributes return the canonical scalar Shape. The formula is retained
     * without binding or evaluating a concrete size.</p>
     *
     * @param inputShape non-null exact input Shape
     * @param attrs non-null validated rank-aligned padding attributes
     * @return a non-null checked same-rank Shape
     * @throws ArithmeticException if checked static extent or symbolic-offset addition overflows
     */
    private static Shape paddedShape(Shape inputShape, PadAttrs attrs) {
        Dimension[] resultDimensions = new Dimension[inputShape.rank()];
        for (int axis = 0; axis < resultDimensions.length; axis++) {
            Dimension dimension = inputShape.dimensions().get(axis);
            long before = attrs.before().get(axis);
            long after = attrs.after().get(axis);
            resultDimensions[axis] = DimensionExpressions.addConstant(
                    DimensionExpressions.addConstant(dimension, before), after);
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Derives one checked same-rank tiling Shape.
     *
     * <p>Each result extent is constructed as {@code multiply(input, repeat)}. Shape
     * {@code [2, 3]} with repeats {@code [2, 4]} therefore becomes Shape {@code [4, 12]}, while
     * named extent {@code N} repeated four times becomes the canonical formula {@code 4 * N}.
     * Repeat one preserves the exact input Dimension reference, and empty attributes return the
     * canonical scalar Shape. The formula is retained without binding or evaluating a concrete
     * size.</p>
     *
     * @param inputShape non-null exact input Shape
     * @param attrs non-null validated rank-aligned tiling attributes
     * @return a non-null checked same-rank Shape
     * @throws ArithmeticException if checked static extent, symbolic coefficient, or symbolic
     *     offset multiplication overflows
     */
    private static Shape tiledShape(Shape inputShape, TileAttrs attrs) {
        Dimension[] resultDimensions = new Dimension[inputShape.rank()];
        for (int axis = 0; axis < resultDimensions.length; axis++) {
            Dimension dimension = inputShape.dimensions().get(axis);
            long repeat = attrs.repeats().get(axis);
            resultDimensions[axis] = DimensionExpressions.multiply(dimension, repeat);
        }
        return Shape.ofDimensions(resultDimensions);
    }



    /**
     * Completes ordinary checked semantics synchronously without creating Tensor identities.
     * The same descriptor guard core serves real construction; the sink is never retained.
     * @param operation non-null ordinary operation selecting this family and its attributes
     * @param inputs non-null ordered operand descriptors, including repeated occurrences
     * @param outputCount exact canonical sibling count
     * @param sink non-null consumer called exactly once with checked semantics and all outputs
     * @throws NullPointerException if a required reference is null
     * @throws IllegalArgumentException if cardinality or ordinary semantic guards reject the request
     */
    static void describeChecked(Operation operation, List<TensorDescriptor> inputs, int outputCount,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> sink) {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(sink, "sink");
        if (inputs.size() != 1 || outputCount != 1) {
            throw new IllegalArgumentException("invalid ordinary descriptor cardinality");
        }
        for (TensorDescriptor input : inputs) {
            Objects.requireNonNull(input, "input descriptor");
        }
        if (operation.kind() == PadKind.PAD) {
            PadAttrs attrs = (PadAttrs) operation.attrs();
            padDescriptors(inputs.get(0),
                    attrs.before().stream().mapToLong(Long::longValue).toArray(),
                    attrs.after().stream().mapToLong(Long::longValue).toArray(),
                    attrs.constantValue(), sink, null);
        } else {
            tileDescriptors(inputs.get(0),
                    ((TileAttrs) operation.attrs()).repeats().stream().mapToLong(Long::longValue).toArray(),
                    sink, null);
        }
    }

    /**
     * Runs the shared descriptor guards; provenance references are used only for real construction.
     * @param before ordinary checked construction parameter
     * @param after ordinary checked construction parameter
     * @param constantValue ordinary checked construction parameter
     * @param descriptorCheck synchronous, unretained sink; null selects real construction
     * @param originalInput exact real operand, or null for descriptor completion
     * @return the real result, or null after synchronous descriptor completion
     */
    private static Tensor padDescriptors(
            TensorDescriptor input,
            long[] before,
            long[] after,
            ScalarValue constantValue,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalInput) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(before, "before");
        Objects.requireNonNull(after, "after");
        Objects.requireNonNull(constantValue, "constantValue");
        TensorDescriptor inputDescriptor = input;
        Shape inputShape = inputDescriptor.shape();
        int rank = inputShape.rank();
        if (before.length != rank) {
            throw new IllegalArgumentException(
                    "padding before length " + before.length + " must equal input rank " + rank);
        }
        if (after.length != rank) {
            throw new IllegalArgumentException(
                    "padding after length " + after.length + " must equal input rank " + rank);
        }
        long[] privateBefore = before.clone();
        long[] privateAfter = after.clone();
        PadAttrs attrs = new PadAttrs(
                Arrays.stream(privateBefore).boxed().toList(),
                Arrays.stream(privateAfter).boxed().toList(),
                constantValue);
        if (constantValue.dataType() != inputDescriptor.dataType()) {
            throw new IllegalArgumentException(
                    "padding constant data type " + constantValue.dataType()
                            + " must match input data type " + inputDescriptor.dataType());
        }
        Shape resultShape = paddedShape(inputShape, attrs);
        Operation operation = new Operation(PadKind.PAD, attrs);
        return createDescriptors(
                inputDescriptor,
                resultShape,
                operation,
                descriptorCheck,
                originalInput);
    }

    /**
     * Runs the shared descriptor guards; provenance references are used only for real construction.
     * @param input descriptor supplying ordinary operand metadata
     * @param repeats ordinary checked construction parameter
     * @param descriptorCheck synchronous, unretained sink; null selects real construction
     * @param originalInput exact real operand, or null for descriptor completion
     * @return the real result, or null after synchronous descriptor completion
     */
    private static Tensor tileDescriptors(
            TensorDescriptor input,
            long[] repeats,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalInput) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(repeats, "repeats");
        TensorDescriptor inputDescriptor = input;
        Shape inputShape = inputDescriptor.shape();
        int rank = inputShape.rank();
        if (repeats.length != rank) {
            throw new IllegalArgumentException(
                    "tile repeats length " + repeats.length + " must equal input rank " + rank);
        }
        long[] privateRepeats = repeats.clone();
        TileAttrs attrs = new TileAttrs(Arrays.stream(privateRepeats).boxed().toList());
        Shape resultShape = tiledShape(inputShape, attrs);
        Operation operation = new Operation(TileKind.TILE, attrs);
        return createDescriptors(
                inputDescriptor,
                resultShape,
                operation,
                descriptorCheck,
                originalInput);
    }

    /**
     * Runs the shared descriptor guards; provenance references are used only for real construction.
     * @param input descriptor supplying ordinary operand metadata
     * @param inputDescriptor ordinary checked construction parameter
     * @param resultShape ordinary checked construction parameter
     * @param operation ordinary checked construction parameter
     * @param descriptorCheck synchronous, unretained sink; null selects real construction
     * @param originalInput exact real operand, or null for descriptor completion
     * @return the real result, or null after synchronous descriptor completion
     */
    private static Tensor createDescriptors(
            TensorDescriptor inputDescriptor,
            Shape resultShape,
            Operation operation,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor originalInput) {
        TensorDescriptor descriptor = new TensorDescriptor(
                inputDescriptor.dataType(),
                resultShape,
                Optional.empty(),
                inputDescriptor.requiresGrad());
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(originalInput));
    }
}
