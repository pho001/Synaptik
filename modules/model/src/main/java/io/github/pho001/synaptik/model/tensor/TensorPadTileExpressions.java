package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free constant-padding and per-axis tiling expressions.
 *
 * <p>Both operations require one width or repeat value per input axis and snapshot caller arrays
 * before constructing immutable semantic attributes. Static result extents use checked
 * {@code long} arithmetic. A dynamic Dimension is retained by exact reference only for an
 * identity transformation: zero padding on both sides or a repeat count of one. Scalar empty
 * requests and static zero extents are valid.</p>
 *
 * <p>Every result preserves the input DataType and gradient-eligibility value but deliberately
 * leaves layout unresolved. Padding introduces new positions and tiling repeats positions, so
 * neither result is described as an input view, even for an identity request. Construction
 * records exact one-input provenance without inspecting values or storage, converting padding
 * constants, defining gradients, capturing a graph, selecting materialization, lowering to a
 * backend, mapping ONNX, or executing work.</p>
 */
final class TensorPadTileExpressions {
    /** Prevents instantiation because pad/tile expression construction owns no state. */
    private TensorPadTileExpressions() {
    }

    /**
     * Validates one complete constant-padding request and creates a fresh PAD expression.
     *
     * <p>References are null-checked in order, then each array length is compared with the exact
     * input rank. The arrays are cloned in parameter order before boxing, and {@link PadAttrs}
     * owns width validation and immutable snapshots. The raw constant is retained unchanged for
     * every input data type. All failures before final factory delegation consume no Tensor
     * identity.</p>
     *
     * @param input non-null Tensor retained as the exact sole provenance input
     * @param before non-null caller-owned before widths, exactly one per input axis
     * @param after non-null caller-owned after widths, exactly one per input axis
     * @param constantValue exact raw binary64 padding constant retained without interpretation
     * @return a non-null fresh unlabeled, storage-free PAD expression with unresolved layout
     * @throws NullPointerException if {@code input}, {@code before}, or {@code after} is null,
     *     with that parameter name as the message
     * @throws IllegalArgumentException if an array length differs from input rank, a width is
     *     negative, or non-zero padding is requested for a dynamic dimension
     * @throws ArithmeticException if checked static extent addition overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor pad(Tensor input, long[] before, long[] after, double constantValue) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(before, "before");
        Objects.requireNonNull(after, "after");
        TensorDescriptor inputDescriptor = input.descriptor();
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
        Shape resultShape = paddedShape(inputShape, attrs);
        Operation operation = new Operation(PadKind.PAD, attrs);
        return create(input, inputDescriptor, resultShape, operation);
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
     * @throws IllegalArgumentException if the array length differs from input rank, a repeat is
     *     non-positive, or a repeat other than one is requested for a dynamic dimension
     * @throws ArithmeticException if checked static extent multiplication overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor tile(Tensor input, long[] repeats) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(repeats, "repeats");
        TensorDescriptor inputDescriptor = input.descriptor();
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
        return create(input, inputDescriptor, resultShape, operation);
    }

    /**
     * Derives one checked same-rank padding Shape.
     *
     * <p>A static extent {@code s} becomes a new static Dimension with exact value
     * {@code (s + before) + after}; Shape {@code [2]} with widths {@code [1]} and {@code [3]}
     * therefore becomes Shape {@code [6]}. A dynamic Dimension is retained by exact reference
     * only for zero widths on both sides. Empty attributes return the canonical scalar Shape.</p>
     *
     * @param inputShape non-null exact input Shape
     * @param attrs non-null validated rank-aligned padding attributes
     * @return a non-null checked same-rank Shape
     * @throws IllegalArgumentException if any dynamic axis has non-zero padding
     * @throws ArithmeticException if checked static extent addition overflows
     */
    private static Shape paddedShape(Shape inputShape, PadAttrs attrs) {
        Dimension[] resultDimensions = new Dimension[inputShape.rank()];
        for (int axis = 0; axis < resultDimensions.length; axis++) {
            Dimension dimension = inputShape.dimensions().get(axis);
            long before = attrs.before().get(axis);
            long after = attrs.after().get(axis);
            if (dimension instanceof StaticDimension staticDimension) {
                resultDimensions[axis] = new StaticDimension(Math.addExact(
                        Math.addExact(staticDimension.size(), before), after));
            } else if (before == 0 && after == 0) {
                resultDimensions[axis] = dimension;
            } else {
                throw new IllegalArgumentException(
                        "cannot pad dynamic axis " + axis
                                + " with before=" + before + " and after=" + after);
            }
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Derives one checked same-rank tiling Shape.
     *
     * <p>A static extent {@code s} becomes a new static Dimension with exact value
     * {@code s * repeat}; Shape {@code [2, 3]} with repeats {@code [2, 4]} therefore becomes
     * Shape {@code [4, 12]}. A dynamic Dimension is retained by exact reference only for repeat
     * one. Empty attributes return the canonical scalar Shape.</p>
     *
     * @param inputShape non-null exact input Shape
     * @param attrs non-null validated rank-aligned tiling attributes
     * @return a non-null checked same-rank Shape
     * @throws IllegalArgumentException if any dynamic axis has a repeat other than one
     * @throws ArithmeticException if checked static extent multiplication overflows
     */
    private static Shape tiledShape(Shape inputShape, TileAttrs attrs) {
        Dimension[] resultDimensions = new Dimension[inputShape.rank()];
        for (int axis = 0; axis < resultDimensions.length; axis++) {
            Dimension dimension = inputShape.dimensions().get(axis);
            long repeat = attrs.repeats().get(axis);
            if (dimension instanceof StaticDimension staticDimension) {
                resultDimensions[axis] =
                        new StaticDimension(Math.multiplyExact(staticDimension.size(), repeat));
            } else if (repeat == 1) {
                resultDimensions[axis] = dimension;
            } else {
                throw new IllegalArgumentException(
                        "cannot tile dynamic axis " + axis + " with repeat=" + repeat);
            }
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Creates one unresolved result descriptor, exact operation provenance, and fresh Tensor.
     *
     * <p>The descriptor retains exact input data type and gradient eligibility with the supplied
     * Shape and unresolved layout. Provenance retains the exact operation and ordered input
     * {@code [input]}. The central derived factory is invoked exactly once with no label or
     * storage.</p>
     *
     * @param input non-null exact sole provenance input
     * @param inputDescriptor non-null exact descriptor supplying data type and eligibility
     * @param resultShape non-null locally derived result Shape
     * @param operation non-null exact PAD or TILE semantic operation
     * @return a non-null fresh unlabeled, storage-free Tensor with unresolved layout
     * @throws IllegalStateException if tensor identifier space is exhausted at factory delegation
     */
    private static Tensor create(
            Tensor input,
            TensorDescriptor inputDescriptor,
            Shape resultShape,
            Operation operation) {
        TensorDescriptor descriptor = new TensorDescriptor(
                inputDescriptor.dataType(),
                resultShape,
                Optional.empty(),
                inputDescriptor.requiresGrad());
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(input));
    }
}
