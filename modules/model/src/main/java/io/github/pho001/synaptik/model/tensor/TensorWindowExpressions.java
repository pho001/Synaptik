package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs the locally validated public unfold, unfold2d, and fold2d Tensor expressions.
 *
 * <p>General-axis unfold replaces one static extent with the count of positive-step windows and
 * appends window size. Two-dimensional unfold maps rank-four NCHW (batch, channel, height, width)
 * geometry to canonical im2col columns; two-dimensional fold maps compatible columns back through
 * overlap-accumulating col2im. The retained compiler-only
 * {@link WindowTransformKind#FOLD_AXIS} semantic has no public construction path in this helper;
 * task 0023 owns its first compiler-generated construction.</p>
 *
 * <p>All Shape arithmetic is local and checked. Floor and ceil window counts use quotient and
 * remainder arithmetic, avoiding {@code numerator + stride - 1}. Results always leave layout
 * unresolved and carry exact one-input provenance. This helper owns no state and never reads
 * values or storage, materializes windows, defines gradients, captures graphs, or executes work.</p>
 */
final class TensorWindowExpressions {
    /** Prevents instantiation because window-expression construction owns no state. */
    private TensorWindowExpressions() {
    }

    /**
     * Validates and creates one general-axis unfold expression.
     *
     * @param input non-null exact sole provenance input
     * @param axis raw positive or negative input axis
     * @param size positive window extent in logical elements
     * @param step positive distance between window starts in logical elements
     * @return a non-null fresh UNFOLD_AXIS Tensor with checked Shape and unresolved layout
     * @throws NullPointerException if {@code input} is null, with message {@code input}
     * @throws IllegalArgumentException if rank, intrinsic attributes, selected staticity, or size
     *     fit is invalid
     * @throws IndexOutOfBoundsException if {@code axis} is outside input rank
     * @throws ArithmeticException if checked window-count arithmetic overflows
     */
    static Tensor unfoldAxis(Tensor input, int axis, long size, long step) {
        Objects.requireNonNull(input, "input");
        Shape inputShape = input.descriptor().shape();
        if (inputShape.rank() < 1) {
            throw new IllegalArgumentException("unfold requires rank at least 1");
        }
        int normalizedAxis = normalizeAxis(axis, inputShape.rank());
        UnfoldAxisAttrs attrs = new UnfoldAxisAttrs(normalizedAxis, size, step);
        long selectedSize = requireStaticSize(
                inputShape, normalizedAxis, "unfold", "selected");
        if (size > selectedSize) {
            throw new IllegalArgumentException(
                    "unfold size " + size + " exceeds selected dimension " + selectedSize);
        }
        Shape resultShape = unfoldAxisShape(inputShape, attrs);
        Operation operation = new Operation(WindowTransformKind.UNFOLD_AXIS, attrs);
        return create(input, resultShape, operation);
    }

    /**
     * Validates rank-four NCHW geometry and creates canonical rank-three im2col metadata.
     *
     * @param input non-null rank-four floating NCHW Tensor retained as sole provenance input
     * @param window non-null exact symmetric kernel/stride/padding/dilation geometry
     * @return a non-null fresh UNFOLD2D Tensor with Shape
     *     {@code [N, C * kernelHeight * kernelWidth, outputHeight * outputWidth]}
     * @throws NullPointerException if {@code input} or {@code window} is null, in that order
     * @throws IllegalArgumentException if rank, type, required staticity, or kernel fit is invalid
     * @throws ArithmeticException if checked geometry arithmetic overflows
     */
    static Tensor unfold2d(Tensor input, Window2dAttrs window) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(window, "window");
        Shape inputShape = input.descriptor().shape();
        if (inputShape.rank() != 4) {
            throw new IllegalArgumentException("unfold2d requires rank-4 NCHW input");
        }
        validateFloating(input.descriptor().dataType(), "unfold2d");
        Shape resultShape = unfold2dShape(inputShape, window);
        Operation operation = new Operation(WindowTransformKind.UNFOLD2D, window);
        return create(input, resultShape, operation);
    }

    /**
     * Validates canonical columns against explicit NCHW geometry and creates fold metadata.
     *
     * @param input non-null rank-three floating canonical-column Tensor
     * @param outputShape non-null explicit rank-four NCHW result Shape retained exactly
     * @param window non-null exact symmetric kernel/stride/padding/dilation geometry retained
     *     exactly
     * @return a non-null fresh FOLD2D Tensor with the exact output Shape and unresolved layout
     * @throws NullPointerException if a reference is null, checked in parameter order
     * @throws IllegalArgumentException if rank, type, batch, staticity, kernel fit, channel-window,
     *     or window-count compatibility is invalid
     * @throws ArithmeticException if checked geometry arithmetic overflows
     */
    static Tensor fold2d(Tensor input, Shape outputShape, Window2dAttrs window) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(outputShape, "outputShape");
        Objects.requireNonNull(window, "window");
        Shape inputShape = input.descriptor().shape();
        if (inputShape.rank() != 3) {
            throw new IllegalArgumentException("fold2d requires rank-3 canonical column input");
        }
        if (outputShape.rank() != 4) {
            throw new IllegalArgumentException("fold2d outputShape must be rank-4 NCHW");
        }
        validateFloating(input.descriptor().dataType(), "fold2d");
        if (!inputShape.dimensions().get(0).equals(outputShape.dimensions().get(0))) {
            throw new IllegalArgumentException(
                    "fold2d output batch dimension must match column batch dimension");
        }
        validateFold2dShape(inputShape, outputShape, window);
        Fold2dAttrs attrs = new Fold2dAttrs(outputShape, window);
        Operation operation = new Operation(WindowTransformKind.FOLD2D, attrs);
        return create(input, outputShape, operation);
    }

    /**
     * Requires one of FLOAT64, FLOAT32, or BFLOAT16 for NCHW transforms.
     *
     * @param dataType non-null exact input data type
     * @param operation non-null constant operation name used in failures
     * @throws IllegalArgumentException if {@code dataType} is not floating
     */
    private static void validateFloating(DataType dataType, String operation) {
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    operation + " requires floating input: " + dataType);
        }
    }

    /**
     * Normalizes one raw axis against an existing positive rank using overflow-safe arithmetic.
     *
     * @param axis raw positive or negative axis
     * @param rank positive rank that defines the accepted range
     * @return normalized non-negative axis below {@code rank}
     * @throws IndexOutOfBoundsException if {@code axis} is outside {@code [-rank, rank - 1]}
     */
    private static int normalizeAxis(int axis, int rank) {
        long normalizedAxis = axis;
        if (normalizedAxis < 0) {
            normalizedAxis += rank;
        }
        if (normalizedAxis < 0 || normalizedAxis >= rank) {
            throw new IndexOutOfBoundsException(
                    "Axis " + axis + " is outside shape rank " + rank);
        }
        return (int) normalizedAxis;
    }

    /**
     * Returns one required static Shape extent with operation-specific failure text.
     *
     * @param shape non-null Shape containing {@code axis}
     * @param axis normalized existing axis
     * @param operation non-null constant operation name
     * @param dimension non-null dimension role used in the exact failure message
     * @return non-negative static extent at {@code axis}
     * @throws IllegalArgumentException if the selected Dimension is dynamic
     */
    private static long requireStaticSize(
            Shape shape, int axis, String operation, String dimension) {
        Dimension selected = shape.dimensions().get(axis);
        if (!(selected instanceof StaticDimension staticDimension)) {
            throw new IllegalArgumentException(
                    operation + " requires static " + dimension + " dimension at axis " + axis);
        }
        return staticDimension.size();
    }

    /**
     * Derives axis-unfold Shape while preserving exact unaffected Dimension references.
     *
     * @param inputShape non-null rank-one-or-greater input Shape
     * @param attrs non-null normalized unfold attributes whose size fits the selected extent
     * @return non-null rank-one-larger Shape with window count at the selected axis and window size
     *     at the final axis
     * @throws ArithmeticException if checked window-count addition overflows
     */
    private static Shape unfoldAxisShape(Shape inputShape, UnfoldAxisAttrs attrs) {
        long selectedSize = ((StaticDimension) inputShape.dimensions().get(attrs.axis())).size();
        long windowCount = Math.addExact(
                (selectedSize - attrs.size()) / attrs.step(), 1L);
        Dimension[] dimensions = new Dimension[inputShape.rank() + 1];
        for (int index = 0; index < inputShape.rank(); index++) {
            dimensions[index] = index == attrs.axis()
                    ? new StaticDimension(windowCount)
                    : inputShape.dimensions().get(index);
        }
        dimensions[inputShape.rank()] = new StaticDimension(attrs.size());
        return Shape.ofDimensions(dimensions);
    }

    /**
     * Calculates checked canonical im2col Shape while retaining the exact batch Dimension.
     *
     * @param inputShape non-null rank-four NCHW Shape
     * @param window non-null validated intrinsic window geometry
     * @return non-null rank-three canonical-column Shape
     * @throws IllegalArgumentException if channel, height, or width is dynamic or a kernel does
     *     not fit its padded spatial dimension
     * @throws ArithmeticException if checked geometry arithmetic overflows
     */
    private static Shape unfold2dShape(Shape inputShape, Window2dAttrs window) {
        long channels = requireStaticSize(inputShape, 1, "unfold2d", "channel");
        long height = requireStaticSize(inputShape, 2, "unfold2d", "height");
        long width = requireStaticSize(inputShape, 3, "unfold2d", "width");
        long channelWindows = Math.multiplyExact(
                Math.multiplyExact(channels, window.kernelHeight()), window.kernelWidth());
        long outputHeight = windowOutputSize(
                height,
                window.kernelHeight(),
                window.paddingHeight(),
                window.strideHeight(),
                window.dilationHeight(),
                window.ceilMode(),
                "unfold2d",
                "height");
        long outputWidth = windowOutputSize(
                width,
                window.kernelWidth(),
                window.paddingWidth(),
                window.strideWidth(),
                window.dilationWidth(),
                window.ceilMode(),
                "unfold2d",
                "width");
        long windowCount = Math.multiplyExact(outputHeight, outputWidth);
        return Shape.ofDimensions(
                inputShape.dimensions().get(0),
                new StaticDimension(channelWindows),
                new StaticDimension(windowCount));
    }

    /**
     * Validates canonical columns against static output NCHW channel and spatial geometry.
     *
     * @param inputShape non-null rank-three columns with batch already matched
     * @param outputShape non-null rank-four NCHW target Shape
     * @param window non-null intrinsic window geometry
     * @throws IllegalArgumentException if required dimensions are dynamic, effective kernel does
     *     not fit, or channel/window counts differ from checked expected values
     * @throws ArithmeticException if checked expected-geometry arithmetic overflows
     */
    private static void validateFold2dShape(
            Shape inputShape, Shape outputShape, Window2dAttrs window) {
        long actualChannelWindows = requireStaticSize(
                inputShape, 1, "fold2d", "column-channel");
        long actualWindowCount = requireStaticSize(
                inputShape, 2, "fold2d", "column-count");
        long outputChannels = requireStaticSize(
                outputShape, 1, "fold2d", "output channel");
        long outputHeightSize = requireStaticSize(
                outputShape, 2, "fold2d", "output height");
        long outputWidthSize = requireStaticSize(
                outputShape, 3, "fold2d", "output width");
        long expectedChannelWindows = Math.multiplyExact(
                Math.multiplyExact(outputChannels, window.kernelHeight()), window.kernelWidth());
        if (actualChannelWindows != expectedChannelWindows) {
            throw new IllegalArgumentException(
                    "fold2d column-channel dimension " + actualChannelWindows
                            + " does not match output channels and kernel geometry: expected="
                            + expectedChannelWindows);
        }
        long outputHeight = windowOutputSize(
                outputHeightSize,
                window.kernelHeight(),
                window.paddingHeight(),
                window.strideHeight(),
                window.dilationHeight(),
                window.ceilMode(),
                "fold2d",
                "height");
        long outputWidth = windowOutputSize(
                outputWidthSize,
                window.kernelWidth(),
                window.paddingWidth(),
                window.strideWidth(),
                window.dilationWidth(),
                window.ceilMode(),
                "fold2d",
                "width");
        long expectedWindowCount = Math.multiplyExact(outputHeight, outputWidth);
        if (actualWindowCount != expectedWindowCount) {
            throw new IllegalArgumentException(
                    "fold2d column count " + actualWindowCount
                            + " does not match output shape and window geometry: expected="
                            + expectedWindowCount);
        }
    }

    /**
     * Calculates one checked floor- or ceil-mode spatial window count.
     *
     * <p>Effective kernel is {@code dilation * (kernel - 1) + 1}; padded input is
     * {@code inputSize + 2 * padding}. A negative difference means the effective kernel does not
     * fit. Ceil mode rounds through quotient and remainder, never through the overflow-prone
     * {@code numerator + stride - 1} expression.</p>
     *
     * @param inputSize non-negative static input spatial extent
     * @param kernel positive count of kernel samples
     * @param padding non-negative symmetric padding on each side
     * @param stride positive distance between window starts
     * @param dilation positive spacing between kernel samples
     * @param ceilMode true for ceil quotient rounding, false for floor rounding
     * @param operation non-null constant operation name for failures
     * @param dimension non-null spatial dimension name for failures
     * @return positive checked window-position count
     * @throws IllegalArgumentException if effective kernel does not fit padded input
     * @throws ArithmeticException if any checked multiplication, addition, or subtraction
     *     overflows
     */
    private static long windowOutputSize(
            long inputSize,
            long kernel,
            long padding,
            long stride,
            long dilation,
            boolean ceilMode,
            String operation,
            String dimension) {
        long effectiveKernel = Math.addExact(
                Math.multiplyExact(dilation, Math.subtractExact(kernel, 1L)), 1L);
        long paddedInput = Math.addExact(inputSize, Math.multiplyExact(2L, padding));
        long numerator = Math.subtractExact(paddedInput, effectiveKernel);
        if (numerator < 0) {
            throw new IllegalArgumentException(
                    operation + " effective kernel does not fit padded " + dimension);
        }
        long quotient = numerator / stride;
        if (ceilMode && numerator % stride != 0) {
            quotient = Math.addExact(quotient, 1L);
        }
        return Math.addExact(quotient, 1L);
    }

    /**
     * Creates one exact unresolved descriptor and delegates producer state for a fresh Tensor.
     *
     * @param input non-null exact sole provenance input and descriptor metadata source
     * @param resultShape non-null derived or exact retained result Shape
     * @param operation non-null exact typed window operation
     * @return non-null fresh unlabeled, storage-free Tensor with unresolved layout
     * @throws IllegalStateException if tensor identifier space is exhausted
     */
    private static Tensor create(Tensor input, Shape resultShape, Operation operation) {
        TensorDescriptor inputDescriptor = input.descriptor();
        TensorDescriptor descriptor = new TensorDescriptor(
                inputDescriptor.dataType(),
                resultShape,
                Optional.empty(),
                inputDescriptor.requiresGrad());
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(input));
    }
}
