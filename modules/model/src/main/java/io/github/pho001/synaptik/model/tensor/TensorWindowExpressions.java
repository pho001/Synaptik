package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs the locally validated public general-axis and NCHW window Tensor expressions.
 *
 * <p>General-axis unfold replaces one static extent with the count of positive-step windows and
 * appends window size. General-axis fold removes that final dimension and restores an explicit
 * target extent under overlap summation. Two-dimensional unfold maps rank-four NCHW (batch,
 * channel, height, width) geometry to canonical im2col columns; two-dimensional fold maps
 * compatible columns back through overlap-accumulating col2im.</p>
 *
 * <p>All Shape arithmetic is local and checked. Static floor and ceil window counts use quotient
 * and remainder arithmetic, avoiding {@code numerator + stride - 1}; unresolved 2D extents use
 * canonical linear, division, and product expressions. Results always leave layout unresolved
 * and carry exact one-input provenance. This helper owns no state and never reads values or
 * storage, materializes windows, defines gradients, captures graphs, or executes work.</p>
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
     * Validates and creates one general-axis overlap-summing fold expression.
     *
     * @param input non-null rank-two-or-greater window Tensor whose final dimension is window size
     * @param axis raw positive or negative target axis, excluding the final input dimension
     * @param outputSize non-negative restored target extent in logical elements
     * @param step positive distance between window starts in logical elements
     * @return a non-null fresh FOLD_AXIS Tensor with checked Shape and unresolved layout
     * @throws NullPointerException if {@code input} is null, with message {@code input}
     * @throws IllegalArgumentException if rank, attributes, type, staticity, final-window extent,
     *     or count geometry is invalid
     * @throws IndexOutOfBoundsException if {@code axis} is outside target rank
     * @throws ArithmeticException if checked expected-window arithmetic overflows
     */
    static Tensor foldAxis(Tensor input, int axis, long outputSize, long step) {
        Objects.requireNonNull(input, "input");
        Shape inputShape = input.descriptor().shape();
        if (inputShape.rank() < 2) {
            throw new IllegalArgumentException("foldAxis requires rank at least 2");
        }
        int targetRank = inputShape.rank() - 1;
        int normalizedAxis = normalizeAxis(axis, targetRank);
        FoldAxisAttrs attrs = new FoldAxisAttrs(normalizedAxis, outputSize, step);
        validateNumeric(input.descriptor().dataType(), "foldAxis");
        long actualWindows = requireStaticSize(
                inputShape, normalizedAxis, "foldAxis", "window-count");
        Dimension finalDimension = inputShape.dimensions().get(inputShape.rank() - 1);
        if (!(finalDimension instanceof StaticDimension staticWindow) || staticWindow.size() <= 0) {
            throw new IllegalArgumentException(
                    "foldAxis requires a positive static final window dimension");
        }
        long windowSize = staticWindow.size();
        if (outputSize == 0) {
            if (actualWindows != 0) {
                throw new IllegalArgumentException(
                        "foldAxis window count " + actualWindows
                                + " does not match output size and window geometry: expected=0");
            }
        } else {
            if (windowSize > outputSize) {
                throw new IllegalArgumentException(
                        "foldAxis window size " + windowSize
                                + " exceeds output size " + outputSize);
            }
            long expectedWindows = Math.addExact((outputSize - windowSize) / step, 1L);
            if (actualWindows != expectedWindows) {
                throw new IllegalArgumentException(
                        "foldAxis window count " + actualWindows
                                + " does not match output size and window geometry: expected="
                                + expectedWindows);
            }
        }
        Shape resultShape = foldAxisShape(inputShape, attrs);
        Operation operation = new Operation(WindowTransformKind.FOLD_AXIS, attrs);
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
     * @throws IllegalArgumentException if rank or type is invalid or static geometry proves that
     *     an effective kernel does not fit
     * @throws ArithmeticException if checked geometry or symbolic canonicalization overflows
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
     * Validates rank-four NCHW geometry and creates canonical columns with exact typed padding.
     *
     * @param input non-null rank-four floating NCHW Tensor retained as sole provenance input
     * @param window non-null exact symmetric kernel/stride/padding/dilation geometry
     * @param paddingValue non-null scalar whose exact type and bits supply out-of-domain samples
     * @return a non-null fresh UNFOLD2D Tensor with canonical rank-three Shape
     * @throws NullPointerException if a reference is null, checked in parameter order
     * @throws IllegalArgumentException if rank, input type, scalar type, or statically provable
     *     geometry is invalid
     * @throws ArithmeticException if checked geometry or symbolic canonicalization overflows
     */
    static Tensor unfold2d(
            Tensor input, Window2dAttrs window, ScalarValue paddingValue) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(window, "window");
        Objects.requireNonNull(paddingValue, "paddingValue");
        Shape inputShape = input.descriptor().shape();
        if (inputShape.rank() != 4) {
            throw new IllegalArgumentException("unfold2d requires rank-4 NCHW input");
        }
        DataType inputDataType = input.descriptor().dataType();
        validateFloating(inputDataType, "unfold2d");
        if (paddingValue.dataType() != inputDataType) {
            throw new IllegalArgumentException(
                    "unfold2d paddingValue data type must match input data type: paddingValue="
                            + paddingValue.dataType() + ", input=" + inputDataType);
        }
        Shape resultShape = unfold2dShape(inputShape, window);
        Unfold2dAttrs attrs = new Unfold2dAttrs(window, paddingValue);
        Operation operation = new Operation(WindowTransformKind.UNFOLD2D, attrs);
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
     * @throws IllegalArgumentException if rank, type, batch, effective-kernel fit, or exact
     *     structural channel/window compatibility is invalid
     * @throws ArithmeticException if checked geometry or symbolic canonicalization overflows
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
     * Creates canonical volumetric columns from one rank-five floating NCDHW tensor.
     *
     * @param input non-null exact sole provenance input
     * @param window non-null exact depth-height-width geometry retained as operation attributes
     * @return a fresh rank-three UNFOLD3D tensor with unresolved layout
     * @throws NullPointerException if a reference is null, checked in parameter order
     * @throws IllegalArgumentException if rank, type, or statically provable geometry is invalid
     * @throws ArithmeticException if checked geometry or symbolic canonicalization overflows
     */
    static Tensor unfold3d(Tensor input, Window3dAttrs window) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(window, "window");
        Shape inputShape = input.descriptor().shape();
        if (inputShape.rank() != 5) {
            throw new IllegalArgumentException("unfold3d requires rank-5 NCDHW input");
        }
        validateFloating(input.descriptor().dataType(), "unfold3d");
        Shape resultShape = unfold3dShape(inputShape, window);
        Operation operation = new Operation(WindowTransformKind.UNFOLD3D, window);
        return create(input, resultShape, operation);
    }

    /**
     * Creates canonical volumetric columns with one exact typed out-of-domain sample value.
     *
     * @param input non-null exact sole provenance input
     * @param window non-null exact depth-height-width geometry retained by reference
     * @param paddingValue non-null exact typed padding value retained by reference
     * @return a fresh rank-three UNFOLD3D tensor with unresolved layout
     * @throws NullPointerException if a reference is null, checked in parameter order
     * @throws IllegalArgumentException if rank, type, padding type, or static geometry is invalid
     * @throws ArithmeticException if checked geometry or symbolic canonicalization overflows
     */
    static Tensor unfold3d(
            Tensor input, Window3dAttrs window, ScalarValue paddingValue) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(window, "window");
        Objects.requireNonNull(paddingValue, "paddingValue");
        Shape inputShape = input.descriptor().shape();
        if (inputShape.rank() != 5) {
            throw new IllegalArgumentException("unfold3d requires rank-5 NCDHW input");
        }
        DataType inputDataType = input.descriptor().dataType();
        validateFloating(inputDataType, "unfold3d");
        if (paddingValue.dataType() != inputDataType) {
            throw new IllegalArgumentException(
                    "unfold3d paddingValue data type must match input data type: paddingValue="
                            + paddingValue.dataType() + ", input=" + inputDataType);
        }
        Shape resultShape = unfold3dShape(inputShape, window);
        Unfold3dAttrs attrs = new Unfold3dAttrs(window, paddingValue);
        Operation operation = new Operation(WindowTransformKind.UNFOLD3D, attrs);
        return create(input, resultShape, operation);
    }

    /**
     * Validates canonical volumetric columns and creates exact NCDHW fold metadata.
     *
     * @param input non-null rank-three floating canonical-column Tensor
     * @param outputShape non-null explicit rank-five NCDHW result Shape retained by reference
     * @param window non-null exact depth-height-width geometry retained by reference
     * @return a fresh FOLD3D tensor retaining the exact output Shape with unresolved layout
     * @throws NullPointerException if a reference is null, checked in parameter order
     * @throws IllegalArgumentException if rank, type, batch, geometry, or structural column
     *     compatibility is invalid
     * @throws ArithmeticException if checked geometry or symbolic canonicalization overflows
     */
    static Tensor fold3d(Tensor input, Shape outputShape, Window3dAttrs window) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(outputShape, "outputShape");
        Objects.requireNonNull(window, "window");
        Shape inputShape = input.descriptor().shape();
        if (inputShape.rank() != 3) {
            throw new IllegalArgumentException("fold3d requires rank-3 canonical column input");
        }
        if (outputShape.rank() != 5) {
            throw new IllegalArgumentException("fold3d outputShape must be rank-5 NCDHW");
        }
        validateFloating(input.descriptor().dataType(), "fold3d");
        if (!inputShape.dimensions().get(0).equals(outputShape.dimensions().get(0))) {
            throw new IllegalArgumentException(
                    "fold3d output batch dimension must match column batch dimension");
        }
        validateFold3dShape(inputShape, outputShape, window);
        Fold3dAttrs attrs = new Fold3dAttrs(outputShape, window);
        Operation operation = new Operation(WindowTransformKind.FOLD3D, attrs);
        return create(input, outputShape, operation);
    }

    /**
     * Requires a floating or integral input type for overlap-summing folding.
     *
     * @param dataType non-null exact input data type
     * @param operation non-null constant operation name used in failures
     * @throws IllegalArgumentException if {@code dataType} is BOOL
     */
    private static void validateNumeric(DataType dataType, String operation) {
        if (!dataType.isFloating() && !dataType.isIntegral()) {
            throw new IllegalArgumentException(
                    operation + " requires floating or integral input: " + dataType);
        }
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
     * Derives axis-fold Shape by removing the final window dimension and restoring target extent.
     *
     * @param inputShape non-null validated rank-two-or-greater window Shape
     * @param attrs non-null normalized fold attributes compatible with the input geometry
     * @return non-null rank-one-smaller Shape preserving exact unaffected Dimension references
     */
    private static Shape foldAxisShape(Shape inputShape, FoldAxisAttrs attrs) {
        Dimension[] dimensions = new Dimension[inputShape.rank() - 1];
        for (int index = 0; index < dimensions.length; index++) {
            dimensions[index] = index == attrs.axis()
                    ? new StaticDimension(attrs.outputSize())
                    : inputShape.dimensions().get(index);
        }
        return Shape.ofDimensions(dimensions);
    }

    /**
     * Calculates checked canonical im2col Shape while retaining the exact batch Dimension.
     *
     * @param inputShape non-null rank-four NCHW Shape
     * @param window non-null validated intrinsic window geometry
     * @return non-null rank-three canonical-column Shape
     * @throws IllegalArgumentException if static geometry proves that a kernel does not fit its
     *     padded spatial dimension
     * @throws ArithmeticException if checked geometry or symbolic canonicalization overflows
     */
    private static Shape unfold2dShape(Shape inputShape, Window2dAttrs window) {
        Dimension channels = inputShape.dimensions().get(1);
        Dimension height = inputShape.dimensions().get(2);
        Dimension width = inputShape.dimensions().get(3);
        Dimension channelWindows = DimensionExpressions.multiply(
                DimensionExpressions.multiply(channels, window.kernelHeight()),
                window.kernelWidth());
        Dimension outputHeight = windowOutputDimension(
                height,
                window.kernelHeight(),
                window.paddingHeight(),
                window.strideHeight(),
                window.dilationHeight(),
                window.ceilMode(),
                "unfold2d",
                "height");
        Dimension outputWidth = windowOutputDimension(
                width,
                window.kernelWidth(),
                window.paddingWidth(),
                window.strideWidth(),
                window.dilationWidth(),
                window.ceilMode(),
                "unfold2d",
                "width");
        Dimension windowCount = DimensionExpressions.multiply(outputHeight, outputWidth);
        return Shape.ofDimensions(
                inputShape.dimensions().get(0),
                channelWindows,
                windowCount);
    }

    /**
     * Validates canonical columns against static or symbolic output NCHW geometry.
     *
     * @param inputShape non-null rank-three columns with batch already matched
     * @param outputShape non-null rank-four NCHW target Shape
     * @param window non-null intrinsic window geometry
     * @throws IllegalArgumentException if static effective-kernel fit fails or the complete
     *     channel/window Dimensions differ structurally from the canonical expected values
     * @throws ArithmeticException if checked expected geometry or symbolic canonicalization
     *     overflows
     */
    private static void validateFold2dShape(
            Shape inputShape, Shape outputShape, Window2dAttrs window) {
        Dimension actualChannelWindows = inputShape.dimensions().get(1);
        Dimension actualWindowCount = inputShape.dimensions().get(2);
        Dimension outputChannels = outputShape.dimensions().get(1);
        Dimension outputHeightSize = outputShape.dimensions().get(2);
        Dimension outputWidthSize = outputShape.dimensions().get(3);
        Dimension expectedChannelWindows = DimensionExpressions.multiply(
                DimensionExpressions.multiply(outputChannels, window.kernelHeight()),
                window.kernelWidth());
        if (!actualChannelWindows.equals(expectedChannelWindows)) {
            throw new IllegalArgumentException(
                    "fold2d column-channel dimension "
                            + dimensionDiagnostic(actualChannelWindows)
                            + " does not match output channels and kernel geometry: expected="
                            + dimensionDiagnostic(expectedChannelWindows));
        }
        Dimension outputHeight = windowOutputDimension(
                outputHeightSize,
                window.kernelHeight(),
                window.paddingHeight(),
                window.strideHeight(),
                window.dilationHeight(),
                window.ceilMode(),
                "fold2d",
                "height");
        Dimension outputWidth = windowOutputDimension(
                outputWidthSize,
                window.kernelWidth(),
                window.paddingWidth(),
                window.strideWidth(),
                window.dilationWidth(),
                window.ceilMode(),
                "fold2d",
                "width");
        Dimension expectedWindowCount = DimensionExpressions.multiply(outputHeight, outputWidth);
        if (!actualWindowCount.equals(expectedWindowCount)) {
            throw new IllegalArgumentException(
                    "fold2d column count " + dimensionDiagnostic(actualWindowCount)
                            + " does not match output shape and window geometry: expected="
                            + dimensionDiagnostic(expectedWindowCount));
        }
    }

    /**
     * Calculates canonical rank-three columns for rank-five NCDHW input.
     *
     * @param inputShape non-null validated rank-five input Shape
     * @param window non-null validated intrinsic window geometry
     * @return non-null canonical rank-three column Shape retaining the exact batch Dimension
     * @throws IllegalArgumentException if a static effective kernel does not fit
     * @throws ArithmeticException if checked geometry or symbolic canonicalization overflows
     */
    private static Shape unfold3dShape(Shape inputShape, Window3dAttrs window) {
        Dimension depth = windowOutputDimension(
                inputShape.dimensions().get(2), window.kernelDepth(), window.paddingDepth(),
                window.strideDepth(), window.dilationDepth(), window.ceilMode(),
                "unfold3d", "depth");
        Dimension height = windowOutputDimension(
                inputShape.dimensions().get(3), window.kernelHeight(), window.paddingHeight(),
                window.strideHeight(), window.dilationHeight(), window.ceilMode(),
                "unfold3d", "height");
        Dimension width = windowOutputDimension(
                inputShape.dimensions().get(4), window.kernelWidth(), window.paddingWidth(),
                window.strideWidth(), window.dilationWidth(), window.ceilMode(),
                "unfold3d", "width");
        Dimension channelWindows = DimensionExpressions.multiply(
                DimensionExpressions.multiply(
                        DimensionExpressions.multiply(
                                inputShape.dimensions().get(1), window.kernelDepth()),
                        window.kernelHeight()),
                window.kernelWidth());
        Dimension windowCount = DimensionExpressions.multiply(
                DimensionExpressions.multiply(depth, height), width);
        return Shape.ofDimensions(
                inputShape.dimensions().get(0), channelWindows, windowCount);
    }

    /**
     * Validates canonical columns against exact static or symbolic NCDHW target geometry.
     *
     * @param inputShape non-null validated rank-three columns whose batch already matches
     * @param outputShape non-null validated rank-five NCDHW target
     * @param window non-null validated intrinsic window geometry
     * @throws IllegalArgumentException if channel, grid, or static effective-kernel fit fails
     * @throws ArithmeticException if checked geometry or symbolic canonicalization overflows
     */
    private static void validateFold3dShape(
            Shape inputShape, Shape outputShape, Window3dAttrs window) {
        Dimension actualChannelWindows = inputShape.dimensions().get(1);
        Dimension expectedChannelWindows = DimensionExpressions.multiply(
                DimensionExpressions.multiply(
                        DimensionExpressions.multiply(
                                outputShape.dimensions().get(1), window.kernelDepth()),
                        window.kernelHeight()),
                window.kernelWidth());
        if (!actualChannelWindows.equals(expectedChannelWindows)) {
            throw new IllegalArgumentException(
                    "fold3d column-channel dimension "
                            + dimensionDiagnostic(actualChannelWindows)
                            + " does not match output channels and kernel geometry: expected="
                            + dimensionDiagnostic(expectedChannelWindows));
        }
        Dimension depth = windowOutputDimension(
                outputShape.dimensions().get(2), window.kernelDepth(), window.paddingDepth(),
                window.strideDepth(), window.dilationDepth(), window.ceilMode(),
                "fold3d", "depth");
        Dimension height = windowOutputDimension(
                outputShape.dimensions().get(3), window.kernelHeight(), window.paddingHeight(),
                window.strideHeight(), window.dilationHeight(), window.ceilMode(),
                "fold3d", "height");
        Dimension width = windowOutputDimension(
                outputShape.dimensions().get(4), window.kernelWidth(), window.paddingWidth(),
                window.strideWidth(), window.dilationWidth(), window.ceilMode(),
                "fold3d", "width");
        Dimension expectedWindowCount = DimensionExpressions.multiply(
                DimensionExpressions.multiply(depth, height), width);
        Dimension actualWindowCount = inputShape.dimensions().get(2);
        if (!actualWindowCount.equals(expectedWindowCount)) {
            throw new IllegalArgumentException(
                    "fold3d column count " + dimensionDiagnostic(actualWindowCount)
                            + " does not match output shape and window geometry: expected="
                            + dimensionDiagnostic(expectedWindowCount));
        }
    }

    /**
     * Calculates one static or symbolic floor- or ceil-mode spatial window count.
     *
     * @param inputSize non-null static or symbolic input spatial extent
     * @param kernel positive count of kernel samples
     * @param padding non-negative symmetric padding on each side
     * @param stride positive distance between window starts
     * @param dilation positive spacing between kernel samples
     * @param ceilMode true for ceil quotient rounding, false for floor rounding
     * @param operation non-null constant operation name for failures
     * @param dimension non-null spatial dimension name for failures
     * @return non-null exact static or symbolic window-position count
     * @throws IllegalArgumentException if static geometry proves the effective kernel does not fit
     * @throws ArithmeticException if checked geometry arithmetic overflows
     */
    private static Dimension windowOutputDimension(
            Dimension inputSize,
            long kernel,
            long padding,
            long stride,
            long dilation,
            boolean ceilMode,
            String operation,
            String dimension) {
        if (inputSize instanceof StaticDimension staticDimension) {
            return new StaticDimension(windowOutputSize(
                    staticDimension.size(), kernel, padding, stride, dilation, ceilMode,
                    operation, dimension));
        }
        long effectiveKernel = Math.addExact(
                Math.multiplyExact(dilation, Math.subtractExact(kernel, 1L)), 1L);
        long offset = Math.subtractExact(Math.multiplyExact(2L, padding), effectiveKernel);
        Dimension numerator = DimensionExpressions.addConstant(inputSize, offset);
        Dimension quotient = ceilMode
                ? DimensionExpressions.ceilingDivide(numerator, stride)
                : DimensionExpressions.floorDivide(numerator, stride);
        return DimensionExpressions.addConstant(quotient, 1L);
    }

    private static String dimensionDiagnostic(Dimension dimension) {
        if (dimension instanceof StaticDimension staticDimension) {
            return Long.toString(staticDimension.size());
        }
        return dimension.toString();
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
