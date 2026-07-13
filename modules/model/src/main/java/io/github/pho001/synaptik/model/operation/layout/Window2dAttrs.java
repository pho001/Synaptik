package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Carries reusable symmetric two-dimensional sliding-window geometry for NCHW transforms.
 *
 * <p>NCHW orders axes as batch, channel, height, and width. Kernel values are sampled at the
 * given positive dilation spacing, consecutive windows begin at the positive stride spacing, and
 * each padding value applies symmetrically to both sides of its spatial dimension. Positions in
 * that padding sampled by the direct {@link WindowTransformKind#UNFOLD2D} pairing are conceptual
 * positive zeros. {@link Unfold2dAttrs} instead supplies an exact typed padding value while
 * retaining this same geometry by reference.</p>
 *
 * <p>For either spatial dimension, current public Tensor construction calculates
 * {@code effectiveKernel = dilation * (kernel - 1) + 1} and
 * {@code numerator = input + 2 * padding - effectiveKernel}. Floor mode uses
 * {@code floor(numerator / stride) + 1}; ceil mode uses
 * {@code ceil(numerator / stride) + 1}. Effective kernel is the span covered after dilation.
 * Static inputs use checked {@code long} arithmetic and local fit checks; unresolved inputs retain
 * exact symbolic formulas and defer their non-negativity obligation. Public Tensor expression
 * construction owns those checks and result descriptors; this record performs no multiplication,
 * addition, division, or Shape validation.</p>
 *
 * <p>The same exact value contract parameterizes {@link WindowTransformKind#UNFOLD2D} directly,
 * is retained by {@link Unfold2dAttrs} for explicit typed padding, and is nested by
 * {@link Fold2dAttrs} for {@link WindowTransformKind#FOLD2D}. It contains no
 * input or output Shape, DataType, Tensor, layout, storage, provenance, gradient, compiler,
 * backend, or execution state.</p>
 *
 * @param kernelHeight the positive kernel extent along height, in logical positions
 * @param kernelWidth the positive kernel extent along width, in logical positions
 * @param strideHeight the positive distance between consecutive window starts along height
 * @param strideWidth the positive distance between consecutive window starts along width
 * @param paddingHeight the non-negative symmetric padding on each height side
 * @param paddingWidth the non-negative symmetric padding on each width side
 * @param dilationHeight the positive spacing between kernel samples along height
 * @param dilationWidth the positive spacing between kernel samples along width
 * @param ceilMode {@code true} to use ceil output-size rounding; {@code false} for floor rounding
 */
public record Window2dAttrs(
        long kernelHeight,
        long kernelWidth,
        long strideHeight,
        long strideWidth,
        long paddingHeight,
        long paddingWidth,
        long dilationHeight,
        long dilationWidth,
        boolean ceilMode) implements OperationAttrs {
    /**
     * Creates immutable normalized two-dimensional window geometry.
     *
     * <p>Validation follows declaration order: positive kernel dimensions, positive stride
     * dimensions, non-negative symmetric padding dimensions, then positive dilation dimensions.
     * Ceil mode is retained unchanged. Construction performs no geometry arithmetic or Shape
     * validation, so every valid component may be {@link Long#MAX_VALUE}.</p>
     *
     * @param kernelHeight the kernel height; must be positive
     * @param kernelWidth the kernel width; must be positive
     * @param strideHeight the height stride; must be positive
     * @param strideWidth the width stride; must be positive
     * @param paddingHeight the symmetric height padding per side; must be non-negative
     * @param paddingWidth the symmetric width padding per side; must be non-negative
     * @param dilationHeight the height dilation; must be positive
     * @param dilationWidth the width dilation; must be positive
     * @param ceilMode the output-size rounding mode to retain unchanged
     * @throws IllegalArgumentException if a kernel, stride, or dilation component is non-positive,
     *     or a padding component is negative; the first invalid component in declaration order is
     *     reported as {@code <component> must be positive: <value>} or
     *     {@code <component> must be non-negative: <value>}
     */
    public Window2dAttrs {
        if (kernelHeight <= 0) {
            throw new IllegalArgumentException(
                    "kernelHeight must be positive: " + kernelHeight);
        }
        if (kernelWidth <= 0) {
            throw new IllegalArgumentException("kernelWidth must be positive: " + kernelWidth);
        }
        if (strideHeight <= 0) {
            throw new IllegalArgumentException(
                    "strideHeight must be positive: " + strideHeight);
        }
        if (strideWidth <= 0) {
            throw new IllegalArgumentException("strideWidth must be positive: " + strideWidth);
        }
        if (paddingHeight < 0) {
            throw new IllegalArgumentException(
                    "paddingHeight must be non-negative: " + paddingHeight);
        }
        if (paddingWidth < 0) {
            throw new IllegalArgumentException(
                    "paddingWidth must be non-negative: " + paddingWidth);
        }
        if (dilationHeight <= 0) {
            throw new IllegalArgumentException(
                    "dilationHeight must be positive: " + dilationHeight);
        }
        if (dilationWidth <= 0) {
            throw new IllegalArgumentException(
                    "dilationWidth must be positive: " + dilationWidth);
        }
    }

    /**
     * Returns the number of logical kernel samples along height.
     *
     * @return the exact positive kernel height supplied at construction
     */
    @Override
    public long kernelHeight() {
        return kernelHeight;
    }

    /**
     * Returns the number of logical kernel samples along width.
     *
     * @return the exact positive kernel width supplied at construction
     */
    @Override
    public long kernelWidth() {
        return kernelWidth;
    }

    /**
     * Returns the distance between consecutive window starts along height.
     *
     * @return the exact positive height stride supplied at construction
     */
    @Override
    public long strideHeight() {
        return strideHeight;
    }

    /**
     * Returns the distance between consecutive window starts along width.
     *
     * @return the exact positive width stride supplied at construction
     */
    @Override
    public long strideWidth() {
        return strideWidth;
    }

    /**
     * Returns the symmetric padding width applied to each height side.
     *
     * @return the exact non-negative symmetric height padding supplied at construction
     */
    @Override
    public long paddingHeight() {
        return paddingHeight;
    }

    /**
     * Returns the symmetric padding width applied to each width side.
     *
     * @return the exact non-negative symmetric width padding supplied at construction
     */
    @Override
    public long paddingWidth() {
        return paddingWidth;
    }

    /**
     * Returns the spacing between adjacent kernel samples along height.
     *
     * @return the exact positive height dilation supplied at construction
     */
    @Override
    public long dilationHeight() {
        return dilationHeight;
    }

    /**
     * Returns the spacing between adjacent kernel samples along width.
     *
     * @return the exact positive width dilation supplied at construction
     */
    @Override
    public long dilationWidth() {
        return dilationWidth;
    }

    /**
     * Reports which rounding mode later output-size calculation uses.
     *
     * @return the exact ceil-mode flag supplied at construction; {@code true} selects ceil mode
     *     and {@code false} selects floor mode
     */
    @Override
    public boolean ceilMode() {
        return ceilMode;
    }
}
