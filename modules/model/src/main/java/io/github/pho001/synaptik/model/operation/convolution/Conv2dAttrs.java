package io.github.pho001.synaptik.model.operation.convolution;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Immutable intrinsic geometry for grouped two-dimensional NCHW convolution.
 *
 * <p>Stride and dilation components are positive element counts. Padding is the non-negative
 * number of conceptual positive-zero positions added symmetrically on each side of an axis, and
 * groups partitions input and output channels into contiguous independent channel sets. Kernel
 * extents and optional bias presence belong to an operation occurrence rather than this value.</p>
 *
 * <p>This value contains no tensor, shape, data type, layout, storage, gradient, compiler,
 * backend, or execution state. Record equality and hashing use all seven components.</p>
 *
 * @param strideHeight positive vertical step between output positions
 * @param strideWidth positive horizontal step between output positions
 * @param paddingHeight non-negative conceptual padding on both top and bottom
 * @param paddingWidth non-negative conceptual padding on both left and right
 * @param dilationHeight positive vertical spacing between kernel positions
 * @param dilationWidth positive horizontal spacing between kernel positions
 * @param groups positive number of independent contiguous channel groups
 */
public record Conv2dAttrs(
        long strideHeight,
        long strideWidth,
        long paddingHeight,
        long paddingWidth,
        long dilationHeight,
        long dilationWidth,
        long groups) implements OperationAttrs {
    /**
     * Creates validated convolution geometry, checking components in declaration order.
     *
     * @param strideHeight positive vertical output stride
     * @param strideWidth positive horizontal output stride
     * @param paddingHeight non-negative symmetric vertical padding per side
     * @param paddingWidth non-negative symmetric horizontal padding per side
     * @param dilationHeight positive vertical kernel dilation
     * @param dilationWidth positive horizontal kernel dilation
     * @param groups positive channel-group count
     * @throws IllegalArgumentException if a stride, dilation, or group count is not positive, or
     *     a padding component is negative
     */
    public Conv2dAttrs {
        requirePositive(strideHeight, "strideHeight");
        requirePositive(strideWidth, "strideWidth");
        requireNonNegative(paddingHeight, "paddingHeight");
        requireNonNegative(paddingWidth, "paddingWidth");
        requirePositive(dilationHeight, "dilationHeight");
        requirePositive(dilationWidth, "dilationWidth");
        requirePositive(groups, "groups");
    }

    /**
     * Returns unit stride and dilation, zero symmetric padding, and one channel group.
     *
     * @return a new immutable attributes value {@code (1, 1, 0, 0, 1, 1, 1)}
     */
    public static Conv2dAttrs defaults() {
        return new Conv2dAttrs(1, 1, 0, 0, 1, 1, 1);
    }

    private static void requirePositive(long value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive: " + value);
        }
    }

    private static void requireNonNegative(long value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative: " + value);
        }
    }
}
