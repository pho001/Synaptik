package io.github.pho001.synaptik.model.operation.convolution;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Immutable geometry for the NCW one-dimensional convolution convenience.
 *
 * <p>Stride and dilation are positive width-element counts. Padding is the non-negative number
 * of conceptual positive-zero positions on each side of the width axis, and groups partitions
 * input and output channels into contiguous independent sets. Construction translates this
 * value to {@link Conv2dAttrs}; no operation occurrence retains it.</p>
 *
 * @param stride positive step between output width positions
 * @param padding non-negative conceptual width padding on both sides
 * @param dilation positive spacing between kernel width positions
 * @param groups positive number of independent contiguous channel groups
 */
public record Conv1dAttrs(long stride, long padding, long dilation, long groups)
        implements OperationAttrs {
    /**
     * Creates validated rank-one convolution geometry in declaration order.
     *
     * @param stride positive output-width stride
     * @param padding non-negative symmetric width padding per side
     * @param dilation positive kernel-width dilation
     * @param groups positive channel-group count
     * @throws IllegalArgumentException if stride, dilation, or groups is not positive, or padding
     *     is negative
     */
    public Conv1dAttrs {
        requirePositive(stride, "stride");
        requireNonNegative(padding, "padding");
        requirePositive(dilation, "dilation");
        requirePositive(groups, "groups");
    }

    /**
     * Returns unit stride and dilation, zero symmetric padding, and one channel group.
     *
     * @return a new immutable attributes value {@code (1, 0, 1, 1)}
     */
    public static Conv1dAttrs defaults() {
        return new Conv1dAttrs(1, 0, 1, 1);
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
