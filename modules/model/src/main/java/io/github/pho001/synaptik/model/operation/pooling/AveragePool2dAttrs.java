package io.github.pho001.synaptik.model.operation.pooling;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Immutable intrinsic geometry for two-dimensional NCHW average pooling.
 *
 * <p>Kernel components count logical divisor positions. Stride components select the step between
 * output windows, symmetric padding components count conceptual positive-zero positions on each
 * side, and dilation components select spacing between samples. {@code ceilMode} selects literal
 * ceiling division for the padded grid and does not remove a terminal all-padding window.</p>
 *
 * <p>The divisor is always {@code kernelHeight * kernelWidth}; padding counts in it and no divisor
 * override is configurable. This value contains no tensor, shape, data type, layout, storage,
 * gradient, compiler, backend, or execution state. Equality and hashing use all components.</p>
 *
 * @param kernelHeight positive number of vertical kernel positions
 * @param kernelWidth positive number of horizontal kernel positions
 * @param strideHeight positive vertical step between output windows
 * @param strideWidth positive horizontal step between output windows
 * @param paddingHeight non-negative symmetric vertical padding per side
 * @param paddingWidth non-negative symmetric horizontal padding per side
 * @param dilationHeight positive vertical spacing between kernel positions
 * @param dilationWidth positive horizontal spacing between kernel positions
 * @param ceilMode {@code true} for the literal ceiling grid; {@code false} for the floor grid
 */
public record AveragePool2dAttrs(
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
     * Creates validated average-pooling geometry, checking components in declaration order.
     *
     * @param kernelHeight positive vertical kernel-position count
     * @param kernelWidth positive horizontal kernel-position count
     * @param strideHeight positive vertical output stride
     * @param strideWidth positive horizontal output stride
     * @param paddingHeight non-negative symmetric vertical padding per side
     * @param paddingWidth non-negative symmetric horizontal padding per side
     * @param dilationHeight positive vertical kernel dilation
     * @param dilationWidth positive horizontal kernel dilation
     * @param ceilMode whether output spatial extents use literal ceiling division
     * @throws IllegalArgumentException if a kernel, stride, or dilation component is not positive,
     *     or a padding component is negative
     */
    public AveragePool2dAttrs {
        requirePositive(kernelHeight, "kernelHeight");
        requirePositive(kernelWidth, "kernelWidth");
        requirePositive(strideHeight, "strideHeight");
        requirePositive(strideWidth, "strideWidth");
        requireNonNegative(paddingHeight, "paddingHeight");
        requireNonNegative(paddingWidth, "paddingWidth");
        requirePositive(dilationHeight, "dilationHeight");
        requirePositive(dilationWidth, "dilationWidth");
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
