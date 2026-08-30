package io.github.pho001.synaptik.model.operation.pooling;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Immutable intrinsic geometry for three-dimensional NCDHW maximum pooling.
 *
 * <p>Kernel components count samples. Stride components select the step between output windows,
 * symmetric padding components count excluded positions on each side, and dilation components
 * select spacing between kernel samples. {@code ceilMode} selects literal ceiling division for
 * each padded spatial grid; it does not remove a terminal all-padding window.</p>
 *
 * <p>This value contains no tensor, shape, data type, layout, storage, gradient, compiler,
 * backend, or execution state. Record equality and hashing use all thirteen components.</p>
 *
 * @param kernelDepth positive number of depth kernel samples
 * @param kernelHeight positive number of vertical kernel samples
 * @param kernelWidth positive number of horizontal kernel samples
 * @param strideDepth positive depth step between output windows
 * @param strideHeight positive vertical step between output windows
 * @param strideWidth positive horizontal step between output windows
 * @param paddingDepth non-negative excluded padding on both front and back
 * @param paddingHeight non-negative excluded padding on both top and bottom
 * @param paddingWidth non-negative excluded padding on both left and right
 * @param dilationDepth positive depth spacing between kernel samples
 * @param dilationHeight positive vertical spacing between kernel samples
 * @param dilationWidth positive horizontal spacing between kernel samples
 * @param ceilMode {@code true} for literal ceiling grids; {@code false} for floor grids
 */
public record MaxPool3dAttrs(
        long kernelDepth,
        long kernelHeight,
        long kernelWidth,
        long strideDepth,
        long strideHeight,
        long strideWidth,
        long paddingDepth,
        long paddingHeight,
        long paddingWidth,
        long dilationDepth,
        long dilationHeight,
        long dilationWidth,
        boolean ceilMode) implements OperationAttrs {
    /**
     * Creates validated maximum-pooling geometry, checking components in declaration order.
     *
     * @param kernelDepth positive depth kernel sample count
     * @param kernelHeight positive vertical kernel sample count
     * @param kernelWidth positive horizontal kernel sample count
     * @param strideDepth positive depth output stride
     * @param strideHeight positive vertical output stride
     * @param strideWidth positive horizontal output stride
     * @param paddingDepth non-negative symmetric depth padding per side
     * @param paddingHeight non-negative symmetric vertical padding per side
     * @param paddingWidth non-negative symmetric horizontal padding per side
     * @param dilationDepth positive depth kernel dilation
     * @param dilationHeight positive vertical kernel dilation
     * @param dilationWidth positive horizontal kernel dilation
     * @param ceilMode whether output spatial extents use literal ceiling division
     * @throws IllegalArgumentException if a kernel, stride, or dilation component is not positive,
     *     or a padding component is negative
     */
    public MaxPool3dAttrs {
        requirePositive(kernelDepth, "kernelDepth");
        requirePositive(kernelHeight, "kernelHeight");
        requirePositive(kernelWidth, "kernelWidth");
        requirePositive(strideDepth, "strideDepth");
        requirePositive(strideHeight, "strideHeight");
        requirePositive(strideWidth, "strideWidth");
        requireNonNegative(paddingDepth, "paddingDepth");
        requireNonNegative(paddingHeight, "paddingHeight");
        requireNonNegative(paddingWidth, "paddingWidth");
        requirePositive(dilationDepth, "dilationDepth");
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
