package io.github.pho001.synaptik.model.operation.pooling;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Immutable NCW width geometry for maximum pooling composed through {@link MaxPool2dAttrs}.
 *
 * <p>The kernel counts width samples, the stride selects the step between output windows,
 * symmetric padding counts excluded positions on each side, and dilation selects spacing between
 * kernel samples. {@code ceilMode} selects literal ceiling division for the padded width grid; it
 * does not remove a terminal window that lies entirely in trailing padding.</p>
 *
 * <p>This value is a public composition parameter, not the attributes of a Pool1d operation kind.
 * Expression construction translates it to a fresh {@code MaxPool2dAttrs} value with singleton-
 * height geometry. It contains no tensor, shape, data type, layout, storage, gradient, compiler,
 * backend, or execution state. Record equality and hashing use all five components.</p>
 *
 * @param kernelWidth positive number of width kernel samples
 * @param strideWidth positive width step between output windows
 * @param paddingWidth non-negative excluded padding on both width sides
 * @param dilationWidth positive spacing between width kernel samples
 * @param ceilMode {@code true} for the literal ceiling grid; {@code false} for the floor grid
 */
public record MaxPool1dAttrs(
        long kernelWidth,
        long strideWidth,
        long paddingWidth,
        long dilationWidth,
        boolean ceilMode) implements OperationAttrs {
    /**
     * Creates validated maximum-pooling width geometry, checking components in declaration order.
     *
     * @param kernelWidth positive width kernel sample count
     * @param strideWidth positive width output stride
     * @param paddingWidth non-negative symmetric width padding per side
     * @param dilationWidth positive width kernel dilation
     * @param ceilMode whether the output width uses literal ceiling division
     * @throws IllegalArgumentException if kernel, stride, or dilation is not positive, or padding
     *     is negative
     */
    public MaxPool1dAttrs {
        requirePositive(kernelWidth, "kernelWidth");
        requirePositive(strideWidth, "strideWidth");
        requireNonNegative(paddingWidth, "paddingWidth");
        requirePositive(dilationWidth, "dilationWidth");
    }

    /**
     * Requires one strictly positive geometry component.
     *
     * @param value component value to validate
     * @param name non-null component name used in a failure message
     * @throws IllegalArgumentException if {@code value} is zero or negative
     */
    private static void requirePositive(long value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive: " + value);
        }
    }

    /**
     * Requires one non-negative geometry component.
     *
     * @param value component value to validate
     * @param name non-null component name used in a failure message
     * @throws IllegalArgumentException if {@code value} is negative
     */
    private static void requireNonNegative(long value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative: " + value);
        }
    }
}
