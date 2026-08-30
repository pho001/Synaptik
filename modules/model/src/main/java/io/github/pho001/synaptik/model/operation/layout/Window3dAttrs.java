package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;

/**
 * Immutable symmetric depth-height-width window geometry for NCDHW transforms.
 *
 * <p>NCDHW orders axes as batch, channel, depth, height, and width. Kernel samples use the
 * positive dilation spacing, window starts use the positive stride spacing, and each padding
 * value applies to both sides of its spatial dimension. Floor and literal ceil geometry are
 * interpreted by public Tensor construction; this value performs no input-aware Shape, type,
 * fit, product, or arithmetic validation and owns no Tensor or execution state.</p>
 *
 * @param kernelDepth positive kernel sample count along depth
 * @param kernelHeight positive kernel sample count along height
 * @param kernelWidth positive kernel sample count along width
 * @param strideDepth positive distance between window starts along depth
 * @param strideHeight positive distance between window starts along height
 * @param strideWidth positive distance between window starts along width
 * @param paddingDepth non-negative symmetric padding on each depth side
 * @param paddingHeight non-negative symmetric padding on each height side
 * @param paddingWidth non-negative symmetric padding on each width side
 * @param dilationDepth positive spacing between kernel samples along depth
 * @param dilationHeight positive spacing between kernel samples along height
 * @param dilationWidth positive spacing between kernel samples along width
 * @param ceilMode {@code true} for literal ceiling output geometry; {@code false} for floor
 */
public record Window3dAttrs(
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
     * Creates validated immutable three-dimensional window geometry.
     *
     * @param kernelDepth positive kernel sample count along depth
     * @param kernelHeight positive kernel sample count along height
     * @param kernelWidth positive kernel sample count along width
     * @param strideDepth positive distance between window starts along depth
     * @param strideHeight positive distance between window starts along height
     * @param strideWidth positive distance between window starts along width
     * @param paddingDepth non-negative symmetric padding on each depth side
     * @param paddingHeight non-negative symmetric padding on each height side
     * @param paddingWidth non-negative symmetric padding on each width side
     * @param dilationDepth positive spacing between kernel samples along depth
     * @param dilationHeight positive spacing between kernel samples along height
     * @param dilationWidth positive spacing between kernel samples along width
     * @param ceilMode exact literal-ceil selection to retain
     * @throws IllegalArgumentException if the first invalid component in declaration order is a
     *     non-positive kernel, stride, or dilation, or a negative padding component
     */
    public Window3dAttrs {
        if (kernelDepth <= 0) {
            throw new IllegalArgumentException("kernelDepth must be positive: " + kernelDepth);
        }
        if (kernelHeight <= 0) {
            throw new IllegalArgumentException("kernelHeight must be positive: " + kernelHeight);
        }
        if (kernelWidth <= 0) {
            throw new IllegalArgumentException("kernelWidth must be positive: " + kernelWidth);
        }
        if (strideDepth <= 0) {
            throw new IllegalArgumentException("strideDepth must be positive: " + strideDepth);
        }
        if (strideHeight <= 0) {
            throw new IllegalArgumentException("strideHeight must be positive: " + strideHeight);
        }
        if (strideWidth <= 0) {
            throw new IllegalArgumentException("strideWidth must be positive: " + strideWidth);
        }
        if (paddingDepth < 0) {
            throw new IllegalArgumentException(
                    "paddingDepth must be non-negative: " + paddingDepth);
        }
        if (paddingHeight < 0) {
            throw new IllegalArgumentException(
                    "paddingHeight must be non-negative: " + paddingHeight);
        }
        if (paddingWidth < 0) {
            throw new IllegalArgumentException(
                    "paddingWidth must be non-negative: " + paddingWidth);
        }
        if (dilationDepth <= 0) {
            throw new IllegalArgumentException(
                    "dilationDepth must be positive: " + dilationDepth);
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
}
