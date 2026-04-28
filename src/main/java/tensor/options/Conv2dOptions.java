package tensor.options;

/**
 * Immutable configuration for NCHW 2-D convolution.
 *
 * @param strideH vertical step between convolution windows; must be positive
 * @param strideW horizontal step between convolution windows; must be positive
 * @param padH zero-padding applied to both top and bottom; must be non-negative
 * @param padW zero-padding applied to both left and right; must be non-negative
 * @param dilationH vertical spacing between kernel elements; must be positive
 * @param dilationW horizontal spacing between kernel elements; must be positive
 * @param groups number of independent channel groups; must be positive and must
 *               divide compatible input/output channel counts at operation time
 */
public record Conv2dOptions(
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW,
        int groups
) {
    /**
     * Validates convolution option invariants.
     *
     * @throws IllegalArgumentException if stride, dilation, or groups are not
     *                                  positive, or if padding is negative
     */
    public Conv2dOptions {
        if (strideH <= 0 || strideW <= 0) {
            throw new IllegalArgumentException("Conv2d stride must be positive.");
        }
        if (padH < 0 || padW < 0) {
            throw new IllegalArgumentException("Conv2d padding cannot be negative.");
        }
        if (dilationH <= 0 || dilationW <= 0) {
            throw new IllegalArgumentException("Conv2d dilation must be positive.");
        }
        if (groups <= 0) {
            throw new IllegalArgumentException("Conv2d groups must be positive.");
        }
    }

    /**
     * Returns standard dense convolution options.
     *
     * @return stride 1, no padding, dilation 1, one group
     */
    public static Conv2dOptions defaults() {
        return new Conv2dOptions(1, 1, 0, 0, 1, 1, 1);
    }

    /**
     * Returns a copy with updated stride.
     *
     * @param strideH vertical stride; must be positive
     * @param strideW horizontal stride; must be positive
     * @return new immutable options instance
     * @throws IllegalArgumentException if either stride is non-positive
     */
    public Conv2dOptions withStride(int strideH, int strideW) {
        return new Conv2dOptions(strideH, strideW, padH, padW, dilationH, dilationW, groups);
    }

    /**
     * Returns a copy with updated symmetric padding.
     *
     * @param padH vertical padding; must be non-negative
     * @param padW horizontal padding; must be non-negative
     * @return new immutable options instance
     * @throws IllegalArgumentException if either padding value is negative
     */
    public Conv2dOptions withPadding(int padH, int padW) {
        return new Conv2dOptions(strideH, strideW, padH, padW, dilationH, dilationW, groups);
    }

    /**
     * Returns a copy with updated dilation.
     *
     * @param dilationH vertical dilation; must be positive
     * @param dilationW horizontal dilation; must be positive
     * @return new immutable options instance
     * @throws IllegalArgumentException if either dilation value is non-positive
     */
    public Conv2dOptions withDilation(int dilationH, int dilationW) {
        return new Conv2dOptions(strideH, strideW, padH, padW, dilationH, dilationW, groups);
    }

    /**
     * Returns a copy with updated group count.
     *
     * @param groups channel group count; must be positive
     * @return new immutable options instance
     * @throws IllegalArgumentException if {@code groups} is non-positive
     */
    public Conv2dOptions withGroups(int groups) {
        return new Conv2dOptions(strideH, strideW, padH, padW, dilationH, dilationW, groups);
    }
}
