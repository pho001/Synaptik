package tensor.options;

/**
 * Immutable configuration for NCHW 2-D pooling.
 *
 * @param kernelH vertical pooling window size; must be positive
 * @param kernelW horizontal pooling window size; must be positive
 * @param strideH vertical step between windows; must be positive
 * @param strideW horizontal step between windows; must be positive
 * @param padH zero-padding applied to both top and bottom; must be non-negative
 * @param padW zero-padding applied to both left and right; must be non-negative
 * @param countIncludePad whether average pooling includes padded cells in the divisor
 * @param ceilMode whether output spatial dimensions use ceil instead of floor
 */
public record Pool2dOptions(
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        boolean countIncludePad,
        boolean ceilMode
) {
    public Pool2dOptions(int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, boolean countIncludePad) {
        this(kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad, false);
    }

    /**
     * Validates pooling option invariants.
     *
     * @throws IllegalArgumentException if kernel or stride values are
     *                                  non-positive, or if padding is negative
     */
    public Pool2dOptions {
        if (kernelH <= 0 || kernelW <= 0) {
            throw new IllegalArgumentException("Pool2d kernel must be positive.");
        }
        if (strideH <= 0 || strideW <= 0) {
            throw new IllegalArgumentException("Pool2d stride must be positive.");
        }
        if (padH < 0 || padW < 0) {
            throw new IllegalArgumentException("Pool2d padding cannot be negative.");
        }
    }

    /**
     * Creates options whose stride equals the rectangular kernel size.
     *
     * @param kernelH vertical kernel size; must be positive
     * @param kernelW horizontal kernel size; must be positive
     * @return pooling options with no padding and {@code countIncludePad=false}
     */
    public static Pool2dOptions of(int kernelH, int kernelW) {
        return new Pool2dOptions(kernelH, kernelW, kernelH, kernelW, 0, 0, false, false);
    }

    /**
     * Creates square-kernel pooling options whose stride equals the kernel.
     *
     * @param kernel square kernel size; must be positive
     * @return pooling options with no padding and {@code countIncludePad=false}
     */
    public static Pool2dOptions square(int kernel) {
        return of(kernel, kernel);
    }

    /**
     * Returns a copy with updated stride.
     *
     * @param strideH vertical stride; must be positive
     * @param strideW horizontal stride; must be positive
     * @return new immutable options instance
     */
    public Pool2dOptions withStride(int strideH, int strideW) {
        return new Pool2dOptions(kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad, ceilMode);
    }

    /**
     * Returns a copy with updated symmetric padding.
     *
     * @param padH vertical padding; must be non-negative
     * @param padW horizontal padding; must be non-negative
     * @return new immutable options instance
     */
    public Pool2dOptions withPadding(int padH, int padW) {
        return new Pool2dOptions(kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad, ceilMode);
    }

    /**
     * Returns a copy with updated average-pooling divisor behavior.
     *
     * @param countIncludePad true to include padded cells in average pooling counts
     * @return new immutable options instance
     */
    public Pool2dOptions withCountIncludePad(boolean countIncludePad) {
        return new Pool2dOptions(kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad, ceilMode);
    }

    /**
     * Returns a copy with updated ceil-mode output shape behavior.
     *
     * @param ceilMode true to use ceil output spatial dimensions
     * @return new immutable options instance
     */
    public Pool2dOptions withCeilMode(boolean ceilMode) {
        return new Pool2dOptions(kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad, ceilMode);
    }
}
