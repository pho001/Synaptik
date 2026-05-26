package tensor.options;

/**
 * Immutable NCHW 2-D sliding-window geometry shared by unfold2d/fold2d.
 *
 * @param kernelH vertical window size; must be positive
 * @param kernelW horizontal window size; must be positive
 * @param strideH vertical step between windows; must be positive
 * @param strideW horizontal step between windows; must be positive
 * @param padH symmetric vertical zero-padding; must be non-negative
 * @param padW symmetric horizontal zero-padding; must be non-negative
 * @param dilationH vertical spacing between window elements; must be positive
 * @param dilationW horizontal spacing between window elements; must be positive
 * @param ceilMode whether output spatial dimensions use ceil instead of floor
 */
public record Window2dOptions(
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW,
        boolean ceilMode
) {
    public Window2dOptions(
            int kernelH,
            int kernelW,
            int strideH,
            int strideW,
            int padH,
            int padW,
            int dilationH,
            int dilationW
    ) {
        this(kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, false);
    }

    public Window2dOptions {
        if (kernelH <= 0 || kernelW <= 0) {
            throw new IllegalArgumentException("Window2d kernel must be positive.");
        }
        if (strideH <= 0 || strideW <= 0) {
            throw new IllegalArgumentException("Window2d stride must be positive.");
        }
        if (padH < 0 || padW < 0) {
            throw new IllegalArgumentException("Window2d padding cannot be negative.");
        }
        if (dilationH <= 0 || dilationW <= 0) {
            throw new IllegalArgumentException("Window2d dilation must be positive.");
        }
    }

    public static Window2dOptions of(int kernelH, int kernelW) {
        return new Window2dOptions(kernelH, kernelW, 1, 1, 0, 0, 1, 1);
    }

    public Window2dOptions withStride(int strideH, int strideW) {
        return new Window2dOptions(kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, ceilMode);
    }

    public Window2dOptions withPadding(int padH, int padW) {
        return new Window2dOptions(kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, ceilMode);
    }

    public Window2dOptions withDilation(int dilationH, int dilationW) {
        return new Window2dOptions(kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, ceilMode);
    }

    public Window2dOptions withCeilMode(boolean ceilMode) {
        return new Window2dOptions(kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, ceilMode);
    }
}
