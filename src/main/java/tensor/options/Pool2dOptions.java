package tensor.options;

public record Pool2dOptions(
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        boolean countIncludePad
) {
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

    public static Pool2dOptions of(int kernelH, int kernelW) {
        return new Pool2dOptions(kernelH, kernelW, kernelH, kernelW, 0, 0, false);
    }

    public static Pool2dOptions square(int kernel) {
        return of(kernel, kernel);
    }

    public Pool2dOptions withStride(int strideH, int strideW) {
        return new Pool2dOptions(kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad);
    }

    public Pool2dOptions withPadding(int padH, int padW) {
        return new Pool2dOptions(kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad);
    }

    public Pool2dOptions withCountIncludePad(boolean countIncludePad) {
        return new Pool2dOptions(kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad);
    }
}
