package tensor.options;

public record Conv2dOptions(
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW,
        int groups
) {
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

    public static Conv2dOptions defaults() {
        return new Conv2dOptions(1, 1, 0, 0, 1, 1, 1);
    }

    public Conv2dOptions withStride(int strideH, int strideW) {
        return new Conv2dOptions(strideH, strideW, padH, padW, dilationH, dilationW, groups);
    }

    public Conv2dOptions withPadding(int padH, int padW) {
        return new Conv2dOptions(strideH, strideW, padH, padW, dilationH, dilationW, groups);
    }

    public Conv2dOptions withDilation(int dilationH, int dilationW) {
        return new Conv2dOptions(strideH, strideW, padH, padW, dilationH, dilationW, groups);
    }

    public Conv2dOptions withGroups(int groups) {
        return new Conv2dOptions(strideH, strideW, padH, padW, dilationH, dilationW, groups);
    }
}
