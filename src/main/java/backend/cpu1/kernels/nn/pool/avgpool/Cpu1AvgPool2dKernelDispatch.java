package backend.cpu1.kernels.nn.pool.avgpool;

/**
 * Resolves prepared cpu1 AVG_POOL2D kernel ids.
 */
public final class Cpu1AvgPool2dKernelDispatch {
    private Cpu1AvgPool2dKernelDispatch() {
    }

    public static Cpu1AvgPool2dKernel kernelFor(Cpu1AvgPool2dKernelId kernelId) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        return switch (kernelId) {
            case AVG_POOL2D_F32_ARRAY_DENSE_SCALAR -> Cpu1AvgPool2dLoops::runF32DenseArray;
            case AVG_POOL2D_F64_ARRAY_DENSE_SCALAR -> Cpu1AvgPool2dLoops::runF64DenseArray;
            case AVG_POOL2D_BF16_ARRAY_DENSE_SCALAR -> Cpu1AvgPool2dLoops::runBf16DenseArray;
            case AVG_POOL2D_F32_SEGMENT_DENSE_SCALAR -> Cpu1AvgPool2dLoops::runF32DenseSegment;
            case AVG_POOL2D_F64_SEGMENT_DENSE_SCALAR -> Cpu1AvgPool2dLoops::runF64DenseSegment;
            case AVG_POOL2D_BF16_SEGMENT_DENSE_SCALAR -> Cpu1AvgPool2dLoops::runBf16DenseSegment;
        };
    }
}
