package backend.cpu1.kernels.nn.conv.conv2d;

/**
 * Resolves prepared cpu1 CONV2D kernel ids.
 */
public final class Cpu1Conv2dKernelDispatch {
    private Cpu1Conv2dKernelDispatch() {
    }

    public static Cpu1Conv2dKernel kernelFor(Cpu1Conv2dKernelId kernelId) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        return switch (kernelId) {
            case CONV2D_F32_ARRAY_DENSE_SCALAR -> Cpu1Conv2dLoops::runF32DenseArray;
            case CONV2D_F64_ARRAY_DENSE_SCALAR -> Cpu1Conv2dLoops::runF64DenseArray;
            case CONV2D_BF16_ARRAY_DENSE_SCALAR -> Cpu1Conv2dLoops::runBf16DenseArray;
            case CONV2D_F32_SEGMENT_DENSE_SCALAR -> Cpu1Conv2dLoops::runF32DenseSegment;
            case CONV2D_F64_SEGMENT_DENSE_SCALAR -> Cpu1Conv2dLoops::runF64DenseSegment;
            case CONV2D_BF16_SEGMENT_DENSE_SCALAR -> Cpu1Conv2dLoops::runBf16DenseSegment;
        };
    }
}
