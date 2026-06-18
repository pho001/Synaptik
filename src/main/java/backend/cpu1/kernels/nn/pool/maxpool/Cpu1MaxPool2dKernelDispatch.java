package backend.cpu1.kernels.nn.pool.maxpool;

/**
 * Resolves prepared cpu1 MAX_POOL2D kernel ids.
 */
public final class Cpu1MaxPool2dKernelDispatch {
    private Cpu1MaxPool2dKernelDispatch() {
    }

    public static Cpu1MaxPool2dKernel kernelFor(Cpu1MaxPool2dKernelId kernelId) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        return switch (kernelId) {
            case MAX_POOL2D_F32_ARRAY_DENSE_SCALAR -> Cpu1MaxPool2dLoops::runF32DenseArray;
            case MAX_POOL2D_F64_ARRAY_DENSE_SCALAR -> Cpu1MaxPool2dLoops::runF64DenseArray;
            case MAX_POOL2D_BF16_ARRAY_DENSE_SCALAR -> Cpu1MaxPool2dLoops::runBf16DenseArray;
            case MAX_POOL2D_F32_SEGMENT_DENSE_SCALAR -> Cpu1MaxPool2dLoops::runF32DenseSegment;
            case MAX_POOL2D_F64_SEGMENT_DENSE_SCALAR -> Cpu1MaxPool2dLoops::runF64DenseSegment;
            case MAX_POOL2D_BF16_SEGMENT_DENSE_SCALAR -> Cpu1MaxPool2dLoops::runBf16DenseSegment;
        };
    }
}
