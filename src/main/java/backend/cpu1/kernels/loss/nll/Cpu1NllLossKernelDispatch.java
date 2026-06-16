package backend.cpu1.kernels.loss.nll;

public final class Cpu1NllLossKernelDispatch {
    private Cpu1NllLossKernelDispatch() {
    }

    public static Cpu1NllLossKernel kernelFor(Cpu1NllLossKernelId kernelId) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        return switch (kernelId) {
            case NLL_DENSE_F32_ARRAY_DENSE_SCALAR -> Cpu1NllLossLoops::runF32DenseArray;
            case NLL_DENSE_F64_ARRAY_DENSE_SCALAR -> Cpu1NllLossLoops::runF64DenseArray;
            case NLL_DENSE_BF16_ARRAY_DENSE_SCALAR -> Cpu1NllLossLoops::runBf16DenseArray;
            case NLL_DENSE_F32_SEGMENT_DENSE_SCALAR -> Cpu1NllLossLoops::runF32DenseSegment;
            case NLL_DENSE_F64_SEGMENT_DENSE_SCALAR -> Cpu1NllLossLoops::runF64DenseSegment;
            case NLL_DENSE_BF16_SEGMENT_DENSE_SCALAR -> Cpu1NllLossLoops::runBf16DenseSegment;
        };
    }
}
