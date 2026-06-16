package backend.cpu1.kernels.loss.crossentropy;

public final class Cpu1DenseCrossEntropyKernelDispatch {
    private Cpu1DenseCrossEntropyKernelDispatch() {
    }

    public static Cpu1DenseCrossEntropyKernel kernelFor(Cpu1DenseCrossEntropyKernelId kernelId) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        return switch (kernelId) {
            case CROSS_ENTROPY_DENSE_F32_ARRAY_DENSE_SCALAR -> Cpu1DenseCrossEntropyLossLoops::runF32DenseArray;
            case CROSS_ENTROPY_DENSE_F64_ARRAY_DENSE_SCALAR -> Cpu1DenseCrossEntropyLossLoops::runF64DenseArray;
            case CROSS_ENTROPY_DENSE_BF16_ARRAY_DENSE_SCALAR -> Cpu1DenseCrossEntropyLossLoops::runBf16DenseArray;
            case CROSS_ENTROPY_DENSE_F32_SEGMENT_DENSE_SCALAR -> Cpu1DenseCrossEntropyLossLoops::runF32DenseSegment;
            case CROSS_ENTROPY_DENSE_F64_SEGMENT_DENSE_SCALAR -> Cpu1DenseCrossEntropyLossLoops::runF64DenseSegment;
            case CROSS_ENTROPY_DENSE_BF16_SEGMENT_DENSE_SCALAR -> Cpu1DenseCrossEntropyLossLoops::runBf16DenseSegment;
        };
    }
}
