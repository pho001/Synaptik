package backend.cpu1.kernels.loss.crossentropy;

public final class Cpu1CrossEntropyKernelDispatch {
    private Cpu1CrossEntropyKernelDispatch() {
    }

    public static Cpu1CrossEntropyKernel kernelFor(Cpu1CrossEntropyKernelId kernelId) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        return switch (kernelId) {
            case CROSS_ENTROPY_INDICES_F32_I32_ARRAY_DENSE_SCALAR ->
                    Cpu1CrossEntropyLossIndicesLoops::runF32I32DenseArray;
            case CROSS_ENTROPY_INDICES_F32_I64_ARRAY_DENSE_SCALAR ->
                    Cpu1CrossEntropyLossIndicesLoops::runF32I64DenseArray;
            case CROSS_ENTROPY_INDICES_F64_I32_ARRAY_DENSE_SCALAR ->
                    Cpu1CrossEntropyLossIndicesLoops::runF64I32DenseArray;
            case CROSS_ENTROPY_INDICES_F64_I64_ARRAY_DENSE_SCALAR ->
                    Cpu1CrossEntropyLossIndicesLoops::runF64I64DenseArray;
            case CROSS_ENTROPY_INDICES_BF16_I32_ARRAY_DENSE_SCALAR ->
                    Cpu1CrossEntropyLossIndicesLoops::runBf16I32DenseArray;
            case CROSS_ENTROPY_INDICES_BF16_I64_ARRAY_DENSE_SCALAR ->
                    Cpu1CrossEntropyLossIndicesLoops::runBf16I64DenseArray;
            case CROSS_ENTROPY_INDICES_F32_I32_SEGMENT_DENSE_SCALAR ->
                    Cpu1CrossEntropyLossIndicesLoops::runF32I32DenseSegment;
            case CROSS_ENTROPY_INDICES_F32_I64_SEGMENT_DENSE_SCALAR ->
                    Cpu1CrossEntropyLossIndicesLoops::runF32I64DenseSegment;
            case CROSS_ENTROPY_INDICES_F64_I32_SEGMENT_DENSE_SCALAR ->
                    Cpu1CrossEntropyLossIndicesLoops::runF64I32DenseSegment;
            case CROSS_ENTROPY_INDICES_F64_I64_SEGMENT_DENSE_SCALAR ->
                    Cpu1CrossEntropyLossIndicesLoops::runF64I64DenseSegment;
            case CROSS_ENTROPY_INDICES_BF16_I32_SEGMENT_DENSE_SCALAR ->
                    Cpu1CrossEntropyLossIndicesLoops::runBf16I32DenseSegment;
            case CROSS_ENTROPY_INDICES_BF16_I64_SEGMENT_DENSE_SCALAR ->
                    Cpu1CrossEntropyLossIndicesLoops::runBf16I64DenseSegment;
        };
    }
}
