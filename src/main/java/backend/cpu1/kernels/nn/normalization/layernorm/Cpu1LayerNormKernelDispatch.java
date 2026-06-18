package backend.cpu1.kernels.nn.normalization.layernorm;

/**
 * Resolves prepared cpu1 LayerNorm kernel ids.
 */
public final class Cpu1LayerNormKernelDispatch {
    private Cpu1LayerNormKernelDispatch() {
    }

    public static Cpu1LayerNormKernel kernelFor(Cpu1LayerNormKernelId kernelId) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        return switch (kernelId) {
            case LAYER_NORM_F32_ARRAY_DENSE_SCALAR -> Cpu1LayerNormLoops::runF32DenseArray;
            case LAYER_NORM_F64_ARRAY_DENSE_SCALAR -> Cpu1LayerNormLoops::runF64DenseArray;
            case LAYER_NORM_BF16_ARRAY_DENSE_SCALAR -> Cpu1LayerNormLoops::runBf16DenseArray;
            case LAYER_NORM_F32_SEGMENT_DENSE_SCALAR -> Cpu1LayerNormLoops::runF32DenseSegment;
            case LAYER_NORM_F64_SEGMENT_DENSE_SCALAR -> Cpu1LayerNormLoops::runF64DenseSegment;
            case LAYER_NORM_BF16_SEGMENT_DENSE_SCALAR -> Cpu1LayerNormLoops::runBf16DenseSegment;
        };
    }
}
