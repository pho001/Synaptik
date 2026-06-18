package backend.cpu1.kernels.nn.normalization.rmsnorm;

/**
 * Resolves prepared cpu1 RMSNorm kernel ids.
 */
public final class Cpu1RmsNormKernelDispatch {
    private Cpu1RmsNormKernelDispatch() {
    }

    public static Cpu1RmsNormKernel kernelFor(Cpu1RmsNormKernelId kernelId) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        return switch (kernelId) {
            case RMS_NORM_F32_ARRAY_DENSE_SCALAR -> Cpu1RmsNormLoops::runF32DenseArray;
            case RMS_NORM_F64_ARRAY_DENSE_SCALAR -> Cpu1RmsNormLoops::runF64DenseArray;
            case RMS_NORM_BF16_ARRAY_DENSE_SCALAR -> Cpu1RmsNormLoops::runBf16DenseArray;
            case RMS_NORM_F32_SEGMENT_DENSE_SCALAR -> Cpu1RmsNormLoops::runF32DenseSegment;
            case RMS_NORM_F64_SEGMENT_DENSE_SCALAR -> Cpu1RmsNormLoops::runF64DenseSegment;
            case RMS_NORM_BF16_SEGMENT_DENSE_SCALAR -> Cpu1RmsNormLoops::runBf16DenseSegment;
        };
    }
}
