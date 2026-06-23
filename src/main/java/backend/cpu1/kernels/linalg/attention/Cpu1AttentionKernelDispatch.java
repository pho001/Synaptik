package backend.cpu1.kernels.linalg.attention;

/**
 * Resolves prepared cpu1 attention kernel ids.
 */
public final class Cpu1AttentionKernelDispatch {
    private Cpu1AttentionKernelDispatch() {
    }

    public static Cpu1AttentionKernel kernelFor(Cpu1AttentionKernelId kernelId) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        return switch (kernelId) {
            case ATTENTION_F32_ARRAY_DENSE_SCALAR,
                 ATTENTION_F32_ARRAY_DENSE_VECTOR,
                 ATTENTION_F64_ARRAY_DENSE_SCALAR,
                 ATTENTION_F64_ARRAY_DENSE_VECTOR,
                 ATTENTION_BF16_ARRAY_DENSE_SCALAR,
                 ATTENTION_F32_SEGMENT_DENSE_SCALAR,
                 ATTENTION_F32_SEGMENT_DENSE_VECTOR,
                 ATTENTION_F64_SEGMENT_DENSE_SCALAR,
                 ATTENTION_F64_SEGMENT_DENSE_VECTOR,
                 ATTENTION_BF16_SEGMENT_DENSE_SCALAR -> Cpu1AttentionLoops::runAttention;
            case ATTENTION_WEIGHTS_F32_ARRAY_DENSE_SCALAR,
                 ATTENTION_WEIGHTS_F64_ARRAY_DENSE_SCALAR,
                 ATTENTION_WEIGHTS_BF16_ARRAY_DENSE_SCALAR,
                 ATTENTION_WEIGHTS_F32_SEGMENT_DENSE_SCALAR,
                 ATTENTION_WEIGHTS_F64_SEGMENT_DENSE_SCALAR,
                 ATTENTION_WEIGHTS_BF16_SEGMENT_DENSE_SCALAR -> Cpu1AttentionLoops::runAttentionWeights;
        };
    }
}
