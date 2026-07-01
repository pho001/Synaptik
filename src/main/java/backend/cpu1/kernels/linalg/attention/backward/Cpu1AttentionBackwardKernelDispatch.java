package backend.cpu1.kernels.linalg.attention.backward;

/**
 * Resolves prepared cpu1 attention backward kernel ids.
 */
public final class Cpu1AttentionBackwardKernelDispatch {
    private Cpu1AttentionBackwardKernelDispatch() {
    }

    public static Cpu1AttentionBackwardKernel kernelFor(Cpu1AttentionBackwardKernelId kernelId) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        return switch (kernelId) {
            case SDPA_BACKWARD_DQ_F32_ARRAY_DENSE_SCALAR,
                 SDPA_BACKWARD_DQ_F32_ARRAY_DENSE_VECTOR,
                 SDPA_BACKWARD_DK_F32_ARRAY_DENSE_SCALAR,
                 SDPA_BACKWARD_DK_F32_ARRAY_DENSE_VECTOR,
                 SDPA_BACKWARD_DV_F32_ARRAY_DENSE_SCALAR,
                 SDPA_BACKWARD_DV_F32_ARRAY_DENSE_VECTOR,
                 SDPA_BACKWARD_DQ_F64_ARRAY_DENSE_SCALAR,
                 SDPA_BACKWARD_DQ_F64_ARRAY_DENSE_VECTOR,
                 SDPA_BACKWARD_DK_F64_ARRAY_DENSE_SCALAR,
                 SDPA_BACKWARD_DK_F64_ARRAY_DENSE_VECTOR,
                 SDPA_BACKWARD_DV_F64_ARRAY_DENSE_SCALAR,
                 SDPA_BACKWARD_DV_F64_ARRAY_DENSE_VECTOR -> Cpu1AttentionBackwardLoops::runArray;
            case SDPA_BACKWARD_DQ_F32_SEGMENT_DENSE_SCALAR,
                 SDPA_BACKWARD_DQ_F32_SEGMENT_DENSE_VECTOR,
                 SDPA_BACKWARD_DK_F32_SEGMENT_DENSE_SCALAR,
                 SDPA_BACKWARD_DK_F32_SEGMENT_DENSE_VECTOR,
                 SDPA_BACKWARD_DV_F32_SEGMENT_DENSE_SCALAR,
                 SDPA_BACKWARD_DV_F32_SEGMENT_DENSE_VECTOR,
                 SDPA_BACKWARD_DQ_F64_SEGMENT_DENSE_SCALAR,
                 SDPA_BACKWARD_DQ_F64_SEGMENT_DENSE_VECTOR,
                 SDPA_BACKWARD_DK_F64_SEGMENT_DENSE_SCALAR,
                 SDPA_BACKWARD_DK_F64_SEGMENT_DENSE_VECTOR,
                 SDPA_BACKWARD_DV_F64_SEGMENT_DENSE_SCALAR,
                 SDPA_BACKWARD_DV_F64_SEGMENT_DENSE_VECTOR -> Cpu1AttentionBackwardLoops::runSegment;
        };
    }
}
