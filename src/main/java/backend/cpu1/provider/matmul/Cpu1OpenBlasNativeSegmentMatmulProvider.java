package backend.cpu1.provider.matmul;

import backend.cpu1.kernels.matmul.Cpu1MatmulKernelId;
import tensor.DataType;

/**
 * Provider for the OpenBLAS native MemorySegment matmul route.
 */
public final class Cpu1OpenBlasNativeSegmentMatmulProvider implements Cpu1MatmulProvider {
    @Override
    public Cpu1MatmulRoute route() {
        return Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT;
    }

    @Override
    public Cpu1MatmulKernelId kernelId(DataType dataType) {
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        return switch (dataType) {
            case FLOAT32 -> Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_NATIVE_SEGMENT;
            case FLOAT64 -> Cpu1MatmulKernelId.MATMUL_F64_OPENBLAS_NATIVE_SEGMENT;
            case BFLOAT16, INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    "cpu1 OPENBLAS_NATIVE_SEGMENT MATMUL does not support " + dataType
            );
        };
    }
}
