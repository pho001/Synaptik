package backend.cpu1.provider.matmul;

import backend.cpu1.kernels.matmul.Cpu1MatmulKernelId;
import tensor.DataType;

import java.util.Objects;

/**
 * Provider for the OpenBLAS array-copy matmul route.
 */
public final class Cpu1OpenBlasArrayMatmulProvider implements Cpu1MatmulProvider {
    @Override
    public Cpu1MatmulRoute route() {
        return Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING;
    }

    @Override
    public Cpu1MatmulKernelId kernelId(DataType dataType) {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        return switch (dataType) {
            case FLOAT32 -> Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_ARRAY_COPYING;
            case FLOAT64 -> Cpu1MatmulKernelId.MATMUL_F64_OPENBLAS_ARRAY_COPYING;
            case BFLOAT16, INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    "cpu1 OPENBLAS_ARRAY_COPYING MATMUL does not support " + dataType
            );
        };
    }
}
