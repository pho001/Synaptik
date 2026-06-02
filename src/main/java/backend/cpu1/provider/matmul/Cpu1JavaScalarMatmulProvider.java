package backend.cpu1.provider.matmul;

import backend.cpu1.kernels.matmul.Cpu1MatmulKernelId;
import tensor.DataType;

import java.util.Objects;

/**
 * Provider for the initial dense Java scalar matmul route.
 */
public final class Cpu1JavaScalarMatmulProvider implements Cpu1MatmulProvider {
    @Override
    public Cpu1MatmulRoute route() {
        return Cpu1MatmulRoute.JAVA_SCALAR;
    }

    @Override
    public Cpu1MatmulKernelId kernelId(DataType dataType) {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        return switch (dataType) {
            case FLOAT32 -> Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR;
            case FLOAT64 -> Cpu1MatmulKernelId.MATMUL_F64_DENSE_SCALAR;
            case BFLOAT16 -> Cpu1MatmulKernelId.MATMUL_BF16_DENSE_SCALAR;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    "cpu1 MATMUL does not support " + dataType
            );
        };
    }
}
