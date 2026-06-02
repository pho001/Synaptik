package backend.cpu1.provider.matmul;

import backend.cpu1.kernels.matmul.Cpu1MatmulKernelId;
import tensor.DataType;

import java.util.Objects;

/**
 * Placeholder provider for the planned OpenBLAS array-copy matmul route.
 */
public final class Cpu1OpenBlasArrayMatmulProvider implements Cpu1MatmulProvider {
    @Override
    public Cpu1MatmulRoute route() {
        return Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING;
    }

    @Override
    public Cpu1MatmulKernelId kernelId(DataType dataType) {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        throw new UnsupportedOperationException(
                "cpu1 OpenBLAS array-copy matmul kernels are not implemented yet for " + dataType
        );
    }
}
