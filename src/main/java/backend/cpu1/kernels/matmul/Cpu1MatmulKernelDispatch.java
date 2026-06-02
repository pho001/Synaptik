package backend.cpu1.kernels.matmul;

import java.util.Objects;

/**
 * Resolves prepared matmul kernel ids to concrete matmul kernels.
 */
public final class Cpu1MatmulKernelDispatch {
    private Cpu1MatmulKernelDispatch() {
    }

    public static Cpu1MatmulKernel kernelFor(Cpu1MatmulKernelId kernelId) {
        Objects.requireNonNull(kernelId, "kernelId cannot be null");
        return switch (kernelId) {
            case MATMUL_F32_DENSE_SCALAR -> Cpu1JavaScalarMatmulLoops::matmulF32DenseScalar;
            case MATMUL_F32_DENSE_PACKED_B_VECTOR -> Cpu1JavaVectorMatmulLoops::matmulF32DensePackedBVector;
            case MATMUL_F32_OPENBLAS_ARRAY_COPYING -> Cpu1OpenBlasArrayMatmulLoops::matmulF32ArrayCopy;
            case MATMUL_F64_DENSE_SCALAR -> Cpu1JavaScalarMatmulLoops::matmulF64DenseScalar;
            case MATMUL_F64_DENSE_PACKED_B_VECTOR -> Cpu1JavaVectorMatmulLoops::matmulF64DensePackedBVector;
            case MATMUL_F64_OPENBLAS_ARRAY_COPYING -> Cpu1OpenBlasArrayMatmulLoops::matmulF64ArrayCopy;
            case MATMUL_BF16_DENSE_SCALAR -> Cpu1JavaScalarMatmulLoops::matmulBf16DenseScalar;
        };
    }
}
