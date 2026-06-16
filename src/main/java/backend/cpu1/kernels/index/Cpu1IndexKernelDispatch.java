package backend.cpu1.kernels.index;

import backend.cpu1.kernels.index.gather.Cpu1GatherLoops;

/**
 * Resolves prepared index kernel ids to concrete kernels.
 */
public final class Cpu1IndexKernelDispatch {
    private Cpu1IndexKernelDispatch() {
    }

    public static Cpu1IndexKernel kernelFor(Cpu1IndexKernelId kernelId) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        return switch (kernelId) {
            case GATHER_F32_I32_DENSE_ARRAY -> Cpu1GatherLoops::gatherF32I32DenseArray;
            case GATHER_F32_I64_DENSE_ARRAY -> Cpu1GatherLoops::gatherF32I64DenseArray;
            case GATHER_F64_I32_DENSE_ARRAY -> Cpu1GatherLoops::gatherF64I32DenseArray;
            case GATHER_F64_I64_DENSE_ARRAY -> Cpu1GatherLoops::gatherF64I64DenseArray;
            case GATHER_BF16_I32_DENSE_ARRAY -> Cpu1GatherLoops::gatherBf16I32DenseArray;
            case GATHER_BF16_I64_DENSE_ARRAY -> Cpu1GatherLoops::gatherBf16I64DenseArray;
            case GATHER_I32_I32_DENSE_ARRAY -> Cpu1GatherLoops::gatherI32I32DenseArray;
            case GATHER_I32_I64_DENSE_ARRAY -> Cpu1GatherLoops::gatherI32I64DenseArray;
            case GATHER_I64_I32_DENSE_ARRAY -> Cpu1GatherLoops::gatherI64I32DenseArray;
            case GATHER_I64_I64_DENSE_ARRAY -> Cpu1GatherLoops::gatherI64I64DenseArray;
            case GATHER_BOOL_I32_DENSE_ARRAY -> Cpu1GatherLoops::gatherBoolI32DenseArray;
            case GATHER_BOOL_I64_DENSE_ARRAY -> Cpu1GatherLoops::gatherBoolI64DenseArray;
        };
    }
}
