package backend.cpu1.kernels.loss.mse;

import java.util.Objects;

public final class Cpu1MseLossKernelDispatch {
    private Cpu1MseLossKernelDispatch() {
    }

    public static Cpu1MseLossKernel kernelFor(Cpu1MseLossKernelId kernelId) {
        Objects.requireNonNull(kernelId, "kernelId cannot be null");
        return switch (kernelId) {
            case MSE_SUM_F32_DENSE_SCALAR -> Cpu1MseLossLoops::sumF32DenseScalar;
            case MSE_MEAN_F32_DENSE_SCALAR -> Cpu1MseLossLoops::meanF32DenseScalar;
            case MSE_SUM_F64_DENSE_SCALAR -> Cpu1MseLossLoops::sumF64DenseScalar;
            case MSE_MEAN_F64_DENSE_SCALAR -> Cpu1MseLossLoops::meanF64DenseScalar;
            case MSE_SUM_BF16_DENSE_SCALAR -> Cpu1MseLossLoops::sumBf16DenseScalar;
            case MSE_MEAN_BF16_DENSE_SCALAR -> Cpu1MseLossLoops::meanBf16DenseScalar;
        };
    }
}
