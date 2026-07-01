package backend.cpu1.kernels.loss.mse;

import backend.cpu1.storage.Cpu1StorageKind;

public final class Cpu1MseLossKernelDispatch {
    private Cpu1MseLossKernelDispatch() {
    }

    public static Cpu1MseLossKernel kernelFor(Cpu1MseLossKernelId kernelId, Cpu1StorageKind storageKind) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        if (storageKind == null) {
            throw new IllegalArgumentException("storageKind cannot be null");
        }
        return switch (storageKind) {
            case JAVA_ARRAY -> arrayKernelFor(kernelId);
            case MEMORY_SEGMENT -> segmentKernelFor(kernelId);
        };
    }

    private static Cpu1MseLossKernel arrayKernelFor(Cpu1MseLossKernelId kernelId) {
        return switch (kernelId) {
            case MSE_SUM_F32_DENSE_SCALAR -> Cpu1MseLossLoops::sumF32DenseScalar;
            case MSE_MEAN_F32_DENSE_SCALAR -> Cpu1MseLossLoops::meanF32DenseScalar;
            case MSE_SUM_F64_DENSE_SCALAR -> Cpu1MseLossLoops::sumF64DenseScalar;
            case MSE_MEAN_F64_DENSE_SCALAR -> Cpu1MseLossLoops::meanF64DenseScalar;
            case MSE_SUM_BF16_DENSE_SCALAR -> Cpu1MseLossLoops::sumBf16DenseScalar;
            case MSE_MEAN_BF16_DENSE_SCALAR -> Cpu1MseLossLoops::meanBf16DenseScalar;
        };
    }

    private static Cpu1MseLossKernel segmentKernelFor(Cpu1MseLossKernelId kernelId) {
        return switch (kernelId) {
            case MSE_SUM_F32_DENSE_SCALAR -> Cpu1MseLossLoops::sumF32DenseScalarSegment;
            case MSE_MEAN_F32_DENSE_SCALAR -> Cpu1MseLossLoops::meanF32DenseScalarSegment;
            case MSE_SUM_F64_DENSE_SCALAR -> Cpu1MseLossLoops::sumF64DenseScalarSegment;
            case MSE_MEAN_F64_DENSE_SCALAR -> Cpu1MseLossLoops::meanF64DenseScalarSegment;
            case MSE_SUM_BF16_DENSE_SCALAR, MSE_MEAN_BF16_DENSE_SCALAR ->
                    throw new UnsupportedOperationException("cpu1 MSE_LOSS MEMORY_SEGMENT supports FLOAT32/FLOAT64 only.");
        };
    }
}
