package backend.kernels.cpu;

public enum CpuComputeMode {
    F64,
    F32,
    BF16_F32_COMPUTE,
    BF16_BLAS,
    INT32,
    BOOL;

    public boolean isFloating() {
        return switch (this) {
            case F64, F32, BF16_F32_COMPUTE, BF16_BLAS -> true;
            case INT32, BOOL -> false;
        };
    }

    public boolean usesBlas() {
        return this == BF16_BLAS;
    }

    public boolean usesF32Compute() {
        return this == F32 || this == BF16_F32_COMPUTE || this == BF16_BLAS;
    }
}
