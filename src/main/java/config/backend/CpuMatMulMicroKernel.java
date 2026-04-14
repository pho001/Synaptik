package config.backend;

import tensor.DataType;

public enum CpuMatMulMicroKernel {
    AUTO,
    F64_2X1,
    F64_4X1,
    F64_2X2,
    F32_2X4,
    F32_2X8,
    F32_4X2,
    F32_4X4;

    public boolean supports(DataType dataType) {
        if (dataType == null) {
            return this == AUTO;
        }
        return switch (dataType) {
            case FLOAT64 -> this == AUTO || this == F64_2X1 || this == F64_4X1 || this == F64_2X2;
            case FLOAT32 -> this == AUTO || this == F32_2X4 || this == F32_2X8 || this == F32_4X2 || this == F32_4X4;
            default -> this == AUTO;
        };
    }

    public CpuMatMulMicroKernel resolve(DataType dataType) {
        if (dataType == null) {
            return AUTO;
        }
        if (supports(dataType) && this != AUTO) {
            return this;
        }
        return defaultFor(dataType);
    }

    public static CpuMatMulMicroKernel defaultFor(DataType dataType) {
        if (dataType == null) {
            return AUTO;
        }
        return switch (dataType) {
            case FLOAT64 -> F64_4X1;
            case FLOAT32 -> F32_4X2;
            default -> AUTO;
        };
    }
}
