package config.backend;

import tensor.DataType;

/**
 * CPU matmul microkernel variants available to calibrated runtime profiles.
 *
 * <p>Each non-{@link #AUTO} value names a dtype-specific register/block shape. For example,
 * {@link #F32_4X2} is a FLOAT32 microkernel tuned around a 4-by-2 output block. Calibration stores
 * these values in {@link config.profile.MatmulPlatformProfile} and dispatch resolves unsupported
 * choices back to a dtype-appropriate default.</p>
 */
public enum CpuMatMulMicroKernel {
    /**
     * Select the dtype-specific default microkernel.
     */
    AUTO,
    /**
     * FLOAT64 2-by-1 output-block microkernel.
     */
    F64_2X1,
    /**
     * FLOAT64 4-by-1 output-block microkernel.
     */
    F64_4X1,
    /**
     * FLOAT64 2-by-2 output-block microkernel.
     */
    F64_2X2,
    /**
     * BF16 2-by-4 output-block microkernel.
     */
    BF16_2X4,
    /**
     * BF16 4-by-2 output-block microkernel.
     */
    BF16_4X2,
    /**
     * BF16 4-by-4 output-block microkernel.
     */
    BF16_4X4,
    /**
     * FLOAT32 2-by-4 output-block microkernel.
     */
    F32_2X4,
    /**
     * FLOAT32 2-by-8 output-block microkernel.
     */
    F32_2X8,
    /**
     * FLOAT32 4-by-2 output-block microkernel.
     */
    F32_4X2,
    /**
     * FLOAT32 4-by-4 output-block microkernel.
     */
    F32_4X4;

    /**
     * Reports whether this microkernel can execute the supplied dtype.
     *
     * @param dataType dtype to test; {@code null} is supported only by {@link #AUTO}
     * @return {@code true} if this value is {@link #AUTO} or a concrete kernel for {@code dataType}
     */
    public boolean supports(DataType dataType) {
        if (dataType == null) {
            return this == AUTO;
        }
        return switch (dataType) {
            case FLOAT64 -> this == AUTO || this == F64_2X1 || this == F64_4X1 || this == F64_2X2;
            case BFLOAT16 -> this == AUTO || this == BF16_2X4 || this == BF16_4X2 || this == BF16_4X4;
            case FLOAT32 -> this == AUTO || this == F32_2X4 || this == F32_2X8 || this == F32_4X2 || this == F32_4X4;
            default -> this == AUTO;
        };
    }

    /**
     * Resolves this value to a concrete microkernel for the supplied dtype.
     *
     * <p>If this value is unsupported for {@code dataType}, the dtype default is returned. This keeps a
     * persisted profile safe when reused with a dtype for which a manually selected kernel is invalid.</p>
     *
     * @param dataType dtype to execute; {@code null} resolves to {@link #AUTO}
     * @return this value when supported and concrete, otherwise {@link #defaultFor(DataType)}
     */
    public CpuMatMulMicroKernel resolve(DataType dataType) {
        if (dataType == null) {
            return AUTO;
        }
        if (supports(dataType) && this != AUTO) {
            return this;
        }
        return defaultFor(dataType);
    }

    /**
     * Returns the preferred concrete CPU matmul microkernel for a dtype.
     *
     * @param dataType dtype to execute; {@code null}, INT32, and BOOL return {@link #AUTO}
     * @return dtype-specific default microkernel
     */
    public static CpuMatMulMicroKernel defaultFor(DataType dataType) {
        if (dataType == null) {
            return AUTO;
        }
        return switch (dataType) {
            case FLOAT64 -> F64_4X1;
            case BFLOAT16 -> BF16_4X2;
            case FLOAT32 -> F32_4X2;
            default -> AUTO;
        };
    }
}
