package backend.cpu1.kernels.matmul;

/**
 * Prepared matmul execution variants for cpu1.
 */
public enum Cpu1MatmulKernelId {
    MATMUL_F32_DENSE_SCALAR,
    MATMUL_F64_DENSE_SCALAR,
    MATMUL_BF16_DENSE_SCALAR
}
