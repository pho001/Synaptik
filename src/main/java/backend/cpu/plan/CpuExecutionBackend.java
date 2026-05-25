package backend.cpu.plan;

public enum CpuExecutionBackend {
    CPU_ELEMENTWISE,
    CPU_FUSED,
    CPU_REDUCTION,
    CPU_MATMUL_JAVA,
    CPU_MATMUL_BLAS,
    CPU_GENERIC
}
