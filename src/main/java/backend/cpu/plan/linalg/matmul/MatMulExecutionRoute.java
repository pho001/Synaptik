package backend.cpu.plan.linalg.matmul;

/**
 * Concrete runtime route selected for a prepared CPU matmul step.
 */
public enum MatMulExecutionRoute {
    JAVA_DIRECT,
    OPENBLAS_ARRAY_COPYING,
    OPENBLAS_NATIVE_SEGMENT
}
