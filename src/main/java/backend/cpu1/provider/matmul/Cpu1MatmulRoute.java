package backend.cpu1.provider.matmul;

/**
 * Prepared cpu1 matmul execution route.
 */
public enum Cpu1MatmulRoute {
    JAVA_SCALAR,
    AUTO,
    OPENBLAS_ARRAY_COPYING,
    OPENBLAS_NATIVE_SEGMENT
}
