package config.runtime;

/**
 * CPU BLAS provider selected by runtime configuration.
 */
public enum BlasProvider {
    /**
     * Disable external BLAS dispatch and use the built-in CPU kernels.
     */
    NONE,
    /**
     * Use the OpenBLAS CBLAS bridge through the Java Foreign Function and Memory API.
     */
    OPENBLAS_FFM
}
