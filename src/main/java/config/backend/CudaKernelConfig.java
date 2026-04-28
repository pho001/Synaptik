package config.backend;

/**
 * CUDA kernel tile defaults used by runtime configuration.
 *
 * <p>CUDA execution is currently represented as a backend configuration surface even when the runtime
 * is unavailable. The object is immutable and only stores planning/tile parameters.</p>
 */
public final class CudaKernelConfig {
    private final int loopUnrollFactor;
    private final int matMulTileM;
    private final int matMulTileN;
    private final int matMulTileK;

    /**
     * Creates CUDA kernel tuning values.
     *
     * @param loopUnrollFactor loop unroll factor for generated kernels
     * @param matMulTileM matmul tile size in M dimension
     * @param matMulTileN matmul tile size in N dimension
     * @param matMulTileK matmul tile size in K dimension
     */
    public CudaKernelConfig(int loopUnrollFactor, int matMulTileM, int matMulTileN, int matMulTileK) {
        this.loopUnrollFactor = loopUnrollFactor;
        this.matMulTileM = matMulTileM;
        this.matMulTileN = matMulTileN;
        this.matMulTileK = matMulTileK;
    }

    /**
     * @return loop unroll factor
     */
    public int loopUnrollFactor() {
        return loopUnrollFactor;
    }

    /**
     * @return matmul M tile size
     */
    public int matMulTileM() {
        return matMulTileM;
    }

    /**
     * @return matmul N tile size
     */
    public int matMulTileN() {
        return matMulTileN;
    }

    /**
     * @return matmul K tile size
     */
    public int matMulTileK() {
        return matMulTileK;
    }

    /**
     * @return CUDA defaults for training-capable execution
     */
    public static CudaKernelConfig defaultsTraining() {
        return new CudaKernelConfig(4, 16, 16, 16);
    }

    /**
     * @return CUDA defaults for forward-only inference
     */
    public static CudaKernelConfig defaultsInference() {
        return new CudaKernelConfig(8, 32, 32, 32);
    }
}
