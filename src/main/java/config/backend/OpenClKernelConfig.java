package config.backend;

/**
 * OpenCL kernel tile defaults used by runtime configuration.
 *
 * <p>The object is immutable and stores only planning/tile parameters. Runtime availability is handled
 * separately by accelerator configuration.</p>
 */
public final class OpenClKernelConfig {
    private final int loopUnrollFactor;
    private final int matMulTileM;
    private final int matMulTileN;
    private final int matMulTileK;

    /**
     * Creates OpenCL kernel tuning values.
     *
     * @param loopUnrollFactor loop unroll factor for generated kernels
     * @param matMulTileM matmul tile size in M dimension
     * @param matMulTileN matmul tile size in N dimension
     * @param matMulTileK matmul tile size in K dimension
     */
    public OpenClKernelConfig(int loopUnrollFactor, int matMulTileM, int matMulTileN, int matMulTileK) {
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
     * @return OpenCL defaults for training-capable execution
     */
    public static OpenClKernelConfig defaultsTraining() {
        return new OpenClKernelConfig(1, 0, 0, 0);
    }

    /**
     * @return OpenCL defaults for forward-only inference
     */
    public static OpenClKernelConfig defaultsInference() {
        return new OpenClKernelConfig(4, 32, 32, 16);
    }
}
