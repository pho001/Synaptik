package Config.backend;

public final class CudaKernelConfig {
    private final int loopUnrollFactor;
    private final int matMulTileM;
    private final int matMulTileN;
    private final int matMulTileK;

    public CudaKernelConfig(int loopUnrollFactor, int matMulTileM, int matMulTileN, int matMulTileK) {
        this.loopUnrollFactor = loopUnrollFactor;
        this.matMulTileM = matMulTileM;
        this.matMulTileN = matMulTileN;
        this.matMulTileK = matMulTileK;
    }

    public int loopUnrollFactor() {
        return loopUnrollFactor;
    }

    public int matMulTileM() {
        return matMulTileM;
    }

    public int matMulTileN() {
        return matMulTileN;
    }

    public int matMulTileK() {
        return matMulTileK;
    }

    public static CudaKernelConfig defaultsTraining() {
        return new CudaKernelConfig(4, 16, 16, 16);
    }

    public static CudaKernelConfig defaultsInference() {
        return new CudaKernelConfig(8, 32, 32, 32);
    }
}
