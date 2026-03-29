package Config.backend;

public final class OpenClKernelConfig {
    private final int loopUnrollFactor;
    private final int matMulTileM;
    private final int matMulTileN;
    private final int matMulTileK;

    public OpenClKernelConfig(int loopUnrollFactor, int matMulTileM, int matMulTileN, int matMulTileK) {
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

    public static OpenClKernelConfig defaultsTraining() {
        return new OpenClKernelConfig(1, 0, 0, 0);
    }

    public static OpenClKernelConfig defaultsInference() {
        return new OpenClKernelConfig(4, 32, 32, 16);
    }
}
