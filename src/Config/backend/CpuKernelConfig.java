package Config.backend;

public final class CpuKernelConfig {
    private final int loopUnrollFactor;
    private final int matMulTileM;
    private final int matMulTileN;
    private final int matMulTileK;

    public CpuKernelConfig(int loopUnrollFactor, int matMulTileM, int matMulTileN, int matMulTileK) {
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

    public static CpuKernelConfig defaultsTraining() {
        return new CpuKernelConfig(1, 0, 0, 0);
    }

    public static CpuKernelConfig defaultsInference() {
        return new CpuKernelConfig(4, 32, 32, 32);
    }
}
