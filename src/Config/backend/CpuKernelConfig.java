package Config.backend;

public final class CpuKernelConfig {
    private final int loopUnrollFactor;
    private final int matMulTileM;
    private final int matMulTileN;
    private final int matMulTileK;
    private final int vectorMinSize;
    private final int parallelMinSize;
    private final int parallelism;
    private final int chunksPerWorker;
    private final int minChunkSize;

    public CpuKernelConfig(int loopUnrollFactor, int matMulTileM, int matMulTileN, int matMulTileK) {
        this(loopUnrollFactor, matMulTileM, matMulTileN, matMulTileK, 1_024, 100_000);
    }

    public CpuKernelConfig(
            int loopUnrollFactor,
            int matMulTileM,
            int matMulTileN,
            int matMulTileK,
            int vectorMinSize,
            int parallelMinSize
    ) {
        this(loopUnrollFactor, matMulTileM, matMulTileN, matMulTileK, vectorMinSize, parallelMinSize, 0, 4, 4_096);
    }

    public CpuKernelConfig(
            int loopUnrollFactor,
            int matMulTileM,
            int matMulTileN,
            int matMulTileK,
            int vectorMinSize,
            int parallelMinSize,
            int parallelism,
            int chunksPerWorker,
            int minChunkSize
    ) {
        this.loopUnrollFactor = loopUnrollFactor;
        this.matMulTileM = matMulTileM;
        this.matMulTileN = matMulTileN;
        this.matMulTileK = matMulTileK;
        this.vectorMinSize = vectorMinSize;
        this.parallelMinSize = parallelMinSize;
        this.parallelism = parallelism;
        this.chunksPerWorker = chunksPerWorker;
        this.minChunkSize = minChunkSize;
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

    public int vectorMinSize() {
        return vectorMinSize;
    }

    public int parallelMinSize() {
        return parallelMinSize;
    }

    public int parallelism() {
        return parallelism;
    }

    public int chunksPerWorker() {
        return chunksPerWorker;
    }

    public int minChunkSize() {
        return minChunkSize;
    }

    public static CpuKernelConfig defaultsTraining() {
        return new CpuKernelConfig(1, 16, 0, 0, 1_024, 100_000, 0, 4, 4_096);
    }

    public static CpuKernelConfig defaultsInference() {
        return new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0, 4, 4_096);
    }
}
