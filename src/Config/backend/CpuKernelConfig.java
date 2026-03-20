package Config.backend;

public final class CpuKernelConfig {
    private static final int DEFAULT_VECTOR_MIN_SIZE = 1_024;
    private static final int DEFAULT_PARALLEL_MIN_SIZE = 100_000;
    private static final int DEFAULT_PARALLELISM = 0;
    private static final int DEFAULT_CHUNKS_PER_WORKER = 4;
    private static final int DEFAULT_MIN_CHUNK_SIZE = 4_096;
    private static final int DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD = 1_000_000_000;

    private final int loopUnrollFactor;
    private final int matMulTileM;
    private final int matMulTileN;
    private final int matMulTileK;
    private final int vectorMinSize;
    private final int parallelMinSize;
    private final int parallelism;
    private final int chunksPerWorker;
    private final int minChunkSize;
    private final int contiguousMaterializeThreshold;

    public CpuKernelConfig(int loopUnrollFactor, int matMulTileM, int matMulTileN, int matMulTileK) {
        this(loopUnrollFactor, matMulTileM, matMulTileN, matMulTileK, DEFAULT_VECTOR_MIN_SIZE, DEFAULT_PARALLEL_MIN_SIZE);
    }

    public CpuKernelConfig(
            int loopUnrollFactor,
            int matMulTileM,
            int matMulTileN,
            int matMulTileK,
            int vectorMinSize,
            int parallelMinSize
    ) {
        this(loopUnrollFactor, matMulTileM, matMulTileN, matMulTileK, vectorMinSize, parallelMinSize, DEFAULT_PARALLELISM, DEFAULT_CHUNKS_PER_WORKER, DEFAULT_MIN_CHUNK_SIZE, DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD);
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
        this(
                loopUnrollFactor,
                matMulTileM,
                matMulTileN,
                matMulTileK,
                vectorMinSize,
                parallelMinSize,
                parallelism,
                chunksPerWorker,
                minChunkSize,
                DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD
        );
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
            int minChunkSize,
            int contiguousMaterializeThreshold
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
        this.contiguousMaterializeThreshold = contiguousMaterializeThreshold;
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

    public int contiguousMaterializeThreshold() {
        return contiguousMaterializeThreshold;
    }

    public static CpuKernelConfig defaultsTraining() {
        return new CpuKernelConfig(
                1,
                16,
                0,
                0,
                DEFAULT_VECTOR_MIN_SIZE,
                DEFAULT_PARALLEL_MIN_SIZE,
                DEFAULT_PARALLELISM,
                DEFAULT_CHUNKS_PER_WORKER,
                DEFAULT_MIN_CHUNK_SIZE,
                DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD
        );
    }

    public static CpuKernelConfig defaultsInference() {
        return new CpuKernelConfig(
                4,
                32,
                32,
                32,
                DEFAULT_VECTOR_MIN_SIZE,
                DEFAULT_PARALLEL_MIN_SIZE,
                DEFAULT_PARALLELISM,
                DEFAULT_CHUNKS_PER_WORKER,
                DEFAULT_MIN_CHUNK_SIZE,
                DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD
        );
    }
}
