package config.backend;

public final class CpuKernelConfig {
    private static final int DEFAULT_VECTOR_MIN_SIZE = 1_024;
    private static final int DEFAULT_PARALLEL_MIN_SIZE = 100_000;
    private static final int DEFAULT_MATMUL_PARALLEL_MIN_SIZE = 2_000_000;
    private static final int DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD = 1_000_000_000;
    private static final int DEFAULT_LOW_COST_TARGET_CHUNKS_PER_WORKER = 4;
    private static final int DEFAULT_MEDIUM_COST_TARGET_CHUNKS_PER_WORKER = 2;
    private static final int DEFAULT_HIGH_COST_TARGET_CHUNKS_PER_WORKER = 1;
    private static final int DEFAULT_MIN_SCALAR_CHUNK_SIZE = 4_096;
    private static final int DEFAULT_MIN_VECTOR_CHUNK_SIZE = 8_192;
    private static final int DEFAULT_MIN_REDUCTION_CHUNK_SIZE = 16_384;
    private static final int DEFAULT_COMMON_POOL_LOW_COST_MAX_WORK_PER_WORKER = 16_384;
    private static final int DEFAULT_FUSED_ASM_VECTOR_WIDTH = 1;
    private static final SumAccuracyMode DEFAULT_SUM_ACCURACY_MODE = SumAccuracyMode.FAST;
    private static final AttentionMatMulPolicy DEFAULT_ATTENTION_MATMUL_POLICY = AttentionMatMulPolicy.AUTO;

    private final int loopUnrollFactor;
    private final int matMulTileM;
    private final int matMulTileN;
    private final int matMulTileK;
    private final int cheapVectorMinSize;
    private final int transcendentalVectorMinSize;
    private final int fusedCheapVectorMinSize;
    private final int fusedTranscendentalVectorMinSize;
    private final int reductionVectorMinSize;
    private final int cheapParallelMinSize;
    private final int transcendentalParallelMinSize;
    private final int fusedCheapParallelMinSize;
    private final int fusedTranscendentalParallelMinSize;
    private final int reductionParallelMinSize;
    private final int matMulParallelMinSize;
    private final int contiguousMaterializeThreshold;
    private final int lowCostTargetChunksPerWorker;
    private final int mediumCostTargetChunksPerWorker;
    private final int highCostTargetChunksPerWorker;
    private final int minScalarChunkSize;
    private final int minVectorChunkSize;
    private final int minReductionChunkSize;
    private final int commonPoolLowCostMaxWorkPerWorker;
    private final int fusedAsmVectorWidth;
    private final SumAccuracyMode sumAccuracyMode;
    private final AttentionMatMulPolicy attentionMatMulPolicy;

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
        this(loopUnrollFactor, matMulTileM, matMulTileN, matMulTileK, vectorMinSize, parallelMinSize, DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD);
    }

    public CpuKernelConfig(
            int loopUnrollFactor,
            int matMulTileM,
            int matMulTileN,
            int matMulTileK,
            int vectorMinSize,
            int parallelMinSize,
            int contiguousMaterializeThreshold
    ) {
        this(loopUnrollFactor, matMulTileM, matMulTileN, matMulTileK, vectorMinSize, parallelMinSize, contiguousMaterializeThreshold, DEFAULT_SUM_ACCURACY_MODE);
    }

    public CpuKernelConfig(
            int loopUnrollFactor,
            int matMulTileM,
            int matMulTileN,
            int matMulTileK,
            int vectorMinSize,
            int parallelMinSize,
            int contiguousMaterializeThreshold,
            SumAccuracyMode sumAccuracyMode
    ) {
        this(
                loopUnrollFactor,
                matMulTileM,
                matMulTileN,
                matMulTileK,
                vectorMinSize,
                vectorMinSize,
                vectorMinSize,
                vectorMinSize,
                vectorMinSize,
                parallelMinSize,
                parallelMinSize,
                parallelMinSize,
                parallelMinSize,
                parallelMinSize,
                contiguousMaterializeThreshold,
                DEFAULT_LOW_COST_TARGET_CHUNKS_PER_WORKER,
                DEFAULT_MEDIUM_COST_TARGET_CHUNKS_PER_WORKER,
                DEFAULT_HIGH_COST_TARGET_CHUNKS_PER_WORKER,
                DEFAULT_MIN_SCALAR_CHUNK_SIZE,
                DEFAULT_MIN_VECTOR_CHUNK_SIZE,
                DEFAULT_MIN_REDUCTION_CHUNK_SIZE,
                DEFAULT_COMMON_POOL_LOW_COST_MAX_WORK_PER_WORKER,
                DEFAULT_FUSED_ASM_VECTOR_WIDTH,
                sumAccuracyMode,
                DEFAULT_MATMUL_PARALLEL_MIN_SIZE,
                DEFAULT_ATTENTION_MATMUL_POLICY
        );
    }

    public CpuKernelConfig(
            int loopUnrollFactor,
            int matMulTileM,
            int matMulTileN,
            int matMulTileK,
            int cheapVectorMinSize,
            int transcendentalVectorMinSize,
            int fusedCheapVectorMinSize,
            int fusedTranscendentalVectorMinSize,
            int reductionVectorMinSize,
            int cheapParallelMinSize,
            int transcendentalParallelMinSize,
            int fusedCheapParallelMinSize,
            int fusedTranscendentalParallelMinSize,
            int reductionParallelMinSize,
            int contiguousMaterializeThreshold,
            SumAccuracyMode sumAccuracyMode,
            int matMulParallelMinSize,
            AttentionMatMulPolicy attentionMatMulPolicy
    ) {
        this(
                loopUnrollFactor,
                matMulTileM,
                matMulTileN,
                matMulTileK,
                cheapVectorMinSize,
                transcendentalVectorMinSize,
                fusedCheapVectorMinSize,
                fusedTranscendentalVectorMinSize,
                reductionVectorMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                fusedCheapParallelMinSize,
                fusedTranscendentalParallelMinSize,
                reductionParallelMinSize,
                contiguousMaterializeThreshold,
                DEFAULT_LOW_COST_TARGET_CHUNKS_PER_WORKER,
                DEFAULT_MEDIUM_COST_TARGET_CHUNKS_PER_WORKER,
                DEFAULT_HIGH_COST_TARGET_CHUNKS_PER_WORKER,
                DEFAULT_MIN_SCALAR_CHUNK_SIZE,
                DEFAULT_MIN_VECTOR_CHUNK_SIZE,
                DEFAULT_MIN_REDUCTION_CHUNK_SIZE,
                DEFAULT_COMMON_POOL_LOW_COST_MAX_WORK_PER_WORKER,
                DEFAULT_FUSED_ASM_VECTOR_WIDTH,
                sumAccuracyMode,
                matMulParallelMinSize,
                attentionMatMulPolicy
        );
    }

    public CpuKernelConfig(
            int loopUnrollFactor,
            int matMulTileM,
            int matMulTileN,
            int matMulTileK,
            int cheapVectorMinSize,
            int transcendentalVectorMinSize,
            int reductionVectorMinSize,
            int cheapParallelMinSize,
            int transcendentalParallelMinSize,
            int reductionParallelMinSize,
            int contiguousMaterializeThreshold,
            int lowCostTargetChunksPerWorker,
            int mediumCostTargetChunksPerWorker,
            int highCostTargetChunksPerWorker,
            int minScalarChunkSize,
            int minVectorChunkSize,
            int minReductionChunkSize,
            int commonPoolLowCostMaxWorkPerWorker,
            int fusedAsmVectorWidth,
            SumAccuracyMode sumAccuracyMode,
            int matMulParallelMinSize,
            AttentionMatMulPolicy attentionMatMulPolicy
    ) {
        this(
                loopUnrollFactor,
                matMulTileM,
                matMulTileN,
                matMulTileK,
                cheapVectorMinSize,
                transcendentalVectorMinSize,
                cheapVectorMinSize,
                transcendentalVectorMinSize,
                reductionVectorMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                reductionParallelMinSize,
                contiguousMaterializeThreshold,
                lowCostTargetChunksPerWorker,
                mediumCostTargetChunksPerWorker,
                highCostTargetChunksPerWorker,
                minScalarChunkSize,
                minVectorChunkSize,
                minReductionChunkSize,
                commonPoolLowCostMaxWorkPerWorker,
                fusedAsmVectorWidth,
                sumAccuracyMode,
                matMulParallelMinSize,
                attentionMatMulPolicy
        );
    }

    public CpuKernelConfig(
            int loopUnrollFactor,
            int matMulTileM,
            int matMulTileN,
            int matMulTileK,
            int cheapVectorMinSize,
            int transcendentalVectorMinSize,
            int fusedCheapVectorMinSize,
            int fusedTranscendentalVectorMinSize,
            int reductionVectorMinSize,
            int cheapParallelMinSize,
            int transcendentalParallelMinSize,
            int fusedCheapParallelMinSize,
            int fusedTranscendentalParallelMinSize,
            int reductionParallelMinSize,
            int contiguousMaterializeThreshold,
            int lowCostTargetChunksPerWorker,
            int mediumCostTargetChunksPerWorker,
            int highCostTargetChunksPerWorker,
            int minScalarChunkSize,
            int minVectorChunkSize,
            int minReductionChunkSize,
            int commonPoolLowCostMaxWorkPerWorker,
            int fusedAsmVectorWidth,
            SumAccuracyMode sumAccuracyMode,
            int matMulParallelMinSize,
            AttentionMatMulPolicy attentionMatMulPolicy
    ) {
        this.loopUnrollFactor = loopUnrollFactor;
        this.matMulTileM = matMulTileM;
        this.matMulTileN = matMulTileN;
        this.matMulTileK = matMulTileK;
        this.cheapVectorMinSize = Math.max(1, cheapVectorMinSize);
        this.transcendentalVectorMinSize = Math.max(1, transcendentalVectorMinSize);
        this.fusedCheapVectorMinSize = Math.max(1, fusedCheapVectorMinSize);
        this.fusedTranscendentalVectorMinSize = Math.max(1, fusedTranscendentalVectorMinSize);
        this.reductionVectorMinSize = Math.max(1, reductionVectorMinSize);
        this.cheapParallelMinSize = Math.max(1, cheapParallelMinSize);
        this.transcendentalParallelMinSize = Math.max(1, transcendentalParallelMinSize);
        this.fusedCheapParallelMinSize = Math.max(1, fusedCheapParallelMinSize);
        this.fusedTranscendentalParallelMinSize = Math.max(1, fusedTranscendentalParallelMinSize);
        this.reductionParallelMinSize = Math.max(1, reductionParallelMinSize);
        this.matMulParallelMinSize = Math.max(1, matMulParallelMinSize);
        this.contiguousMaterializeThreshold = Math.max(0, contiguousMaterializeThreshold);
        this.lowCostTargetChunksPerWorker = Math.max(1, lowCostTargetChunksPerWorker);
        this.mediumCostTargetChunksPerWorker = Math.max(1, mediumCostTargetChunksPerWorker);
        this.highCostTargetChunksPerWorker = Math.max(1, highCostTargetChunksPerWorker);
        this.minScalarChunkSize = Math.max(1, minScalarChunkSize);
        this.minVectorChunkSize = Math.max(1, minVectorChunkSize);
        this.minReductionChunkSize = Math.max(1, minReductionChunkSize);
        this.commonPoolLowCostMaxWorkPerWorker = Math.max(1, commonPoolLowCostMaxWorkPerWorker);
        this.fusedAsmVectorWidth = normalizeFusedAsmVectorWidth(fusedAsmVectorWidth);
        this.sumAccuracyMode = sumAccuracyMode == null ? DEFAULT_SUM_ACCURACY_MODE : sumAccuracyMode;
        this.attentionMatMulPolicy = attentionMatMulPolicy == null ? DEFAULT_ATTENTION_MATMUL_POLICY : attentionMatMulPolicy;
    }

    public int loopUnrollFactor() { return loopUnrollFactor; }
    public int matMulTileM() { return matMulTileM; }
    public int matMulTileN() { return matMulTileN; }
    public int matMulTileK() { return matMulTileK; }
    public int cheapVectorMinSize() { return cheapVectorMinSize; }
    public int transcendentalVectorMinSize() { return transcendentalVectorMinSize; }
    public int fusedCheapVectorMinSize() { return fusedCheapVectorMinSize; }
    public int fusedTranscendentalVectorMinSize() { return fusedTranscendentalVectorMinSize; }
    public int reductionVectorMinSize() { return reductionVectorMinSize; }
    public int cheapParallelMinSize() { return cheapParallelMinSize; }
    public int transcendentalParallelMinSize() { return transcendentalParallelMinSize; }
    public int fusedCheapParallelMinSize() { return fusedCheapParallelMinSize; }
    public int fusedTranscendentalParallelMinSize() { return fusedTranscendentalParallelMinSize; }
    public int reductionParallelMinSize() { return reductionParallelMinSize; }
    public int parallelMinSize() { return cheapParallelMinSize; }
    public int matMulParallelMinSize() { return matMulParallelMinSize; }
    public int contiguousMaterializeThreshold() { return contiguousMaterializeThreshold; }
    public int lowCostTargetChunksPerWorker() { return lowCostTargetChunksPerWorker; }
    public int mediumCostTargetChunksPerWorker() { return mediumCostTargetChunksPerWorker; }
    public int highCostTargetChunksPerWorker() { return highCostTargetChunksPerWorker; }
    public int minScalarChunkSize() { return minScalarChunkSize; }
    public int minVectorChunkSize() { return minVectorChunkSize; }
    public int minReductionChunkSize() { return minReductionChunkSize; }
    public int commonPoolLowCostMaxWorkPerWorker() { return commonPoolLowCostMaxWorkPerWorker; }
    public int fusedAsmVectorWidth() { return fusedAsmVectorWidth; }
    public SumAccuracyMode sumAccuracyMode() { return sumAccuracyMode; }
    public AttentionMatMulPolicy attentionMatMulPolicy() { return attentionMatMulPolicy; }

    public static CpuKernelConfig defaultsTraining() {
        return new CpuKernelConfig(
                1, 16, 0, 0,
                DEFAULT_VECTOR_MIN_SIZE,
                DEFAULT_PARALLEL_MIN_SIZE,
                DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD,
                DEFAULT_SUM_ACCURACY_MODE
        );
    }

    public static CpuKernelConfig defaultsInference() {
        return new CpuKernelConfig(
                4, 32, 32, 32,
                DEFAULT_VECTOR_MIN_SIZE,
                DEFAULT_PARALLEL_MIN_SIZE,
                DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD,
                DEFAULT_SUM_ACCURACY_MODE
        );
    }

    private static int normalizeFusedAsmVectorWidth(int value) {
        if (value <= 1) {
            return 1;
        }
        if (value <= 2) {
            return 2;
        }
        if (value <= 4) {
            return 4;
        }
        return 8;
    }
}
