package config.backend;

public final class CpuKernelConfig {
    private static final int DEFAULT_VECTOR_MIN_SIZE = 1_024;
    private static final int DEFAULT_PARALLEL_MIN_SIZE = 100_000;
    private static final int DEFAULT_MATMUL_PARALLEL_MIN_SIZE = 2_000_000;
    private static final int DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD = 1_000_000_000;
    private static final int DEFAULT_CHEAP_F64_MATERIALIZE_THRESHOLD = DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD;
    private static final int DEFAULT_CHEAP_F32_MATERIALIZE_THRESHOLD = DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD;
    private static final int DEFAULT_CHEAP_BF16_MATERIALIZE_THRESHOLD = DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD;
    private static final int DEFAULT_WHERE_MATERIALIZE_THRESHOLD = DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD;
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
    private static final CpuMatMulMicroKernel DEFAULT_MATMUL_MICRO_KERNEL = CpuMatMulMicroKernel.AUTO;
    private static final CpuMatMulMicroKernel DEFAULT_ATTENTION_MATMUL_MICRO_KERNEL = CpuMatMulMicroKernel.AUTO;

    private final int loopUnrollFactor;
    private final int matMulTileM;
    private final int matMulTileN;
    private final int matMulTileK;
    private final int attentionMatMulTileM;
    private final int attentionMatMulTileN;
    private final int attentionMatMulTileK;
    private final int cheapVectorMinSize;
    private final int transcendentalVectorMinSize;
    private final int fusedCheapVectorMinSize;
    private final int fusedTranscendentalVectorMinSize;
    private final int reductionVectorMinSize;
    private final int attentionVectorMinSize;
    private final int cheapParallelMinSize;
    private final int transcendentalParallelMinSize;
    private final int fusedCheapParallelMinSize;
    private final int fusedTranscendentalParallelMinSize;
    private final int reductionParallelMinSize;
    private final int attentionParallelMinSize;
    private final int matMulParallelMinSize;
    private final int contiguousMaterializeThreshold;
    private final int cheapF64MaterializeThreshold;
    private final int cheapF32MaterializeThreshold;
    private final int cheapBF16MaterializeThreshold;
    private final int whereMaterializeThreshold;
    private final int lowCostTargetChunksPerWorker;
    private final int mediumCostTargetChunksPerWorker;
    private final int highCostTargetChunksPerWorker;
    private final int minScalarChunkSize;
    private final int minVectorChunkSize;
    private final int minReductionChunkSize;
    private final int commonPoolLowCostMaxWorkPerWorker;
    private final int fusedCheapContiguousAsmVectorWidth;
    private final int fusedCheapStridedAsmVectorWidth;
    private final int fusedNonCheapContiguousAsmVectorWidth;
    private final int fusedNonCheapStridedAsmVectorWidth;
    private final SumAccuracyMode sumAccuracyMode;
    private final AttentionMatMulPolicy attentionMatMulPolicy;
    private final CpuMatMulMicroKernel matMulMicroKernel;
    private final CpuMatMulMicroKernel attentionMatMulMicroKernel;

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
                vectorMinSize,
                parallelMinSize,
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
                DEFAULT_FUSED_ASM_VECTOR_WIDTH,
                DEFAULT_FUSED_ASM_VECTOR_WIDTH,
                DEFAULT_FUSED_ASM_VECTOR_WIDTH,
                sumAccuracyMode,
                DEFAULT_MATMUL_PARALLEL_MIN_SIZE,
                DEFAULT_ATTENTION_MATMUL_POLICY,
                DEFAULT_MATMUL_MICRO_KERNEL
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
            int attentionVectorMinSize,
            int cheapParallelMinSize,
            int transcendentalParallelMinSize,
            int fusedCheapParallelMinSize,
            int fusedTranscendentalParallelMinSize,
            int reductionParallelMinSize,
            int attentionParallelMinSize,
            int contiguousMaterializeThreshold,
            int lowCostTargetChunksPerWorker,
            int mediumCostTargetChunksPerWorker,
            int highCostTargetChunksPerWorker,
            int minScalarChunkSize,
            int minVectorChunkSize,
            int minReductionChunkSize,
            int commonPoolLowCostMaxWorkPerWorker,
            int fusedCheapContiguousAsmVectorWidth,
            int fusedCheapStridedAsmVectorWidth,
            int fusedNonCheapContiguousAsmVectorWidth,
            int fusedNonCheapStridedAsmVectorWidth,
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
                reductionVectorMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                fusedCheapParallelMinSize,
                fusedTranscendentalParallelMinSize,
                reductionParallelMinSize,
                reductionParallelMinSize,
                contiguousMaterializeThreshold,
                lowCostTargetChunksPerWorker,
                mediumCostTargetChunksPerWorker,
                highCostTargetChunksPerWorker,
                minScalarChunkSize,
                minVectorChunkSize,
                minReductionChunkSize,
                commonPoolLowCostMaxWorkPerWorker,
                fusedCheapContiguousAsmVectorWidth,
                fusedCheapStridedAsmVectorWidth,
                fusedNonCheapContiguousAsmVectorWidth,
                fusedNonCheapStridedAsmVectorWidth,
                sumAccuracyMode,
                matMulParallelMinSize,
                attentionMatMulPolicy,
                DEFAULT_MATMUL_MICRO_KERNEL
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
            int attentionVectorMinSize,
            int cheapParallelMinSize,
            int transcendentalParallelMinSize,
            int fusedCheapParallelMinSize,
            int fusedTranscendentalParallelMinSize,
            int reductionParallelMinSize,
            int attentionParallelMinSize,
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
                reductionVectorMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                fusedCheapParallelMinSize,
                fusedTranscendentalParallelMinSize,
                reductionParallelMinSize,
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
                DEFAULT_FUSED_ASM_VECTOR_WIDTH,
                DEFAULT_FUSED_ASM_VECTOR_WIDTH,
                DEFAULT_FUSED_ASM_VECTOR_WIDTH,
                sumAccuracyMode,
                matMulParallelMinSize,
                attentionMatMulPolicy,
                DEFAULT_MATMUL_MICRO_KERNEL
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
                reductionVectorMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                reductionParallelMinSize,
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
                fusedAsmVectorWidth,
                fusedAsmVectorWidth,
                fusedAsmVectorWidth,
                sumAccuracyMode,
                matMulParallelMinSize,
                attentionMatMulPolicy,
                DEFAULT_MATMUL_MICRO_KERNEL
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
                reductionVectorMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                fusedCheapParallelMinSize,
                fusedTranscendentalParallelMinSize,
                reductionParallelMinSize,
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
                fusedAsmVectorWidth,
                fusedAsmVectorWidth,
                fusedAsmVectorWidth,
                sumAccuracyMode,
                matMulParallelMinSize,
                attentionMatMulPolicy,
                DEFAULT_MATMUL_MICRO_KERNEL
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
            int fusedCheapContiguousAsmVectorWidth,
            int fusedCheapStridedAsmVectorWidth,
            int fusedNonCheapContiguousAsmVectorWidth,
            int fusedNonCheapStridedAsmVectorWidth,
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
                reductionVectorMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                fusedCheapParallelMinSize,
                fusedTranscendentalParallelMinSize,
                reductionParallelMinSize,
                reductionParallelMinSize,
                contiguousMaterializeThreshold,
                lowCostTargetChunksPerWorker,
                mediumCostTargetChunksPerWorker,
                highCostTargetChunksPerWorker,
                minScalarChunkSize,
                minVectorChunkSize,
                minReductionChunkSize,
                commonPoolLowCostMaxWorkPerWorker,
                fusedCheapContiguousAsmVectorWidth,
                fusedCheapStridedAsmVectorWidth,
                fusedNonCheapContiguousAsmVectorWidth,
                fusedNonCheapStridedAsmVectorWidth,
                sumAccuracyMode,
                matMulParallelMinSize,
                attentionMatMulPolicy,
                DEFAULT_MATMUL_MICRO_KERNEL
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
            int fusedCheapContiguousAsmVectorWidth,
            int fusedCheapStridedAsmVectorWidth,
            int fusedNonCheapContiguousAsmVectorWidth,
            int fusedNonCheapStridedAsmVectorWidth,
            SumAccuracyMode sumAccuracyMode,
            int matMulParallelMinSize,
            AttentionMatMulPolicy attentionMatMulPolicy,
            CpuMatMulMicroKernel matMulMicroKernel
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
                reductionVectorMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                fusedCheapParallelMinSize,
                fusedTranscendentalParallelMinSize,
                reductionParallelMinSize,
                reductionParallelMinSize,
                contiguousMaterializeThreshold,
                lowCostTargetChunksPerWorker,
                mediumCostTargetChunksPerWorker,
                highCostTargetChunksPerWorker,
                minScalarChunkSize,
                minVectorChunkSize,
                minReductionChunkSize,
                commonPoolLowCostMaxWorkPerWorker,
                fusedCheapContiguousAsmVectorWidth,
                fusedCheapStridedAsmVectorWidth,
                fusedNonCheapContiguousAsmVectorWidth,
                fusedNonCheapStridedAsmVectorWidth,
                sumAccuracyMode,
                matMulParallelMinSize,
                attentionMatMulPolicy,
                matMulMicroKernel
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
            int attentionVectorMinSize,
            int cheapParallelMinSize,
            int transcendentalParallelMinSize,
            int fusedCheapParallelMinSize,
            int fusedTranscendentalParallelMinSize,
            int reductionParallelMinSize,
            int attentionParallelMinSize,
            int contiguousMaterializeThreshold,
            int lowCostTargetChunksPerWorker,
            int mediumCostTargetChunksPerWorker,
            int highCostTargetChunksPerWorker,
            int minScalarChunkSize,
            int minVectorChunkSize,
            int minReductionChunkSize,
            int commonPoolLowCostMaxWorkPerWorker,
            int fusedCheapContiguousAsmVectorWidth,
            int fusedCheapStridedAsmVectorWidth,
            int fusedNonCheapContiguousAsmVectorWidth,
            int fusedNonCheapStridedAsmVectorWidth,
            SumAccuracyMode sumAccuracyMode,
            int matMulParallelMinSize,
            AttentionMatMulPolicy attentionMatMulPolicy,
            CpuMatMulMicroKernel matMulMicroKernel
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
                attentionVectorMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                fusedCheapParallelMinSize,
                fusedTranscendentalParallelMinSize,
                reductionParallelMinSize,
                attentionParallelMinSize,
                contiguousMaterializeThreshold,
                lowCostTargetChunksPerWorker,
                mediumCostTargetChunksPerWorker,
                highCostTargetChunksPerWorker,
                minScalarChunkSize,
                minVectorChunkSize,
                minReductionChunkSize,
                commonPoolLowCostMaxWorkPerWorker,
                fusedCheapContiguousAsmVectorWidth,
                fusedCheapStridedAsmVectorWidth,
                fusedNonCheapContiguousAsmVectorWidth,
                fusedNonCheapStridedAsmVectorWidth,
                sumAccuracyMode,
                matMulParallelMinSize,
                attentionMatMulPolicy,
                matMulMicroKernel,
                matMulMicroKernel
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
            int attentionVectorMinSize,
            int cheapParallelMinSize,
            int transcendentalParallelMinSize,
            int fusedCheapParallelMinSize,
            int fusedTranscendentalParallelMinSize,
            int reductionParallelMinSize,
            int attentionParallelMinSize,
            int contiguousMaterializeThreshold,
            int lowCostTargetChunksPerWorker,
            int mediumCostTargetChunksPerWorker,
            int highCostTargetChunksPerWorker,
            int minScalarChunkSize,
            int minVectorChunkSize,
            int minReductionChunkSize,
            int commonPoolLowCostMaxWorkPerWorker,
            int fusedCheapContiguousAsmVectorWidth,
            int fusedCheapStridedAsmVectorWidth,
            int fusedNonCheapContiguousAsmVectorWidth,
            int fusedNonCheapStridedAsmVectorWidth,
            SumAccuracyMode sumAccuracyMode,
            int matMulParallelMinSize,
            AttentionMatMulPolicy attentionMatMulPolicy,
            CpuMatMulMicroKernel matMulMicroKernel,
            CpuMatMulMicroKernel attentionMatMulMicroKernel
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
                attentionVectorMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                fusedCheapParallelMinSize,
                fusedTranscendentalParallelMinSize,
                reductionParallelMinSize,
                attentionParallelMinSize,
                contiguousMaterializeThreshold,
                contiguousMaterializeThreshold,
                contiguousMaterializeThreshold,
                contiguousMaterializeThreshold,
                contiguousMaterializeThreshold,
                lowCostTargetChunksPerWorker,
                mediumCostTargetChunksPerWorker,
                highCostTargetChunksPerWorker,
                minScalarChunkSize,
                minVectorChunkSize,
                minReductionChunkSize,
                commonPoolLowCostMaxWorkPerWorker,
                fusedCheapContiguousAsmVectorWidth,
                fusedCheapStridedAsmVectorWidth,
                fusedNonCheapContiguousAsmVectorWidth,
                fusedNonCheapStridedAsmVectorWidth,
                sumAccuracyMode,
                matMulParallelMinSize,
                attentionMatMulPolicy,
                matMulMicroKernel,
                attentionMatMulMicroKernel,
                matMulTileM,
                matMulTileN,
                matMulTileK
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
            int attentionVectorMinSize,
            int cheapParallelMinSize,
            int transcendentalParallelMinSize,
            int fusedCheapParallelMinSize,
            int fusedTranscendentalParallelMinSize,
            int reductionParallelMinSize,
            int attentionParallelMinSize,
            int contiguousMaterializeThreshold,
            int lowCostTargetChunksPerWorker,
            int mediumCostTargetChunksPerWorker,
            int highCostTargetChunksPerWorker,
            int minScalarChunkSize,
            int minVectorChunkSize,
            int minReductionChunkSize,
            int commonPoolLowCostMaxWorkPerWorker,
            int fusedCheapContiguousAsmVectorWidth,
            int fusedCheapStridedAsmVectorWidth,
            int fusedNonCheapContiguousAsmVectorWidth,
            int fusedNonCheapStridedAsmVectorWidth,
            SumAccuracyMode sumAccuracyMode,
            int matMulParallelMinSize,
            AttentionMatMulPolicy attentionMatMulPolicy,
            CpuMatMulMicroKernel matMulMicroKernel,
            CpuMatMulMicroKernel attentionMatMulMicroKernel,
            int attentionMatMulTileM,
            int attentionMatMulTileN,
            int attentionMatMulTileK
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
                attentionVectorMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                fusedCheapParallelMinSize,
                fusedTranscendentalParallelMinSize,
                reductionParallelMinSize,
                attentionParallelMinSize,
                contiguousMaterializeThreshold,
                contiguousMaterializeThreshold,
                contiguousMaterializeThreshold,
                contiguousMaterializeThreshold,
                contiguousMaterializeThreshold,
                lowCostTargetChunksPerWorker,
                mediumCostTargetChunksPerWorker,
                highCostTargetChunksPerWorker,
                minScalarChunkSize,
                minVectorChunkSize,
                minReductionChunkSize,
                commonPoolLowCostMaxWorkPerWorker,
                fusedCheapContiguousAsmVectorWidth,
                fusedCheapStridedAsmVectorWidth,
                fusedNonCheapContiguousAsmVectorWidth,
                fusedNonCheapStridedAsmVectorWidth,
                sumAccuracyMode,
                matMulParallelMinSize,
                attentionMatMulPolicy,
                matMulMicroKernel,
                attentionMatMulMicroKernel,
                attentionMatMulTileM,
                attentionMatMulTileN,
                attentionMatMulTileK
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
            int attentionVectorMinSize,
            int cheapParallelMinSize,
            int transcendentalParallelMinSize,
            int fusedCheapParallelMinSize,
            int fusedTranscendentalParallelMinSize,
            int reductionParallelMinSize,
            int attentionParallelMinSize,
            int contiguousMaterializeThreshold,
            int cheapF64MaterializeThreshold,
            int cheapF32MaterializeThreshold,
            int cheapBF16MaterializeThreshold,
            int whereMaterializeThreshold,
            int lowCostTargetChunksPerWorker,
            int mediumCostTargetChunksPerWorker,
            int highCostTargetChunksPerWorker,
            int minScalarChunkSize,
            int minVectorChunkSize,
            int minReductionChunkSize,
            int commonPoolLowCostMaxWorkPerWorker,
            int fusedCheapContiguousAsmVectorWidth,
            int fusedCheapStridedAsmVectorWidth,
            int fusedNonCheapContiguousAsmVectorWidth,
            int fusedNonCheapStridedAsmVectorWidth,
            SumAccuracyMode sumAccuracyMode,
            int matMulParallelMinSize,
            AttentionMatMulPolicy attentionMatMulPolicy,
            CpuMatMulMicroKernel matMulMicroKernel,
            CpuMatMulMicroKernel attentionMatMulMicroKernel,
            int attentionMatMulTileM,
            int attentionMatMulTileN,
            int attentionMatMulTileK
    ) {
        this.loopUnrollFactor = loopUnrollFactor;
        this.matMulTileM = matMulTileM;
        this.matMulTileN = matMulTileN;
        this.matMulTileK = matMulTileK;
        this.attentionMatMulTileM = attentionMatMulTileM <= 0 ? matMulTileM : attentionMatMulTileM;
        this.attentionMatMulTileN = attentionMatMulTileN <= 0 ? matMulTileN : attentionMatMulTileN;
        this.attentionMatMulTileK = attentionMatMulTileK <= 0 ? matMulTileK : attentionMatMulTileK;
        this.cheapVectorMinSize = Math.max(1, cheapVectorMinSize);
        this.transcendentalVectorMinSize = Math.max(1, transcendentalVectorMinSize);
        this.fusedCheapVectorMinSize = Math.max(1, fusedCheapVectorMinSize);
        this.fusedTranscendentalVectorMinSize = Math.max(1, fusedTranscendentalVectorMinSize);
        this.reductionVectorMinSize = Math.max(1, reductionVectorMinSize);
        this.attentionVectorMinSize = Math.max(1, attentionVectorMinSize);
        this.cheapParallelMinSize = Math.max(1, cheapParallelMinSize);
        this.transcendentalParallelMinSize = Math.max(1, transcendentalParallelMinSize);
        this.fusedCheapParallelMinSize = Math.max(1, fusedCheapParallelMinSize);
        this.fusedTranscendentalParallelMinSize = Math.max(1, fusedTranscendentalParallelMinSize);
        this.reductionParallelMinSize = Math.max(1, reductionParallelMinSize);
        this.attentionParallelMinSize = Math.max(1, attentionParallelMinSize);
        this.matMulParallelMinSize = Math.max(1, matMulParallelMinSize);
        this.contiguousMaterializeThreshold = Math.max(0, contiguousMaterializeThreshold);
        this.cheapF64MaterializeThreshold = Math.max(0, cheapF64MaterializeThreshold);
        this.cheapF32MaterializeThreshold = Math.max(0, cheapF32MaterializeThreshold);
        this.cheapBF16MaterializeThreshold = Math.max(0, cheapBF16MaterializeThreshold);
        this.whereMaterializeThreshold = Math.max(0, whereMaterializeThreshold);
        this.lowCostTargetChunksPerWorker = Math.max(1, lowCostTargetChunksPerWorker);
        this.mediumCostTargetChunksPerWorker = Math.max(1, mediumCostTargetChunksPerWorker);
        this.highCostTargetChunksPerWorker = Math.max(1, highCostTargetChunksPerWorker);
        this.minScalarChunkSize = Math.max(1, minScalarChunkSize);
        this.minVectorChunkSize = Math.max(1, minVectorChunkSize);
        this.minReductionChunkSize = Math.max(1, minReductionChunkSize);
        this.commonPoolLowCostMaxWorkPerWorker = Math.max(1, commonPoolLowCostMaxWorkPerWorker);
        this.fusedCheapContiguousAsmVectorWidth = normalizeFusedAsmVectorWidth(fusedCheapContiguousAsmVectorWidth);
        this.fusedCheapStridedAsmVectorWidth = normalizeFusedAsmVectorWidth(fusedCheapStridedAsmVectorWidth);
        this.fusedNonCheapContiguousAsmVectorWidth = normalizeFusedAsmVectorWidth(fusedNonCheapContiguousAsmVectorWidth);
        this.fusedNonCheapStridedAsmVectorWidth = normalizeFusedAsmVectorWidth(fusedNonCheapStridedAsmVectorWidth);
        this.sumAccuracyMode = sumAccuracyMode == null ? DEFAULT_SUM_ACCURACY_MODE : sumAccuracyMode;
        this.attentionMatMulPolicy = attentionMatMulPolicy == null ? DEFAULT_ATTENTION_MATMUL_POLICY : attentionMatMulPolicy;
        this.matMulMicroKernel = matMulMicroKernel == null ? DEFAULT_MATMUL_MICRO_KERNEL : matMulMicroKernel;
        this.attentionMatMulMicroKernel = attentionMatMulMicroKernel == null
                ? this.matMulMicroKernel
                : attentionMatMulMicroKernel;
    }

    public int loopUnrollFactor() { return loopUnrollFactor; }
    public int matMulTileM() { return matMulTileM; }
    public int matMulTileN() { return matMulTileN; }
    public int matMulTileK() { return matMulTileK; }
    public int attentionMatMulTileM() { return attentionMatMulTileM; }
    public int attentionMatMulTileN() { return attentionMatMulTileN; }
    public int attentionMatMulTileK() { return attentionMatMulTileK; }
    public int cheapVectorMinSize() { return cheapVectorMinSize; }
    public int transcendentalVectorMinSize() { return transcendentalVectorMinSize; }
    public int fusedCheapVectorMinSize() { return fusedCheapVectorMinSize; }
    public int fusedTranscendentalVectorMinSize() { return fusedTranscendentalVectorMinSize; }
    public int reductionVectorMinSize() { return reductionVectorMinSize; }
    public int attentionVectorMinSize() { return attentionVectorMinSize; }
    public int cheapParallelMinSize() { return cheapParallelMinSize; }
    public int transcendentalParallelMinSize() { return transcendentalParallelMinSize; }
    public int fusedCheapParallelMinSize() { return fusedCheapParallelMinSize; }
    public int fusedTranscendentalParallelMinSize() { return fusedTranscendentalParallelMinSize; }
    public int reductionParallelMinSize() { return reductionParallelMinSize; }
    public int attentionParallelMinSize() { return attentionParallelMinSize; }
    public int parallelMinSize() { return cheapParallelMinSize; }
    public int matMulParallelMinSize() { return matMulParallelMinSize; }
    public int contiguousMaterializeThreshold() { return contiguousMaterializeThreshold; }
    public int cheapF64MaterializeThreshold() { return cheapF64MaterializeThreshold; }
    public int cheapF32MaterializeThreshold() { return cheapF32MaterializeThreshold; }
    public int cheapBF16MaterializeThreshold() { return cheapBF16MaterializeThreshold; }
    public int whereMaterializeThreshold() { return whereMaterializeThreshold; }
    public int lowCostTargetChunksPerWorker() { return lowCostTargetChunksPerWorker; }
    public int mediumCostTargetChunksPerWorker() { return mediumCostTargetChunksPerWorker; }
    public int highCostTargetChunksPerWorker() { return highCostTargetChunksPerWorker; }
    public int minScalarChunkSize() { return minScalarChunkSize; }
    public int minVectorChunkSize() { return minVectorChunkSize; }
    public int minReductionChunkSize() { return minReductionChunkSize; }
    public int commonPoolLowCostMaxWorkPerWorker() { return commonPoolLowCostMaxWorkPerWorker; }
    public int fusedCheapContiguousAsmVectorWidth() { return fusedCheapContiguousAsmVectorWidth; }
    public int fusedCheapStridedAsmVectorWidth() { return fusedCheapStridedAsmVectorWidth; }
    public int fusedNonCheapContiguousAsmVectorWidth() { return fusedNonCheapContiguousAsmVectorWidth; }
    public int fusedNonCheapStridedAsmVectorWidth() { return fusedNonCheapStridedAsmVectorWidth; }
    public int fusedAsmVectorWidth() { return fusedCheapContiguousAsmVectorWidth; }
    public SumAccuracyMode sumAccuracyMode() { return sumAccuracyMode; }
    public AttentionMatMulPolicy attentionMatMulPolicy() { return attentionMatMulPolicy; }
    public CpuMatMulMicroKernel matMulMicroKernel() { return matMulMicroKernel; }
    public CpuMatMulMicroKernel attentionMatMulMicroKernel() {
        return attentionMatMulMicroKernel == null ? DEFAULT_ATTENTION_MATMUL_MICRO_KERNEL : attentionMatMulMicroKernel;
    }

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
