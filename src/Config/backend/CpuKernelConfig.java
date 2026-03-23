package Config.backend;

public final class CpuKernelConfig {
    private static final int DEFAULT_VECTOR_MIN_SIZE = 1_024;
    private static final int DEFAULT_PARALLEL_MIN_SIZE = 100_000;
    private static final int DEFAULT_PARALLELISM = 0;
    private static final int DEFAULT_CHUNKS_PER_WORKER = 4;
    private static final int DEFAULT_MIN_CHUNK_SIZE = 4_096;
    private static final int DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD = 1_000_000_000;
    private static final SumAccuracyMode DEFAULT_SUM_ACCURACY_MODE = SumAccuracyMode.FAST;
    private static final double DEFAULT_LOW_COST_NS_PER_ELEMENT_THRESHOLD = 2.0d;
    private static final VectorPolicy DEFAULT_VECTOR_POLICY = VectorPolicy.AUTO;

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
    private final SumAccuracyMode sumAccuracyMode;
    private final double lowCostNsPerElementThreshold;
    private final VectorPolicy vectorPolicyCheap;
    private final VectorPolicy vectorPolicyTranscendental;
    private final VectorPolicy vectorPolicyReduction;

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
                contiguousMaterializeThreshold,
                DEFAULT_SUM_ACCURACY_MODE
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
            int contiguousMaterializeThreshold,
            SumAccuracyMode sumAccuracyMode
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
                contiguousMaterializeThreshold,
                sumAccuracyMode,
                DEFAULT_LOW_COST_NS_PER_ELEMENT_THRESHOLD
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
            int contiguousMaterializeThreshold,
            SumAccuracyMode sumAccuracyMode,
            double lowCostNsPerElementThreshold
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
                contiguousMaterializeThreshold,
                sumAccuracyMode,
                lowCostNsPerElementThreshold,
                DEFAULT_VECTOR_POLICY,
                DEFAULT_VECTOR_POLICY,
                DEFAULT_VECTOR_POLICY
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
            int contiguousMaterializeThreshold,
            SumAccuracyMode sumAccuracyMode,
            double lowCostNsPerElementThreshold,
            VectorPolicy vectorPolicyCheap,
            VectorPolicy vectorPolicyTranscendental,
            VectorPolicy vectorPolicyReduction
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
        this.sumAccuracyMode = sumAccuracyMode == null ? DEFAULT_SUM_ACCURACY_MODE : sumAccuracyMode;
        this.lowCostNsPerElementThreshold = lowCostNsPerElementThreshold <= 0.0d
                ? DEFAULT_LOW_COST_NS_PER_ELEMENT_THRESHOLD
                : lowCostNsPerElementThreshold;
        this.vectorPolicyCheap = vectorPolicyCheap == null ? DEFAULT_VECTOR_POLICY : vectorPolicyCheap;
        this.vectorPolicyTranscendental = vectorPolicyTranscendental == null ? DEFAULT_VECTOR_POLICY : vectorPolicyTranscendental;
        this.vectorPolicyReduction = vectorPolicyReduction == null ? DEFAULT_VECTOR_POLICY : vectorPolicyReduction;
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

    public SumAccuracyMode sumAccuracyMode() {
        return sumAccuracyMode;
    }

    public double lowCostNsPerElementThreshold() {
        return lowCostNsPerElementThreshold;
    }

    public VectorPolicy vectorPolicyCheap() {
        return vectorPolicyCheap;
    }

    public VectorPolicy vectorPolicyTranscendental() {
        return vectorPolicyTranscendental;
    }

    public VectorPolicy vectorPolicyReduction() {
        return vectorPolicyReduction;
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
                DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD,
                DEFAULT_SUM_ACCURACY_MODE,
                DEFAULT_LOW_COST_NS_PER_ELEMENT_THRESHOLD
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
                DEFAULT_CONTIGUOUS_MATERIALIZE_THRESHOLD,
                DEFAULT_SUM_ACCURACY_MODE,
                DEFAULT_LOW_COST_NS_PER_ELEMENT_THRESHOLD
        );
    }
}
