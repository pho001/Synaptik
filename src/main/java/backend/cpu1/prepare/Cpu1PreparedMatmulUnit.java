package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.matmul.Cpu1MatmulKernel;
import backend.cpu1.kernels.matmul.Cpu1MatmulKernelDispatch;
import backend.cpu1.kernels.matmul.Cpu1MatmulKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.provider.matmul.Cpu1MatmulRoute;
import backend.cpu1.storage.Cpu1StorageKind;
import tensor.DataType;

/**
 * Immutable prepare-time contract for one cpu1 matmul node.
 */
public final class Cpu1PreparedMatmulUnit {
    private final int nodeId;
    private final int leftNodeId;
    private final int rightNodeId;
    private final DataType dataType;
    private final Cpu1StorageKind storageKind;
    private final Cpu1MatmulRoute route;
    private final Cpu1MatmulPostOp postOp;
    private final int biasNodeId;
    private final int biasRowStride;
    private final int biasColStride;
    private final int[] biasBatchOffsets;
    private final Cpu1VectorizationKind vectorizationKind;
    private final Cpu1MatmulKernelId kernelId;
    private final Cpu1MatmulKernel kernel;
    private final int batchCount;
    private final int m;
    private final int n;
    private final int k;
    private final int leftRowStride;
    private final int leftColStride;
    private final int rightRowStride;
    private final int rightColStride;
    private final int outputRowStride;
    private final int outputColStride;
    private final int[] leftBatchOffsets;
    private final int[] rightBatchOffsets;
    private final int[] outputBatchOffsets;
    private final long work;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1ScratchBufferSpec scratchBufferSpec;
    private final int openBlasThreads;
    private final boolean openblasSgemmAvailable;
    private final boolean openblasDgemmAvailable;
    private final boolean openblasSbgemmAvailable;
    private final boolean openblasBgemmAvailable;
    private final String openblasLookupSource;
    private final String blasThreadPolicy;

    public Cpu1PreparedMatmulUnit(
            int nodeId,
            int leftNodeId,
            int rightNodeId,
            DataType dataType,
            Cpu1StorageKind storageKind,
            Cpu1MatmulRoute route,
            Cpu1MatmulPostOp postOp,
            Cpu1VectorizationKind vectorizationKind,
            Cpu1MatmulKernelId kernelId,
            int batchCount,
            int m,
            int n,
            int k,
            int leftRowStride,
            int leftColStride,
            int rightRowStride,
            int rightColStride,
            int outputRowStride,
            int outputColStride,
            int[] leftBatchOffsets,
            int[] rightBatchOffsets,
            int[] outputBatchOffsets,
            Cpu1LaunchConfig launchConfig,
            Cpu1ScratchBufferSpec scratchBufferSpec,
            int openBlasThreads,
            boolean openblasSgemmAvailable,
            boolean openblasDgemmAvailable,
            boolean openblasSbgemmAvailable,
            boolean openblasBgemmAvailable,
            String openblasLookupSource,
            String blasThreadPolicy
    ) {
        this(
                nodeId,
                leftNodeId,
                rightNodeId,
                dataType,
                storageKind,
                route,
                postOp,
                -1,
                0,
                0,
                null,
                vectorizationKind,
                kernelId,
                batchCount,
                m,
                n,
                k,
                leftRowStride,
                leftColStride,
                rightRowStride,
                rightColStride,
                outputRowStride,
                outputColStride,
                leftBatchOffsets,
                rightBatchOffsets,
                outputBatchOffsets,
                launchConfig,
                scratchBufferSpec,
                openBlasThreads,
                openblasSgemmAvailable,
                openblasDgemmAvailable,
                openblasSbgemmAvailable,
                openblasBgemmAvailable,
                openblasLookupSource,
                blasThreadPolicy
        );
    }

    public Cpu1PreparedMatmulUnit(
            int nodeId,
            int leftNodeId,
            int rightNodeId,
            DataType dataType,
            Cpu1StorageKind storageKind,
            Cpu1MatmulRoute route,
            Cpu1MatmulPostOp postOp,
            int biasNodeId,
            int biasRowStride,
            int biasColStride,
            int[] biasBatchOffsets,
            Cpu1VectorizationKind vectorizationKind,
            Cpu1MatmulKernelId kernelId,
            int batchCount,
            int m,
            int n,
            int k,
            int leftRowStride,
            int leftColStride,
            int rightRowStride,
            int rightColStride,
            int outputRowStride,
            int outputColStride,
            int[] leftBatchOffsets,
            int[] rightBatchOffsets,
            int[] outputBatchOffsets,
            Cpu1LaunchConfig launchConfig,
            Cpu1ScratchBufferSpec scratchBufferSpec,
            int openBlasThreads,
            boolean openblasSgemmAvailable,
            boolean openblasDgemmAvailable,
            boolean openblasSbgemmAvailable,
            boolean openblasBgemmAvailable,
            String openblasLookupSource,
            String blasThreadPolicy
    ) {
        if (nodeId < 0 || leftNodeId < 0 || rightNodeId < 0) {
            throw new IllegalArgumentException("node ids cannot be negative");
        }
        requirePositive(batchCount, "batchCount");
        requirePositive(m, "m");
        requirePositive(n, "n");
        requirePositive(k, "k");
        this.nodeId = nodeId;
        this.leftNodeId = leftNodeId;
        this.rightNodeId = rightNodeId;
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        if (storageKind == null) {
            throw new IllegalArgumentException("storageKind cannot be null");
        }
        if (route == null) {
            throw new IllegalArgumentException("route cannot be null");
        }
        if (postOp == null) {
            throw new IllegalArgumentException("postOp cannot be null");
        }
        this.dataType = dataType;
        this.storageKind = storageKind;
        this.route = route;
        this.postOp = postOp;
        if (!postOp.supportedBy(route)) {
            throw new UnsupportedOperationException("cpu1 " + route + " MATMUL does not support post-op " + postOp);
        }
        if (postOp.requiresBias() && biasNodeId < 0) {
            throw new IllegalArgumentException("cpu1 " + postOp + " MATMUL requires a bias node id");
        }
        if (!postOp.requiresBias() && biasNodeId >= 0) {
            throw new IllegalArgumentException("cpu1 " + postOp + " MATMUL does not accept a bias node id");
        }
        this.biasNodeId = biasNodeId;
        this.biasRowStride = biasRowStride;
        this.biasColStride = biasColStride;
        this.biasBatchOffsets = postOp.requiresBias()
                ? requireOffsets(biasBatchOffsets, batchCount, "biasBatchOffsets")
                : new int[batchCount];
        if (vectorizationKind == null) {
            throw new IllegalArgumentException("vectorizationKind cannot be null");
        }
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        this.vectorizationKind = vectorizationKind;
        this.kernelId = kernelId;
        this.kernel = Cpu1MatmulKernelDispatch.kernelFor(kernelId);
        this.batchCount = batchCount;
        this.m = m;
        this.n = n;
        this.k = k;
        this.leftRowStride = leftRowStride;
        this.leftColStride = leftColStride;
        this.rightRowStride = rightRowStride;
        this.rightColStride = rightColStride;
        this.outputRowStride = outputRowStride;
        this.outputColStride = outputColStride;
        this.leftBatchOffsets = requireOffsets(leftBatchOffsets, batchCount, "leftBatchOffsets");
        this.rightBatchOffsets = requireOffsets(rightBatchOffsets, batchCount, "rightBatchOffsets");
        this.outputBatchOffsets = requireOffsets(outputBatchOffsets, batchCount, "outputBatchOffsets");
        this.work = Math.multiplyExact(Math.multiplyExact(Math.multiplyExact((long) batchCount, m), n), k);
        if (launchConfig == null) {
            throw new IllegalArgumentException("launchConfig cannot be null");
        }
        if (scratchBufferSpec == null) {
            throw new IllegalArgumentException("scratchBufferSpec cannot be null");
        }
        this.launchConfig = launchConfig;
        this.scratchBufferSpec = scratchBufferSpec;
        if (openBlasThreads < 0) {
            throw new IllegalArgumentException("openBlasThreads must be non-negative: " + openBlasThreads);
        }
        this.openBlasThreads = openBlasThreads;
        this.openblasSgemmAvailable = openblasSgemmAvailable;
        this.openblasDgemmAvailable = openblasDgemmAvailable;
        this.openblasSbgemmAvailable = openblasSbgemmAvailable;
        this.openblasBgemmAvailable = openblasBgemmAvailable;
        this.openblasLookupSource = requireText(openblasLookupSource, "openblasLookupSource");
        this.blasThreadPolicy = requireText(blasThreadPolicy, "blasThreadPolicy");
    }

    public int nodeId() {
        return nodeId;
    }

    public int leftNodeId() {
        return leftNodeId;
    }

    public int rightNodeId() {
        return rightNodeId;
    }

    public DataType dataType() {
        return dataType;
    }

    public Cpu1StorageKind storageKind() {
        return storageKind;
    }

    public Cpu1MatmulRoute route() {
        return route;
    }

    public Cpu1MatmulPostOp postOp() {
        return postOp;
    }

    public boolean hasBias() {
        return postOp.requiresBias();
    }

    public int biasNodeId() {
        if (!hasBias()) {
            throw new IllegalStateException("This cpu1 matmul unit does not have a bias input");
        }
        return biasNodeId;
    }

    public int biasRowStride() {
        return biasRowStride;
    }

    public int biasColStride() {
        return biasColStride;
    }

    public Cpu1VectorizationKind vectorizationKind() {
        return vectorizationKind;
    }

    public Cpu1MatmulKernelId kernelId() {
        return kernelId;
    }

    public Cpu1MatmulKernel kernel() {
        return kernel;
    }

    public int batchCount() {
        return batchCount;
    }

    public int m() {
        return m;
    }

    public int n() {
        return n;
    }

    public int k() {
        return k;
    }

    public int leftRowStride() {
        return leftRowStride;
    }

    public int leftColStride() {
        return leftColStride;
    }

    public int rightRowStride() {
        return rightRowStride;
    }

    public int rightColStride() {
        return rightColStride;
    }

    public int outputRowStride() {
        return outputRowStride;
    }

    public int outputColStride() {
        return outputColStride;
    }

    public int leftBatchOffset(int batch) {
        return leftBatchOffsets[batch];
    }

    public int rightBatchOffset(int batch) {
        return rightBatchOffsets[batch];
    }

    public int outputBatchOffset(int batch) {
        return outputBatchOffsets[batch];
    }

    public int biasBatchOffset(int batch) {
        return biasBatchOffsets[batch];
    }

    public long work() {
        return work;
    }

    public Cpu1LaunchConfig launchConfig() {
        return launchConfig;
    }

    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return scratchBufferSpec;
    }

    public int openBlasThreads() {
        return openBlasThreads;
    }

    public boolean openblasSgemmAvailable() {
        return openblasSgemmAvailable;
    }

    public boolean openblasDgemmAvailable() {
        return openblasDgemmAvailable;
    }

    public boolean openblasSbgemmAvailable() {
        return openblasSbgemmAvailable;
    }

    public boolean openblasBgemmAvailable() {
        return openblasBgemmAvailable;
    }

    public String openblasLookupSource() {
        return openblasLookupSource;
    }

    public String blasThreadPolicy() {
        return blasThreadPolicy;
    }

    private static int[] requireOffsets(int[] offsets, int batchCount, String name) {
        if (offsets == null) {
            throw new IllegalArgumentException(name + " cannot be null");
        }
        if (offsets.length != batchCount) {
            throw new IllegalArgumentException(name + " length " + offsets.length
                    + " does not match batchCount " + batchCount);
        }
        return offsets.clone();
    }

    private static void requirePositive(int value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive: " + value);
        }
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " cannot be blank");
        }
        return value;
    }
}
