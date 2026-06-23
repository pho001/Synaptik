package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.linalg.attention.Cpu1AttentionKernel;
import backend.cpu1.kernels.linalg.attention.Cpu1AttentionKernelDispatch;
import backend.cpu1.kernels.linalg.attention.Cpu1AttentionKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable prepare-time contract for one dense cpu1 attention node.
 */
public final class Cpu1PreparedAttentionUnit {
    private final int nodeId;
    private final int queryNodeId;
    private final int keyNodeId;
    private final int valueNodeId;
    private final int maskNodeId;
    private final int attentionOutputNodeId;
    private final Operation.OpType opType;
    private final DataType dataType;
    private final Cpu1StorageKind storageKind;
    private final Cpu1VectorizationKind vectorizationKind;
    private final Cpu1AttentionKernelId kernelId;
    private final Cpu1AttentionKernel kernel;
    private final int[] queryShape;
    private final int[] keyShape;
    private final int[] valueShape;
    private final int[] maskShape;
    private final int[] outputShape;
    private final int[] scoresShape;
    private final int[] queryBatchOffsets;
    private final int[] keyBatchOffsets;
    private final int[] valueBatchOffsets;
    private final int[] maskBatchOffsets;
    private final int batchCount;
    private final int queryLen;
    private final int keyLen;
    private final int depth;
    private final int valueDim;
    private final int outputElementCount;
    private final double scale;
    private final boolean hasMask;
    private final boolean useFastExpApprox;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1LaunchPolicy launchPolicy;
    private final int scratchSlotCount;
    private final Cpu1StorageAccessPlan queryAccessPlan;
    private final Cpu1StorageAccessPlan keyAccessPlan;
    private final Cpu1StorageAccessPlan valueAccessPlan;
    private final Cpu1StorageAccessPlan maskAccessPlan;
    private final Cpu1StorageAccessPlan outputAccessPlan;

    public Cpu1PreparedAttentionUnit(
            int nodeId,
            int queryNodeId,
            int keyNodeId,
            int valueNodeId,
            int maskNodeId,
            int attentionOutputNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind,
            Cpu1AttentionKernelId kernelId,
            int[] queryShape,
            int[] keyShape,
            int[] valueShape,
            int[] maskShape,
            int[] outputShape,
            int[] scoresShape,
            int[] queryBatchOffsets,
            int[] keyBatchOffsets,
            int[] valueBatchOffsets,
            int[] maskBatchOffsets,
            int batchCount,
            int queryLen,
            int keyLen,
            int depth,
            int valueDim,
            int outputElementCount,
            double scale,
            boolean hasMask,
            boolean useFastExpApprox,
            Cpu1LaunchConfig launchConfig,
            Cpu1LaunchPolicy launchPolicy,
            int scratchSlotCount,
            Cpu1StorageAccessPlan queryAccessPlan,
            Cpu1StorageAccessPlan keyAccessPlan,
            Cpu1StorageAccessPlan valueAccessPlan,
            Cpu1StorageAccessPlan maskAccessPlan,
            Cpu1StorageAccessPlan outputAccessPlan
    ) {
        if (nodeId < 0) {
            throw new IllegalArgumentException("nodeId cannot be negative");
        }
        if (opType != Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION
                && opType != Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS) {
            throw new IllegalArgumentException("Cpu1PreparedAttentionUnit requires attention op, got " + opType);
        }
        this.nodeId = nodeId;
        this.queryNodeId = queryNodeId;
        this.keyNodeId = keyNodeId;
        this.valueNodeId = valueNodeId;
        this.maskNodeId = maskNodeId;
        this.attentionOutputNodeId = attentionOutputNodeId;
        this.opType = opType;
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        this.storageKind = Objects.requireNonNull(storageKind, "storageKind cannot be null");
        this.vectorizationKind = Objects.requireNonNull(vectorizationKind, "vectorizationKind cannot be null");
        this.kernelId = Objects.requireNonNull(kernelId, "kernelId cannot be null");
        this.kernel = Cpu1AttentionKernelDispatch.kernelFor(kernelId);
        this.queryShape = cloneOrEmpty(queryShape);
        this.keyShape = cloneOrEmpty(keyShape);
        this.valueShape = cloneOrEmpty(valueShape);
        this.maskShape = cloneOrEmpty(maskShape);
        this.outputShape = cloneOrEmpty(outputShape);
        this.scoresShape = cloneOrEmpty(scoresShape);
        this.queryBatchOffsets = cloneOrEmpty(queryBatchOffsets);
        this.keyBatchOffsets = cloneOrEmpty(keyBatchOffsets);
        this.valueBatchOffsets = cloneOrEmpty(valueBatchOffsets);
        this.maskBatchOffsets = cloneOrEmpty(maskBatchOffsets);
        this.batchCount = requireNonNegative(batchCount, "batchCount");
        this.queryLen = requireNonNegative(queryLen, "queryLen");
        this.keyLen = requireNonNegative(keyLen, "keyLen");
        this.depth = requireNonNegative(depth, "depth");
        this.valueDim = requireNonNegative(valueDim, "valueDim");
        this.outputElementCount = requireNonNegative(outputElementCount, "outputElementCount");
        this.scale = scale;
        this.hasMask = hasMask;
        this.useFastExpApprox = useFastExpApprox;
        this.launchConfig = Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        this.launchPolicy = Objects.requireNonNull(launchPolicy, "launchPolicy cannot be null");
        this.scratchSlotCount = requireNonNegative(scratchSlotCount, "scratchSlotCount");
        this.queryAccessPlan = queryAccessPlan;
        this.keyAccessPlan = keyAccessPlan;
        this.valueAccessPlan = valueAccessPlan;
        this.maskAccessPlan = maskAccessPlan;
        this.outputAccessPlan = Objects.requireNonNull(outputAccessPlan, "outputAccessPlan cannot be null");
        validateByOp();
    }

    public int nodeId() {
        return nodeId;
    }

    public int queryNodeId() {
        return queryNodeId;
    }

    public int keyNodeId() {
        return keyNodeId;
    }

    public int valueNodeId() {
        return valueNodeId;
    }

    public int maskNodeId() {
        return maskNodeId;
    }

    public int attentionOutputNodeId() {
        return attentionOutputNodeId;
    }

    public Operation.OpType opType() {
        return opType;
    }

    public DataType dataType() {
        return dataType;
    }

    public Cpu1StorageKind storageKind() {
        return storageKind;
    }

    public Cpu1VectorizationKind vectorizationKind() {
        return vectorizationKind;
    }

    public Cpu1AttentionKernelId kernelId() {
        return kernelId;
    }

    public Cpu1AttentionKernel kernel() {
        return kernel;
    }

    public int[] queryShape() {
        return queryShape.clone();
    }

    public int[] keyShape() {
        return keyShape.clone();
    }

    public int[] valueShape() {
        return valueShape.clone();
    }

    public int[] maskShape() {
        return maskShape.clone();
    }

    public int[] outputShape() {
        return outputShape.clone();
    }

    public int[] scoresShape() {
        return scoresShape.clone();
    }

    public int[] queryBatchOffsets() {
        return queryBatchOffsets.clone();
    }

    public int queryBatchOffset(int batch) {
        return queryBatchOffsets[batch];
    }

    public int[] keyBatchOffsets() {
        return keyBatchOffsets.clone();
    }

    public int keyBatchOffset(int batch) {
        return keyBatchOffsets[batch];
    }

    public int[] valueBatchOffsets() {
        return valueBatchOffsets.clone();
    }

    public int valueBatchOffset(int batch) {
        return valueBatchOffsets[batch];
    }

    public int[] maskBatchOffsets() {
        return maskBatchOffsets.clone();
    }

    public int maskBatchOffset(int batch) {
        return maskBatchOffsets[batch];
    }

    public int batchCount() {
        return batchCount;
    }

    public int queryLen() {
        return queryLen;
    }

    public int keyLen() {
        return keyLen;
    }

    public int depth() {
        return depth;
    }

    public int valueDim() {
        return valueDim;
    }

    public int outputElementCount() {
        return outputElementCount;
    }

    public double scale() {
        return scale;
    }

    public float scaleF32() {
        return (float) scale;
    }

    public boolean hasMask() {
        return hasMask;
    }

    public boolean useFastExpApprox() {
        return useFastExpApprox;
    }

    public Cpu1LaunchConfig launchConfig() {
        return launchConfig;
    }

    public Cpu1LaunchPolicy launchPolicy() {
        return launchPolicy;
    }

    public int scratchSlotCount() {
        return scratchSlotCount;
    }

    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        if (opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS) {
            return Cpu1ScratchBufferSpec.none();
        }
        int f32Scratch = dataType == DataType.FLOAT64 ? 0 : Math.multiplyExact(scratchSlotCount, keyLen);
        int f64Scratch = dataType == DataType.FLOAT64 ? Math.multiplyExact(scratchSlotCount, keyLen) : 0;
        return Cpu1ScratchBufferSpec.arrays(f32Scratch, f64Scratch, 0);
    }

    public Cpu1StorageAccessPlan queryAccessPlan() {
        return queryAccessPlan;
    }

    public Cpu1StorageAccessPlan keyAccessPlan() {
        return keyAccessPlan;
    }

    public Cpu1StorageAccessPlan valueAccessPlan() {
        return valueAccessPlan;
    }

    public Cpu1StorageAccessPlan maskAccessPlan() {
        return maskAccessPlan;
    }

    public Cpu1StorageAccessPlan outputAccessPlan() {
        return outputAccessPlan;
    }

    private void validateByOp() {
        if (opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS) {
            if (attentionOutputNodeId < 0) {
                throw new IllegalArgumentException("attention weights unit requires attentionOutputNodeId");
            }
            return;
        }
        if (queryNodeId < 0 || keyNodeId < 0 || valueNodeId < 0) {
            throw new IllegalArgumentException("attention unit requires q/k/v node ids");
        }
        if (hasMask && maskNodeId < 0) {
            throw new IllegalArgumentException("masked attention unit requires mask node id");
        }
        if (queryShape.length < 2 || keyShape.length < 2 || valueShape.length < 2
                || outputShape.length < 2 || scoresShape.length < 2) {
            throw new IllegalArgumentException("attention unit shapes must have rank >= 2");
        }
        if (queryLen <= 0 || keyLen <= 0 || depth <= 0 || valueDim <= 0 || batchCount <= 0) {
            throw new IllegalArgumentException("attention dimensions must be positive. qLen=" + queryLen
                    + ", keyLen=" + keyLen + ", depth=" + depth + ", valueDim=" + valueDim
                    + ", batchCount=" + batchCount);
        }
        if (queryBatchOffsets.length != batchCount
                || keyBatchOffsets.length != batchCount
                || valueBatchOffsets.length != batchCount) {
            throw new IllegalArgumentException("attention batch offset arrays must match batchCount=" + batchCount);
        }
        if (hasMask && maskBatchOffsets.length != batchCount) {
            throw new IllegalArgumentException("attention mask batch offsets must match batchCount=" + batchCount);
        }
        if (!hasMask && maskBatchOffsets.length != 0) {
            throw new IllegalArgumentException("unmasked attention cannot carry mask batch offsets");
        }
        if (!hasMask && maskShape.length != 0) {
            throw new IllegalArgumentException("unmasked attention cannot carry mask shape");
        }
        if (!(scale > 0.0d)) {
            throw new IllegalArgumentException("attention scale must be positive: " + scale);
        }
        if (scratchSlotCount <= 0) {
            throw new IllegalArgumentException("attention scratchSlotCount must be positive");
        }
    }

    private static int[] cloneOrEmpty(int[] values) {
        return values == null ? new int[0] : values.clone();
    }

    private static int requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " cannot be negative: " + value);
        }
        return value;
    }

    @Override
    public String toString() {
        return "Cpu1PreparedAttentionUnit{"
                + "nodeId=" + nodeId
                + ", opType=" + opType
                + ", dataType=" + dataType
                + ", storageKind=" + storageKind
                + ", vectorizationKind=" + vectorizationKind
                + ", outputShape=" + Arrays.toString(outputShape)
                + ", scoresShape=" + Arrays.toString(scoresShape)
                + '}';
    }
}
