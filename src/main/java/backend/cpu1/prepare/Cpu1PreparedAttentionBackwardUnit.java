package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.linalg.attention.backward.Cpu1AttentionBackwardKernel;
import backend.cpu1.kernels.linalg.attention.backward.Cpu1AttentionBackwardKernelDispatch;
import backend.cpu1.kernels.linalg.attention.backward.Cpu1AttentionBackwardKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import planning.region.specialization.SdpaBackwardOutputKind;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable prepare-time contract for one dense cpu1 SDPA backward specialized region.
 */
public final class Cpu1PreparedAttentionBackwardUnit {
    private final int nodeId;
    private final SdpaBackwardOutputKind outputKind;
    private final DataType dataType;
    private final Cpu1StorageKind storageKind;
    private final Cpu1VectorizationKind vectorizationKind;
    private final Cpu1AttentionBackwardKernelId kernelId;
    private final Cpu1AttentionBackwardKernel kernel;
    private final int weightsNodeId;
    private final int outGradNodeId;
    private final int queryNodeId;
    private final int keyNodeId;
    private final int valueNodeId;
    private final int maskNodeId;
    private final int[] weightsShape;
    private final int[] outGradShape;
    private final int[] queryShape;
    private final int[] keyShape;
    private final int[] valueShape;
    private final int[] maskShape;
    private final int[] outputShape;
    private final int batchCount;
    private final int queryLen;
    private final int keyLen;
    private final int depth;
    private final int valueDim;
    private final int outputElementCount;
    private final double scale;
    private final boolean hasMask;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1LaunchPolicy launchPolicy;
    private final int scratchSlotCount;
    private final Cpu1StorageAccessPlan weightsAccessPlan;
    private final Cpu1StorageAccessPlan outGradAccessPlan;
    private final Cpu1StorageAccessPlan queryAccessPlan;
    private final Cpu1StorageAccessPlan keyAccessPlan;
    private final Cpu1StorageAccessPlan valueAccessPlan;
    private final Cpu1StorageAccessPlan maskAccessPlan;
    private final Cpu1StorageAccessPlan outputAccessPlan;

    public Cpu1PreparedAttentionBackwardUnit(
            int nodeId,
            SdpaBackwardOutputKind outputKind,
            DataType dataType,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind,
            Cpu1AttentionBackwardKernelId kernelId,
            int weightsNodeId,
            int outGradNodeId,
            int queryNodeId,
            int keyNodeId,
            int valueNodeId,
            int maskNodeId,
            int[] weightsShape,
            int[] outGradShape,
            int[] queryShape,
            int[] keyShape,
            int[] valueShape,
            int[] maskShape,
            int[] outputShape,
            int batchCount,
            int queryLen,
            int keyLen,
            int depth,
            int valueDim,
            int outputElementCount,
            double scale,
            boolean hasMask,
            Cpu1LaunchConfig launchConfig,
            Cpu1LaunchPolicy launchPolicy,
            int scratchSlotCount,
            Cpu1StorageAccessPlan weightsAccessPlan,
            Cpu1StorageAccessPlan outGradAccessPlan,
            Cpu1StorageAccessPlan queryAccessPlan,
            Cpu1StorageAccessPlan keyAccessPlan,
            Cpu1StorageAccessPlan valueAccessPlan,
            Cpu1StorageAccessPlan maskAccessPlan,
            Cpu1StorageAccessPlan outputAccessPlan
    ) {
        if (nodeId < 0) {
            throw new IllegalArgumentException("nodeId cannot be negative");
        }
        this.nodeId = nodeId;
        this.outputKind = Objects.requireNonNull(outputKind, "outputKind cannot be null");
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        this.storageKind = Objects.requireNonNull(storageKind, "storageKind cannot be null");
        this.vectorizationKind = Objects.requireNonNull(vectorizationKind, "vectorizationKind cannot be null");
        this.kernelId = Objects.requireNonNull(kernelId, "kernelId cannot be null");
        this.kernel = Cpu1AttentionBackwardKernelDispatch.kernelFor(kernelId);
        this.weightsNodeId = requireNodeId(weightsNodeId, "weightsNodeId");
        this.outGradNodeId = requireNodeId(outGradNodeId, "outGradNodeId");
        this.queryNodeId = queryNodeId;
        this.keyNodeId = keyNodeId;
        this.valueNodeId = valueNodeId;
        this.maskNodeId = maskNodeId;
        this.weightsShape = cloneOrEmpty(weightsShape);
        this.outGradShape = cloneOrEmpty(outGradShape);
        this.queryShape = cloneOrEmpty(queryShape);
        this.keyShape = cloneOrEmpty(keyShape);
        this.valueShape = cloneOrEmpty(valueShape);
        this.maskShape = cloneOrEmpty(maskShape);
        this.outputShape = cloneOrEmpty(outputShape);
        this.batchCount = requirePositive(batchCount, "batchCount");
        this.queryLen = requirePositive(queryLen, "queryLen");
        this.keyLen = requirePositive(keyLen, "keyLen");
        this.depth = requirePositive(depth, "depth");
        this.valueDim = requirePositive(valueDim, "valueDim");
        this.outputElementCount = requireNonNegative(outputElementCount, "outputElementCount");
        if (!(scale > 0.0d)) {
            throw new IllegalArgumentException("scale must be positive: " + scale);
        }
        this.scale = scale;
        this.hasMask = hasMask;
        this.launchConfig = Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        this.launchPolicy = Objects.requireNonNull(launchPolicy, "launchPolicy cannot be null");
        this.scratchSlotCount = requireNonNegative(scratchSlotCount, "scratchSlotCount");
        this.weightsAccessPlan = Objects.requireNonNull(weightsAccessPlan, "weightsAccessPlan cannot be null");
        this.outGradAccessPlan = Objects.requireNonNull(outGradAccessPlan, "outGradAccessPlan cannot be null");
        this.queryAccessPlan = queryAccessPlan;
        this.keyAccessPlan = keyAccessPlan;
        this.valueAccessPlan = valueAccessPlan;
        this.maskAccessPlan = maskAccessPlan;
        this.outputAccessPlan = Objects.requireNonNull(outputAccessPlan, "outputAccessPlan cannot be null");
        validateByOutputKind();
    }

    public int nodeId() {
        return nodeId;
    }

    public SdpaBackwardOutputKind outputKind() {
        return outputKind;
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

    public Cpu1AttentionBackwardKernelId kernelId() {
        return kernelId;
    }

    public Cpu1AttentionBackwardKernel kernel() {
        return kernel;
    }

    public int weightsNodeId() {
        return weightsNodeId;
    }

    public int outGradNodeId() {
        return outGradNodeId;
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

    public int[] weightsShape() {
        return weightsShape.clone();
    }

    public int[] outGradShape() {
        return outGradShape.clone();
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

    public Cpu1LaunchConfig launchConfig() {
        return launchConfig;
    }

    public Cpu1LaunchPolicy launchPolicy() {
        return launchPolicy;
    }

    public int scratchSlotCount() {
        return scratchSlotCount;
    }

    public int rowCount() {
        return switch (outputKind) {
            case QUERY -> Math.multiplyExact(batchCount, queryLen);
            case KEY, VALUE -> Math.multiplyExact(batchCount, keyLen);
        };
    }

    public int scratchElementsPerSlot() {
        return switch (outputKind) {
            case VALUE -> 0;
            case QUERY -> Math.multiplyExact(keyLen, 2);
            case KEY -> Math.addExact(
                    Math.multiplyExact(keyLen, 2),
                    Math.multiplyExact(queryLen, keyLen)
            );
        };
    }

    public int dScoresScratchOffset(int slotIndex) {
        if (outputKind != SdpaBackwardOutputKind.KEY) {
            throw new IllegalStateException("dScores matrix scratch is used only by SDPA dK.");
        }
        return Math.addExact(
                Math.multiplyExact(slotIndex, scratchElementsPerSlot()),
                Math.multiplyExact(keyLen, 2)
        );
    }

    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        if (outputKind == SdpaBackwardOutputKind.VALUE) {
            return Cpu1ScratchBufferSpec.none();
        }
        int elements = Math.multiplyExact(scratchSlotCount, scratchElementsPerSlot());
        return dataType == DataType.FLOAT64
                ? Cpu1ScratchBufferSpec.arrays(0, elements, 0)
                : Cpu1ScratchBufferSpec.arrays(elements, 0, 0);
    }

    public Cpu1StorageAccessPlan weightsAccessPlan() {
        return weightsAccessPlan;
    }

    public Cpu1StorageAccessPlan outGradAccessPlan() {
        return outGradAccessPlan;
    }

    public boolean outGradDenseContiguousNoOffset() {
        return outGradAccessPlan.kind() == Cpu1StorageAccessKind.DENSE_CONTIGUOUS
                && outGradAccessPlan.storageOffset() == 0;
    }

    public boolean outGradBroadcastNoOffset() {
        return outGradAccessPlan.kind() == Cpu1StorageAccessKind.BROADCAST
                && outGradAccessPlan.storageOffset() == 0;
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

    private void validateByOutputKind() {
        if (weightsShape.length < 2 || outGradShape.length < 2 || outputShape.length < 2) {
            throw new IllegalArgumentException("SDPA backward weights/outGrad/output shapes must have rank >= 2");
        }
        if (outputKind == SdpaBackwardOutputKind.QUERY && (keyNodeId < 0 || valueNodeId < 0)) {
            throw new IllegalArgumentException("SDPA dQ unit requires key/value node ids");
        }
        if (outputKind == SdpaBackwardOutputKind.KEY && (queryNodeId < 0 || valueNodeId < 0)) {
            throw new IllegalArgumentException("SDPA dK unit requires query/value node ids");
        }
        if (outputKind == SdpaBackwardOutputKind.VALUE && (queryNodeId < -1 || keyNodeId < -1 || valueNodeId < -1)) {
            throw new IllegalArgumentException("SDPA dV optional q/k/v node ids must be >= -1");
        }
        if (hasMask && maskNodeId < 0) {
            throw new IllegalArgumentException("masked SDPA backward unit requires mask node id");
        }
        if (!hasMask && maskShape.length != 0) {
            throw new IllegalArgumentException("unmasked SDPA backward unit cannot carry mask shape");
        }
        if (outputKind != SdpaBackwardOutputKind.VALUE && scratchSlotCount <= 0) {
            throw new IllegalArgumentException("SDPA dQ/dK units require scratch slots");
        }
    }

    private static int[] cloneOrEmpty(int[] values) {
        return values == null ? new int[0] : values.clone();
    }

    private static int requireNodeId(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be >= 0");
        }
        return value;
    }

    private static int requirePositive(int value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be > 0: " + value);
        }
        return value;
    }

    private static int requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " cannot be negative: " + value);
        }
        return value;
    }

    @Override
    public String toString() {
        return "Cpu1PreparedAttentionBackwardUnit{"
                + "nodeId=" + nodeId
                + ", outputKind=" + outputKind
                + ", dataType=" + dataType
                + ", storageKind=" + storageKind
                + ", kernelId=" + kernelId
                + ", weightsShape=" + Arrays.toString(weightsShape)
                + ", outputShape=" + Arrays.toString(outputShape)
                + '}';
    }
}
