package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.kernels.nn.normalization.layernorm.Cpu1LayerNormKernel;
import backend.cpu1.kernels.nn.normalization.layernorm.Cpu1LayerNormKernelDispatch;
import backend.cpu1.kernels.nn.normalization.layernorm.Cpu1LayerNormKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable prepare-time contract for one dense cpu1 LayerNorm node.
 */
public final class Cpu1PreparedLayerNormUnit {
    private final int nodeId;
    private final int inputNodeId;
    private final int gammaNodeId;
    private final int betaNodeId;
    private final Operation.OpType opType;
    private final DataType dataType;
    private final Cpu1StorageKind storageKind;
    private final Cpu1LayerNormKernelId kernelId;
    private final Cpu1LayerNormKernel kernel;
    private final int[] inputShape;
    private final int normalizedRank;
    private final int normalizedSize;
    private final int groupCount;
    private final int outputElementCount;
    private final double epsilon;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1LaunchPolicy launchPolicy;
    private final Cpu1StorageAccessPlan inputAccessPlan;
    private final Cpu1StorageAccessPlan gammaAccessPlan;
    private final Cpu1StorageAccessPlan betaAccessPlan;
    private final Cpu1StorageAccessPlan outputAccessPlan;

    public Cpu1PreparedLayerNormUnit(
            int nodeId,
            int inputNodeId,
            int gammaNodeId,
            int betaNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1StorageKind storageKind,
            Cpu1LayerNormKernelId kernelId,
            int[] inputShape,
            int normalizedRank,
            int normalizedSize,
            int groupCount,
            int outputElementCount,
            double epsilon,
            Cpu1LaunchConfig launchConfig,
            Cpu1LaunchPolicy launchPolicy,
            Cpu1StorageAccessPlan inputAccessPlan,
            Cpu1StorageAccessPlan gammaAccessPlan,
            Cpu1StorageAccessPlan betaAccessPlan,
            Cpu1StorageAccessPlan outputAccessPlan
    ) {
        if (nodeId < 0 || inputNodeId < 0 || gammaNodeId < 0 || betaNodeId < 0) {
            throw new IllegalArgumentException("node ids cannot be negative");
        }
        if (opType != Operation.OpType.LAYER_NORM) {
            throw new IllegalArgumentException("Cpu1PreparedLayerNormUnit requires LAYER_NORM op, got " + opType);
        }
        if (inputShape == null || inputShape.length == 0) {
            throw new IllegalArgumentException("inputShape must have rank >= 1");
        }
        if (normalizedRank < 1 || normalizedRank > inputShape.length) {
            throw new IllegalArgumentException("normalizedRank out of bounds: " + normalizedRank
                    + " for input shape " + Arrays.toString(inputShape));
        }
        requirePositive(normalizedSize, "normalizedSize");
        requireNonNegative(groupCount, "groupCount");
        requireNonNegative(outputElementCount, "outputElementCount");
        if (!(epsilon > 0.0d)) {
            throw new IllegalArgumentException("epsilon must be > 0: " + epsilon);
        }
        this.nodeId = nodeId;
        this.inputNodeId = inputNodeId;
        this.gammaNodeId = gammaNodeId;
        this.betaNodeId = betaNodeId;
        this.opType = opType;
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        this.storageKind = Objects.requireNonNull(storageKind, "storageKind cannot be null");
        this.kernelId = Objects.requireNonNull(kernelId, "kernelId cannot be null");
        this.kernel = Cpu1LayerNormKernelDispatch.kernelFor(kernelId);
        this.inputShape = inputShape.clone();
        this.normalizedRank = normalizedRank;
        this.normalizedSize = normalizedSize;
        this.groupCount = groupCount;
        this.outputElementCount = outputElementCount;
        this.epsilon = epsilon;
        this.launchConfig = Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        this.launchPolicy = Objects.requireNonNull(launchPolicy, "launchPolicy cannot be null");
        this.inputAccessPlan = Objects.requireNonNull(inputAccessPlan, "inputAccessPlan cannot be null");
        this.gammaAccessPlan = Objects.requireNonNull(gammaAccessPlan, "gammaAccessPlan cannot be null");
        this.betaAccessPlan = Objects.requireNonNull(betaAccessPlan, "betaAccessPlan cannot be null");
        this.outputAccessPlan = Objects.requireNonNull(outputAccessPlan, "outputAccessPlan cannot be null");
    }

    public int nodeId() {
        return nodeId;
    }

    public int inputNodeId() {
        return inputNodeId;
    }

    public int gammaNodeId() {
        return gammaNodeId;
    }

    public int betaNodeId() {
        return betaNodeId;
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

    public Cpu1LayerNormKernelId kernelId() {
        return kernelId;
    }

    public Cpu1LayerNormKernel kernel() {
        return kernel;
    }

    public int[] inputShape() {
        return inputShape.clone();
    }

    public int normalizedRank() {
        return normalizedRank;
    }

    public int normalizedSize() {
        return normalizedSize;
    }

    public int groupCount() {
        return groupCount;
    }

    public int outputElementCount() {
        return outputElementCount;
    }

    public double epsilon() {
        return epsilon;
    }

    public float epsilonF32() {
        return (float) epsilon;
    }

    public Cpu1LaunchConfig launchConfig() {
        return launchConfig;
    }

    public Cpu1LaunchPolicy launchPolicy() {
        return launchPolicy;
    }

    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return Cpu1ScratchBufferSpec.none();
    }

    public Cpu1StorageAccessPlan inputAccessPlan() {
        return inputAccessPlan;
    }

    public Cpu1StorageAccessPlan gammaAccessPlan() {
        return gammaAccessPlan;
    }

    public Cpu1StorageAccessPlan betaAccessPlan() {
        return betaAccessPlan;
    }

    public Cpu1StorageAccessPlan outputAccessPlan() {
        return outputAccessPlan;
    }

    private static void requirePositive(int value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive: " + value);
        }
    }

    private static void requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " cannot be negative: " + value);
        }
    }
}
