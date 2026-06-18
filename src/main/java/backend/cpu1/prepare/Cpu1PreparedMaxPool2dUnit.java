package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.kernels.nn.pool.maxpool.Cpu1MaxPool2dKernel;
import backend.cpu1.kernels.nn.pool.maxpool.Cpu1MaxPool2dKernelDispatch;
import backend.cpu1.kernels.nn.pool.maxpool.Cpu1MaxPool2dKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;
import tensor.options.Pool2dOptions;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable prepare-time contract for one dense cpu1 MAX_POOL2D node.
 */
public final class Cpu1PreparedMaxPool2dUnit {
    private final int nodeId;
    private final int inputNodeId;
    private final Operation.OpType opType;
    private final DataType dataType;
    private final Cpu1StorageKind storageKind;
    private final Cpu1MaxPool2dKernelId kernelId;
    private final Cpu1MaxPool2dKernel kernel;
    private final int[] inputShape;
    private final int[] outputShape;
    private final int batchCount;
    private final int channels;
    private final int inputH;
    private final int inputW;
    private final int outputH;
    private final int outputW;
    private final Pool2dOptions options;
    private final int outputElementCount;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1LaunchPolicy launchPolicy;
    private final Cpu1StorageAccessPlan inputAccessPlan;
    private final Cpu1StorageAccessPlan outputAccessPlan;

    public Cpu1PreparedMaxPool2dUnit(
            int nodeId,
            int inputNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1StorageKind storageKind,
            Cpu1MaxPool2dKernelId kernelId,
            int[] inputShape,
            int[] outputShape,
            Pool2dOptions options,
            int outputElementCount,
            Cpu1LaunchConfig launchConfig,
            Cpu1LaunchPolicy launchPolicy,
            Cpu1StorageAccessPlan inputAccessPlan,
            Cpu1StorageAccessPlan outputAccessPlan
    ) {
        if (nodeId < 0 || inputNodeId < 0) {
            throw new IllegalArgumentException("node ids cannot be negative");
        }
        if (opType != Operation.OpType.MAX_POOL2D) {
            throw new IllegalArgumentException("Cpu1PreparedMaxPool2dUnit requires MAX_POOL2D op, got " + opType);
        }
        Objects.requireNonNull(inputShape, "inputShape cannot be null");
        Objects.requireNonNull(outputShape, "outputShape cannot be null");
        if (inputShape.length != 4 || outputShape.length != 4) {
            throw new IllegalArgumentException("MAX_POOL2D requires rank-4 NCHW shapes. input="
                    + Arrays.toString(inputShape) + ", output=" + Arrays.toString(outputShape));
        }
        requirePositive(inputShape[0], "input batch");
        requirePositive(inputShape[1], "input channels");
        requirePositive(inputShape[2], "input height");
        requirePositive(inputShape[3], "input width");
        requirePositive(outputShape[0], "output batch");
        requirePositive(outputShape[1], "output channels");
        requirePositive(outputShape[2], "output height");
        requirePositive(outputShape[3], "output width");
        if (inputShape[0] != outputShape[0] || inputShape[1] != outputShape[1]) {
            throw new IllegalArgumentException("MAX_POOL2D input/output N and C dimensions must match. input="
                    + Arrays.toString(inputShape) + ", output=" + Arrays.toString(outputShape));
        }
        requireNonNegative(outputElementCount, "outputElementCount");
        this.nodeId = nodeId;
        this.inputNodeId = inputNodeId;
        this.opType = opType;
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        this.storageKind = Objects.requireNonNull(storageKind, "storageKind cannot be null");
        this.kernelId = Objects.requireNonNull(kernelId, "kernelId cannot be null");
        this.kernel = Cpu1MaxPool2dKernelDispatch.kernelFor(kernelId);
        this.inputShape = inputShape.clone();
        this.outputShape = outputShape.clone();
        this.batchCount = inputShape[0];
        this.channels = inputShape[1];
        this.inputH = inputShape[2];
        this.inputW = inputShape[3];
        this.outputH = outputShape[2];
        this.outputW = outputShape[3];
        this.options = Objects.requireNonNull(options, "options cannot be null");
        this.outputElementCount = outputElementCount;
        this.launchConfig = Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        this.launchPolicy = Objects.requireNonNull(launchPolicy, "launchPolicy cannot be null");
        this.inputAccessPlan = Objects.requireNonNull(inputAccessPlan, "inputAccessPlan cannot be null");
        this.outputAccessPlan = Objects.requireNonNull(outputAccessPlan, "outputAccessPlan cannot be null");
    }

    public int nodeId() {
        return nodeId;
    }

    public int inputNodeId() {
        return inputNodeId;
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

    public Cpu1MaxPool2dKernelId kernelId() {
        return kernelId;
    }

    public Cpu1MaxPool2dKernel kernel() {
        return kernel;
    }

    public int[] inputShape() {
        return inputShape.clone();
    }

    public int[] outputShape() {
        return outputShape.clone();
    }

    public int batchCount() {
        return batchCount;
    }

    public int channels() {
        return channels;
    }

    public int inputH() {
        return inputH;
    }

    public int inputW() {
        return inputW;
    }

    public int outputH() {
        return outputH;
    }

    public int outputW() {
        return outputW;
    }

    public Pool2dOptions options() {
        return options;
    }

    public int kernelH() {
        return options.kernelH();
    }

    public int kernelW() {
        return options.kernelW();
    }

    public int strideH() {
        return options.strideH();
    }

    public int strideW() {
        return options.strideW();
    }

    public int padH() {
        return options.padH();
    }

    public int padW() {
        return options.padW();
    }

    public int outputElementCount() {
        return outputElementCount;
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
