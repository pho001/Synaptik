package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.kernels.nn.conv.conv2d.Cpu1Conv2dKernel;
import backend.cpu1.kernels.nn.conv.conv2d.Cpu1Conv2dKernelDispatch;
import backend.cpu1.kernels.nn.conv.conv2d.Cpu1Conv2dKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;
import tensor.options.Conv2dOptions;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable prepare-time contract for one dense direct cpu1 CONV2D node.
 */
public final class Cpu1PreparedConv2dUnit {
    private final int nodeId;
    private final int inputNodeId;
    private final int weightNodeId;
    private final int biasNodeId;
    private final Operation.OpType opType;
    private final DataType dataType;
    private final Cpu1StorageKind storageKind;
    private final Cpu1Conv2dKernelId kernelId;
    private final Cpu1Conv2dKernel kernel;
    private final int[] inputShape;
    private final int[] weightShape;
    private final int[] biasShape;
    private final int[] outputShape;
    private final boolean hasBias;
    private final int batchCount;
    private final int inChannels;
    private final int outChannels;
    private final int channelsPerGroup;
    private final int outChannelsPerGroup;
    private final int inputH;
    private final int inputW;
    private final int kernelH;
    private final int kernelW;
    private final int outputH;
    private final int outputW;
    private final Conv2dOptions options;
    private final long work;
    private final int outputElementCount;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1LaunchPolicy launchPolicy;
    private final Cpu1StorageAccessPlan inputAccessPlan;
    private final Cpu1StorageAccessPlan weightAccessPlan;
    private final Cpu1StorageAccessPlan biasAccessPlan;
    private final Cpu1StorageAccessPlan outputAccessPlan;

    public Cpu1PreparedConv2dUnit(
            int nodeId,
            int inputNodeId,
            int weightNodeId,
            int biasNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1StorageKind storageKind,
            Cpu1Conv2dKernelId kernelId,
            int[] inputShape,
            int[] weightShape,
            int[] biasShape,
            int[] outputShape,
            boolean hasBias,
            Conv2dOptions options,
            long work,
            int outputElementCount,
            Cpu1LaunchConfig launchConfig,
            Cpu1LaunchPolicy launchPolicy,
            Cpu1StorageAccessPlan inputAccessPlan,
            Cpu1StorageAccessPlan weightAccessPlan,
            Cpu1StorageAccessPlan biasAccessPlan,
            Cpu1StorageAccessPlan outputAccessPlan
    ) {
        if (nodeId < 0 || inputNodeId < 0 || weightNodeId < 0 || (hasBias && biasNodeId < 0)) {
            throw new IllegalArgumentException("node ids cannot be negative");
        }
        if (opType != Operation.OpType.CONV2D) {
            throw new IllegalArgumentException("Cpu1PreparedConv2dUnit requires CONV2D op, got " + opType);
        }
        Objects.requireNonNull(inputShape, "inputShape cannot be null");
        Objects.requireNonNull(weightShape, "weightShape cannot be null");
        Objects.requireNonNull(outputShape, "outputShape cannot be null");
        if (inputShape.length != 4 || weightShape.length != 4 || outputShape.length != 4) {
            throw new IllegalArgumentException("CONV2D requires rank-4 NCHW/OIHW shapes. input="
                    + Arrays.toString(inputShape) + ", weight=" + Arrays.toString(weightShape)
                    + ", output=" + Arrays.toString(outputShape));
        }
        requirePositive(inputShape[0], "input batch");
        requirePositive(inputShape[1], "input channels");
        requirePositive(inputShape[2], "input height");
        requirePositive(inputShape[3], "input width");
        requirePositive(weightShape[0], "weight output channels");
        requirePositive(weightShape[1], "weight channels per group");
        requirePositive(weightShape[2], "weight kernel height");
        requirePositive(weightShape[3], "weight kernel width");
        requirePositive(outputShape[0], "output batch");
        requirePositive(outputShape[1], "output channels");
        requirePositive(outputShape[2], "output height");
        requirePositive(outputShape[3], "output width");
        if (hasBias) {
            Objects.requireNonNull(biasShape, "biasShape cannot be null when hasBias=true");
            if (biasShape.length != 1 || biasShape[0] != weightShape[0]) {
                throw new IllegalArgumentException("CONV2D bias must have shape [outChannels]. bias="
                        + Arrays.toString(biasShape) + ", outChannels=" + weightShape[0]);
            }
        }
        if (inputShape[0] != outputShape[0] || weightShape[0] != outputShape[1]) {
            throw new IllegalArgumentException("CONV2D output N/C dimensions mismatch. input="
                    + Arrays.toString(inputShape) + ", weight=" + Arrays.toString(weightShape)
                    + ", output=" + Arrays.toString(outputShape));
        }
        int groups = Objects.requireNonNull(options, "options cannot be null").groups();
        if (inputShape[1] % groups != 0 || weightShape[0] % groups != 0
                || weightShape[1] * groups != inputShape[1]) {
            throw new IllegalArgumentException("CONV2D grouped channel contract mismatch. input="
                    + Arrays.toString(inputShape) + ", weight=" + Arrays.toString(weightShape)
                    + ", groups=" + groups);
        }
        requireNonNegative(work, "work");
        requireNonNegative(outputElementCount, "outputElementCount");
        this.nodeId = nodeId;
        this.inputNodeId = inputNodeId;
        this.weightNodeId = weightNodeId;
        this.biasNodeId = hasBias ? biasNodeId : -1;
        this.opType = opType;
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        this.storageKind = Objects.requireNonNull(storageKind, "storageKind cannot be null");
        this.kernelId = Objects.requireNonNull(kernelId, "kernelId cannot be null");
        this.kernel = Cpu1Conv2dKernelDispatch.kernelFor(kernelId);
        this.inputShape = inputShape.clone();
        this.weightShape = weightShape.clone();
        this.biasShape = biasShape == null ? null : biasShape.clone();
        this.outputShape = outputShape.clone();
        this.hasBias = hasBias;
        this.batchCount = inputShape[0];
        this.inChannels = inputShape[1];
        this.outChannels = weightShape[0];
        this.channelsPerGroup = weightShape[1];
        this.outChannelsPerGroup = weightShape[0] / groups;
        this.inputH = inputShape[2];
        this.inputW = inputShape[3];
        this.kernelH = weightShape[2];
        this.kernelW = weightShape[3];
        this.outputH = outputShape[2];
        this.outputW = outputShape[3];
        this.options = options;
        this.work = work;
        this.outputElementCount = outputElementCount;
        this.launchConfig = Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        this.launchPolicy = Objects.requireNonNull(launchPolicy, "launchPolicy cannot be null");
        this.inputAccessPlan = Objects.requireNonNull(inputAccessPlan, "inputAccessPlan cannot be null");
        this.weightAccessPlan = Objects.requireNonNull(weightAccessPlan, "weightAccessPlan cannot be null");
        this.biasAccessPlan = biasAccessPlan;
        this.outputAccessPlan = Objects.requireNonNull(outputAccessPlan, "outputAccessPlan cannot be null");
    }

    public int nodeId() {
        return nodeId;
    }

    public int inputNodeId() {
        return inputNodeId;
    }

    public int weightNodeId() {
        return weightNodeId;
    }

    public int biasNodeId() {
        return biasNodeId;
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

    public Cpu1Conv2dKernelId kernelId() {
        return kernelId;
    }

    public Cpu1Conv2dKernel kernel() {
        return kernel;
    }

    public int[] inputShape() {
        return inputShape.clone();
    }

    public int[] weightShape() {
        return weightShape.clone();
    }

    public int[] biasShape() {
        return biasShape == null ? null : biasShape.clone();
    }

    public int[] outputShape() {
        return outputShape.clone();
    }

    public boolean hasBias() {
        return hasBias;
    }

    public int batchCount() {
        return batchCount;
    }

    public int inChannels() {
        return inChannels;
    }

    public int outChannels() {
        return outChannels;
    }

    public int channelsPerGroup() {
        return channelsPerGroup;
    }

    public int outChannelsPerGroup() {
        return outChannelsPerGroup;
    }

    public int inputH() {
        return inputH;
    }

    public int inputW() {
        return inputW;
    }

    public int kernelH() {
        return kernelH;
    }

    public int kernelW() {
        return kernelW;
    }

    public int outputH() {
        return outputH;
    }

    public int outputW() {
        return outputW;
    }

    public Conv2dOptions options() {
        return options;
    }

    public long work() {
        return work;
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

    public int dilationH() {
        return options.dilationH();
    }

    public int dilationW() {
        return options.dilationW();
    }

    public int groups() {
        return options.groups();
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

    public Cpu1StorageAccessPlan weightAccessPlan() {
        return weightAccessPlan;
    }

    public Cpu1StorageAccessPlan biasAccessPlan() {
        return biasAccessPlan;
    }

    public Cpu1StorageAccessPlan outputAccessPlan() {
        return outputAccessPlan;
    }

    private static void requirePositive(int value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be > 0, got " + value);
        }
    }

    private static void requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be >= 0, got " + value);
        }
    }

    private static void requireNonNegative(long value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be >= 0, got " + value);
        }
    }
}
