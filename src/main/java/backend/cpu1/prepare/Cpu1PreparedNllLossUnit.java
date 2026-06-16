package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.kernels.loss.nll.Cpu1NllLossKernel;
import backend.cpu1.kernels.loss.nll.Cpu1NllLossKernelDispatch;
import backend.cpu1.kernels.loss.nll.Cpu1NllLossKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

public final class Cpu1PreparedNllLossUnit {
    private final int nodeId;
    private final int logProbsNodeId;
    private final int targetsNodeId;
    private final Operation.OpType opType;
    private final DataType dataType;
    private final Cpu1StorageKind storageKind;
    private final Cpu1NllLossKernelId kernelId;
    private final Cpu1NllLossKernel kernel;
    private final int classAxis;
    private final int axisSize;
    private final int axisStride;
    private final int groupCount;
    private final int[] logProbsShape;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1LaunchPolicy launchPolicy;
    private final Cpu1ScratchBufferSpec scratchBufferSpec;

    public Cpu1PreparedNllLossUnit(
            int nodeId,
            int logProbsNodeId,
            int targetsNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1StorageKind storageKind,
            Cpu1NllLossKernelId kernelId,
            int classAxis,
            int axisSize,
            int axisStride,
            int groupCount,
            int[] logProbsShape,
            Cpu1LaunchConfig launchConfig,
            Cpu1LaunchPolicy launchPolicy,
            Cpu1ScratchBufferSpec scratchBufferSpec
    ) {
        if (nodeId < 0 || logProbsNodeId < 0 || targetsNodeId < 0) {
            throw new IllegalArgumentException("node ids cannot be negative");
        }
        requireNonNegative(axisSize, "axisSize");
        requirePositive(axisStride, "axisStride");
        requireNonNegative(groupCount, "groupCount");
        this.nodeId = nodeId;
        this.logProbsNodeId = logProbsNodeId;
        this.targetsNodeId = targetsNodeId;
        this.opType = Objects.requireNonNull(opType, "opType cannot be null");
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        this.storageKind = Objects.requireNonNull(storageKind, "storageKind cannot be null");
        this.kernelId = Objects.requireNonNull(kernelId, "kernelId cannot be null");
        this.kernel = Cpu1NllLossKernelDispatch.kernelFor(kernelId);
        this.classAxis = classAxis;
        this.axisSize = axisSize;
        this.axisStride = axisStride;
        this.groupCount = groupCount;
        this.logProbsShape = Objects.requireNonNull(logProbsShape, "logProbsShape cannot be null").clone();
        this.launchConfig = Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        this.launchPolicy = Objects.requireNonNull(launchPolicy, "launchPolicy cannot be null");
        this.scratchBufferSpec = Objects.requireNonNull(scratchBufferSpec, "scratchBufferSpec cannot be null");
        if (classAxis < 0 || classAxis >= this.logProbsShape.length) {
            throw new IllegalArgumentException("classAxis out of bounds: " + classAxis
                    + " for logProbs shape " + Arrays.toString(this.logProbsShape));
        }
    }

    public int nodeId() {
        return nodeId;
    }

    public int logProbsNodeId() {
        return logProbsNodeId;
    }

    public int targetsNodeId() {
        return targetsNodeId;
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

    public Cpu1NllLossKernelId kernelId() {
        return kernelId;
    }

    public Cpu1NllLossKernel kernel() {
        return kernel;
    }

    public int classAxis() {
        return classAxis;
    }

    public int axisSize() {
        return axisSize;
    }

    public int axisStride() {
        return axisStride;
    }

    public int groupCount() {
        return groupCount;
    }

    public int[] logProbsShape() {
        return logProbsShape.clone();
    }

    public Cpu1LaunchConfig launchConfig() {
        return launchConfig;
    }

    public Cpu1LaunchPolicy launchPolicy() {
        return launchPolicy;
    }

    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return scratchBufferSpec;
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
