package backend.cpu1.prepare;

import backend.cpu1.kernels.index.Cpu1IndexKernel;
import backend.cpu1.kernels.index.Cpu1IndexKernelDispatch;
import backend.cpu1.kernels.index.Cpu1IndexKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

import java.util.List;
import java.util.Objects;

/**
 * Immutable prepare-time contract for one cpu1 index node.
 */
public final class Cpu1PreparedIndexUnit {
    private final int nodeId;
    private final int inputNodeId;
    private final int indexNodeId;
    private final Operation.OpType opType;
    private final DataType valueDataType;
    private final DataType indexDataType;
    private final Cpu1StorageKind storageKind;
    private final Cpu1IndexKernelId kernelId;
    private final Cpu1IndexKernel kernel;
    private final int dimension;
    private final int axisSize;
    private final int innerSize;
    private final int outerSize;
    private final int outputElementCount;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1LaunchPolicy launchPolicy;
    private final Cpu1StorageAccessPlan inputAccessPlan;
    private final Cpu1StorageAccessPlan indexAccessPlan;
    private final Cpu1StorageAccessPlan outputAccessPlan;

    public Cpu1PreparedIndexUnit(
            int nodeId,
            int inputNodeId,
            int indexNodeId,
            Operation.OpType opType,
            DataType valueDataType,
            DataType indexDataType,
            Cpu1StorageKind storageKind,
            Cpu1IndexKernelId kernelId,
            int dimension,
            int axisSize,
            int innerSize,
            int outerSize,
            int outputElementCount,
            Cpu1LaunchConfig launchConfig,
            Cpu1LaunchPolicy launchPolicy,
            Cpu1StorageAccessPlan inputAccessPlan,
            Cpu1StorageAccessPlan indexAccessPlan,
            Cpu1StorageAccessPlan outputAccessPlan
    ) {
        if (nodeId < 0) {
            throw new IllegalArgumentException("nodeId cannot be negative");
        }
        if (inputNodeId < 0) {
            throw new IllegalArgumentException("inputNodeId cannot be negative");
        }
        if (indexNodeId < 0) {
            throw new IllegalArgumentException("indexNodeId cannot be negative");
        }
        if (dimension < 0) {
            throw new IllegalArgumentException("dimension cannot be negative");
        }
        requireNonNegative(axisSize, "axisSize");
        requireNonNegative(innerSize, "innerSize");
        requireNonNegative(outerSize, "outerSize");
        requireNonNegative(outputElementCount, "outputElementCount");
        this.nodeId = nodeId;
        this.inputNodeId = inputNodeId;
        this.indexNodeId = indexNodeId;
        this.opType = Objects.requireNonNull(opType, "opType cannot be null");
        this.valueDataType = Objects.requireNonNull(valueDataType, "valueDataType cannot be null");
        this.indexDataType = Objects.requireNonNull(indexDataType, "indexDataType cannot be null");
        this.storageKind = Objects.requireNonNull(storageKind, "storageKind cannot be null");
        this.kernelId = Objects.requireNonNull(kernelId, "kernelId cannot be null");
        this.kernel = Cpu1IndexKernelDispatch.kernelFor(kernelId);
        this.dimension = dimension;
        this.axisSize = axisSize;
        this.innerSize = innerSize;
        this.outerSize = outerSize;
        this.outputElementCount = outputElementCount;
        this.launchConfig = Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        this.launchPolicy = Objects.requireNonNull(launchPolicy, "launchPolicy cannot be null");
        this.inputAccessPlan = Objects.requireNonNull(inputAccessPlan, "inputAccessPlan cannot be null");
        this.indexAccessPlan = Objects.requireNonNull(indexAccessPlan, "indexAccessPlan cannot be null");
        this.outputAccessPlan = Objects.requireNonNull(outputAccessPlan, "outputAccessPlan cannot be null");
    }

    public int nodeId() {
        return nodeId;
    }

    public int inputNodeId() {
        return inputNodeId;
    }

    public int indexNodeId() {
        return indexNodeId;
    }

    public List<Integer> inputNodeIds() {
        return List.of(inputNodeId, indexNodeId);
    }

    public Operation.OpType opType() {
        return opType;
    }

    public DataType valueDataType() {
        return valueDataType;
    }

    public DataType indexDataType() {
        return indexDataType;
    }

    public Cpu1StorageKind storageKind() {
        return storageKind;
    }

    public Cpu1IndexKernelId kernelId() {
        return kernelId;
    }

    public Cpu1IndexKernel kernel() {
        return kernel;
    }

    public int dimension() {
        return dimension;
    }

    public int axisSize() {
        return axisSize;
    }

    public int innerSize() {
        return innerSize;
    }

    public int outerSize() {
        return outerSize;
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

    public Cpu1StorageAccessPlan inputAccessPlan() {
        return inputAccessPlan;
    }

    public Cpu1StorageAccessPlan indexAccessPlan() {
        return indexAccessPlan;
    }

    public Cpu1StorageAccessPlan outputAccessPlan() {
        return outputAccessPlan;
    }

    private static void requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " cannot be negative: " + value);
        }
    }
}
