package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.kernels.reduction.Cpu1ReductionKernel;
import backend.cpu1.kernels.reduction.Cpu1ReductionKernelDispatch;
import backend.cpu1.kernels.reduction.Cpu1ReductionKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

import java.util.Objects;

/**
 * Immutable prepare-time contract for one cpu1 reduction node.
 */
public final class Cpu1PreparedReductionUnit {
    private final int nodeId;
    private final int inputNodeId;
    private final Operation.OpType opType;
    private final DataType dataType;
    private final Cpu1StorageKind storageKind;
    private final Cpu1ReductionKernelId kernelId;
    private final Cpu1ReductionKernel kernel;
    private final int axis;
    private final int axisSize;
    private final int innerSize;
    private final int outerSize;
    private final int outputElementCount;
    private final boolean keepDims;
    private final boolean argMaxLastIndexWins;
    private final boolean cumSumExclusive;
    private final boolean cumSumReverse;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1LaunchPolicy launchPolicy;
    private final Cpu1ScratchBufferSpec scratchBufferSpec;
    private final Cpu1StorageAccessPlan inputAccessPlan;
    private final Cpu1StorageAccessPlan outputAccessPlan;

    public Cpu1PreparedReductionUnit(
            int nodeId,
            int inputNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1StorageKind storageKind,
            Cpu1ReductionKernelId kernelId,
            int axis,
            int axisSize,
            int innerSize,
            int outerSize,
            int outputElementCount,
            boolean keepDims,
            boolean argMaxLastIndexWins,
            boolean cumSumExclusive,
            boolean cumSumReverse,
            Cpu1LaunchConfig launchConfig,
            Cpu1LaunchPolicy launchPolicy,
            Cpu1ScratchBufferSpec scratchBufferSpec,
            Cpu1StorageAccessPlan inputAccessPlan,
            Cpu1StorageAccessPlan outputAccessPlan
    ) {
        if (nodeId < 0) {
            throw new IllegalArgumentException("nodeId cannot be negative");
        }
        if (inputNodeId < 0) {
            throw new IllegalArgumentException("inputNodeId cannot be negative");
        }
        requirePositive(axisSize, "axisSize");
        requirePositive(innerSize, "innerSize");
        requirePositive(outerSize, "outerSize");
        requirePositive(outputElementCount, "outputElementCount");
        this.nodeId = nodeId;
        this.inputNodeId = inputNodeId;
        this.opType = Objects.requireNonNull(opType, "opType cannot be null");
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        this.storageKind = Objects.requireNonNull(storageKind, "storageKind cannot be null");
        this.kernelId = Objects.requireNonNull(kernelId, "kernelId cannot be null");
        this.kernel = Cpu1ReductionKernelDispatch.kernelFor(kernelId, storageKind);
        this.axis = axis;
        this.axisSize = axisSize;
        this.innerSize = innerSize;
        this.outerSize = outerSize;
        this.outputElementCount = outputElementCount;
        this.keepDims = keepDims;
        this.argMaxLastIndexWins = argMaxLastIndexWins;
        this.cumSumExclusive = cumSumExclusive;
        this.cumSumReverse = cumSumReverse;
        this.launchConfig = Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        this.launchPolicy = Objects.requireNonNull(launchPolicy, "launchPolicy cannot be null");
        this.scratchBufferSpec = Objects.requireNonNull(scratchBufferSpec, "scratchBufferSpec cannot be null");
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

    public Cpu1ReductionKernelId kernelId() {
        return kernelId;
    }

    public Cpu1ReductionKernel kernel() {
        return kernel;
    }

    public int axis() {
        return axis;
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

    public boolean keepDims() {
        return keepDims;
    }

    public boolean argMaxLastIndexWins() {
        return argMaxLastIndexWins;
    }

    public boolean cumSumExclusive() {
        return cumSumExclusive;
    }

    public boolean cumSumReverse() {
        return cumSumReverse;
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
}
