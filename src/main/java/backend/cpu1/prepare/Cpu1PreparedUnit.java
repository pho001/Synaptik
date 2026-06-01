package backend.cpu1.prepare;

import backend.cpu1.kernels.Cpu1KernelDispatch;
import backend.cpu1.kernels.Cpu1KernelId;
import backend.cpu1.kernels.Cpu1KernelRangeRunner;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.plan.Cpu1IterationPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

import java.util.List;
import java.util.Objects;

/**
 * Immutable prepare-time cpu1 unit.
 */
public final class Cpu1PreparedUnit {
    private final int nodeId;
    private final List<Integer> inputNodeIds;
    private final int outputNodeId;
    private final Operation.OpType opType;
    private final DataType dataType;
    private final List<DataType> inputDataTypes;
    private final Cpu1IterationPlan iterationPlan;
    private final Cpu1LayoutKind layoutKind;
    private final Cpu1StorageKind storageKind;
    private final Cpu1KernelId kernelId;
    private final Cpu1KernelRangeRunner kernelRunner;
    private final Cpu1LaunchPolicy launchPolicy;
    private final boolean hasScalarParameter;
    private final float scalarParameterF32;
    private final double scalarParameterF64;

    public Cpu1PreparedUnit(
            int nodeId,
            List<Integer> inputNodeIds,
            int outputNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1IterationPlan iterationPlan,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1KernelId kernelId,
            Cpu1LaunchPolicy launchPolicy
    ) {
        this(
                nodeId,
                inputNodeIds,
                outputNodeId,
                opType,
                dataType,
                iterationPlan,
                layoutKind,
                storageKind,
                kernelId,
                launchPolicy,
                false,
                0.0f,
                0.0d,
                repeatedInputDataTypes(inputNodeIds, dataType)
        );
    }

    public Cpu1PreparedUnit(
            int nodeId,
            List<Integer> inputNodeIds,
            int outputNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1IterationPlan iterationPlan,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1KernelId kernelId,
            Cpu1LaunchPolicy launchPolicy,
            boolean hasScalarParameter,
            float scalarParameterF32,
            double scalarParameterF64
    ) {
        this(
                nodeId,
                inputNodeIds,
                outputNodeId,
                opType,
                dataType,
                iterationPlan,
                layoutKind,
                storageKind,
                kernelId,
                launchPolicy,
                hasScalarParameter,
                scalarParameterF32,
                scalarParameterF64,
                repeatedInputDataTypes(inputNodeIds, dataType)
        );
    }

    public Cpu1PreparedUnit(
            int nodeId,
            List<Integer> inputNodeIds,
            int outputNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1IterationPlan iterationPlan,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1KernelId kernelId,
            Cpu1LaunchPolicy launchPolicy,
            boolean hasScalarParameter,
            float scalarParameterF32,
            double scalarParameterF64,
            DataType inputDataType
    ) {
        this(
                nodeId,
                inputNodeIds,
                outputNodeId,
                opType,
                dataType,
                iterationPlan,
                layoutKind,
                storageKind,
                kernelId,
                launchPolicy,
                hasScalarParameter,
                scalarParameterF32,
                scalarParameterF64,
                repeatedInputDataTypes(inputNodeIds, inputDataType)
        );
    }

    public Cpu1PreparedUnit(
            int nodeId,
            List<Integer> inputNodeIds,
            int outputNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1IterationPlan iterationPlan,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1KernelId kernelId,
            Cpu1LaunchPolicy launchPolicy,
            boolean hasScalarParameter,
            float scalarParameterF32,
            double scalarParameterF64,
            List<DataType> inputDataTypes
    ) {
        this.nodeId = nodeId;
        this.inputNodeIds = List.copyOf(Objects.requireNonNull(inputNodeIds, "inputNodeIds cannot be null"));
        this.outputNodeId = outputNodeId;
        this.opType = Objects.requireNonNull(opType, "opType cannot be null");
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        this.inputDataTypes = List.copyOf(Objects.requireNonNull(inputDataTypes, "inputDataTypes cannot be null"));
        if (this.inputDataTypes.size() != this.inputNodeIds.size()) {
            throw new IllegalArgumentException("inputDataTypes size " + this.inputDataTypes.size()
                    + " does not match inputNodeIds size " + this.inputNodeIds.size());
        }
        this.iterationPlan = Objects.requireNonNull(iterationPlan, "iterationPlan cannot be null");
        this.layoutKind = Objects.requireNonNull(layoutKind, "layoutKind cannot be null");
        this.storageKind = Objects.requireNonNull(storageKind, "storageKind cannot be null");
        this.kernelId = Objects.requireNonNull(kernelId, "kernelId cannot be null");
        this.kernelRunner = Cpu1KernelDispatch.runnerFor(kernelId);
        this.launchPolicy = Objects.requireNonNull(launchPolicy, "launchPolicy cannot be null");
        this.hasScalarParameter = hasScalarParameter;
        this.scalarParameterF32 = scalarParameterF32;
        this.scalarParameterF64 = scalarParameterF64;
    }

    public int nodeId() {
        return nodeId;
    }

    public List<Integer> inputNodeIds() {
        return inputNodeIds;
    }

    public int outputNodeId() {
        return outputNodeId;
    }

    public Operation.OpType opType() {
        return opType;
    }

    public DataType dataType() {
        return dataType;
    }

    public DataType inputDataType() {
        if (inputDataTypes.isEmpty()) {
            return dataType;
        }
        return inputDataTypes.getFirst();
    }

    public DataType inputDataType(int index) {
        return inputDataTypes.get(index);
    }

    public List<DataType> inputDataTypes() {
        return inputDataTypes;
    }

    public Cpu1IterationPlan iterationPlan() {
        return iterationPlan;
    }

    public Cpu1LayoutKind layoutKind() {
        return layoutKind;
    }

    public Cpu1StorageKind storageKind() {
        return storageKind;
    }

    public int elementCount() {
        return iterationPlan.elementCount();
    }

    public Cpu1KernelId kernelId() {
        return kernelId;
    }

    public Cpu1KernelRangeRunner kernelRunner() {
        return kernelRunner;
    }

    public Cpu1LaunchPolicy launchPolicy() {
        return launchPolicy;
    }

    public boolean hasScalarParameter() {
        return hasScalarParameter;
    }

    public float scalarParameterF32() {
        if (!hasScalarParameter) {
            throw new IllegalStateException("cpu1 unit does not have a scalar parameter.");
        }
        return scalarParameterF32;
    }

    public double scalarParameterF64() {
        if (!hasScalarParameter) {
            throw new IllegalStateException("cpu1 unit does not have a scalar parameter.");
        }
        return scalarParameterF64;
    }

    private static List<DataType> repeatedInputDataTypes(List<Integer> inputNodeIds, DataType dataType) {
        Objects.requireNonNull(inputNodeIds, "inputNodeIds cannot be null");
        Objects.requireNonNull(dataType, "dataType cannot be null");
        return inputNodeIds.stream().map(ignored -> dataType).toList();
    }
}
