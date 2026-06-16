package backend.cpu1.prepare;

import backend.cpu1.kernels.elementwise.Cpu1ElementwiseKernelDispatch;
import backend.cpu1.kernels.elementwise.Cpu1ElementwiseKernelId;
import backend.cpu1.kernels.elementwise.Cpu1ElementwiseRangeRunner;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.plan.Cpu1IterationPlan;
import backend.cpu1.prepare.dispatch.Cpu1DispatchDecision;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;
import tensor.TensorMetadata;

import java.util.ArrayList;
import java.util.List;

/**
 * Immutable prepare-time cpu1 elementwise unit.
 */
public final class Cpu1PreparedElementwiseUnit {
    private final int nodeId;
    private final List<Integer> inputNodeIds;
    private final int outputNodeId;
    private final Operation.OpType opType;
    private final DataType dataType;
    private final List<DataType> inputDataTypes;
    private final Cpu1IterationPlan iterationPlan;
    private final Cpu1LayoutKind layoutKind;
    private final Cpu1StorageKind storageKind;
    private final Cpu1ElementwiseKernelId kernelId;
    private final Cpu1ElementwiseRangeRunner kernelRunner;
    private final Cpu1LaunchPolicy launchPolicy;
    private final boolean hasScalarParameter;
    private final float scalarParameterF32;
    private final double scalarParameterF64;
    private final Cpu1DispatchDecision dispatchDecision;
    private final List<Cpu1StorageAccessPlan> inputAccessPlans;
    private final Cpu1StorageAccessPlan outputAccessPlan;

    public Cpu1PreparedElementwiseUnit(
            int nodeId,
            List<Integer> inputNodeIds,
            int outputNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1IterationPlan iterationPlan,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1ElementwiseKernelId kernelId,
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

    public Cpu1PreparedElementwiseUnit(
            int nodeId,
            List<Integer> inputNodeIds,
            int outputNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1IterationPlan iterationPlan,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1ElementwiseKernelId kernelId,
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

    public Cpu1PreparedElementwiseUnit(
            int nodeId,
            List<Integer> inputNodeIds,
            int outputNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1IterationPlan iterationPlan,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1ElementwiseKernelId kernelId,
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

    public Cpu1PreparedElementwiseUnit(
            int nodeId,
            List<Integer> inputNodeIds,
            int outputNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1IterationPlan iterationPlan,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1ElementwiseKernelId kernelId,
            Cpu1LaunchPolicy launchPolicy,
            boolean hasScalarParameter,
            float scalarParameterF32,
            double scalarParameterF64,
            List<DataType> inputDataTypes
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
                inputDataTypes,
                null
        );
    }

    public Cpu1PreparedElementwiseUnit(
            int nodeId,
            List<Integer> inputNodeIds,
            int outputNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1IterationPlan iterationPlan,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1ElementwiseKernelId kernelId,
            Cpu1LaunchPolicy launchPolicy,
            boolean hasScalarParameter,
            float scalarParameterF32,
            double scalarParameterF64,
            List<DataType> inputDataTypes,
            Cpu1DispatchDecision dispatchDecision
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
                inputDataTypes,
                dispatchDecision,
                List.of(),
                null
        );
    }

    public Cpu1PreparedElementwiseUnit(
            int nodeId,
            List<Integer> inputNodeIds,
            int outputNodeId,
            Operation.OpType opType,
            DataType dataType,
            Cpu1IterationPlan iterationPlan,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1ElementwiseKernelId kernelId,
            Cpu1LaunchPolicy launchPolicy,
            boolean hasScalarParameter,
            float scalarParameterF32,
            double scalarParameterF64,
            List<DataType> inputDataTypes,
            Cpu1DispatchDecision dispatchDecision,
            List<Cpu1StorageAccessPlan> inputAccessPlans,
            Cpu1StorageAccessPlan outputAccessPlan
    ) {
        if (inputNodeIds == null) {
            throw new IllegalArgumentException("inputNodeIds cannot be null");
        }
        if (opType == null) {
            throw new IllegalArgumentException("opType cannot be null");
        }
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        if (inputDataTypes == null) {
            throw new IllegalArgumentException("inputDataTypes cannot be null");
        }
        if (inputAccessPlans == null) {
            throw new IllegalArgumentException("inputAccessPlans cannot be null");
        }
        if (iterationPlan == null) {
            throw new IllegalArgumentException("iterationPlan cannot be null");
        }
        if (layoutKind == null) {
            throw new IllegalArgumentException("layoutKind cannot be null");
        }
        if (storageKind == null) {
            throw new IllegalArgumentException("storageKind cannot be null");
        }
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        if (launchPolicy == null) {
            throw new IllegalArgumentException("launchPolicy cannot be null");
        }
        this.nodeId = nodeId;
        this.inputNodeIds = List.copyOf(inputNodeIds);
        this.outputNodeId = outputNodeId;
        this.opType = opType;
        this.dataType = dataType;
        this.inputDataTypes = List.copyOf(inputDataTypes);
        if (this.inputDataTypes.size() != this.inputNodeIds.size()) {
            throw new IllegalArgumentException("inputDataTypes size " + this.inputDataTypes.size()
                    + " does not match inputNodeIds size " + this.inputNodeIds.size());
        }
        this.inputAccessPlans = inputAccessPlans.isEmpty()
                ? defaultInputAccessPlans(this.inputNodeIds.size(), iterationPlan)
                : List.copyOf(inputAccessPlans);
        if (!this.inputAccessPlans.isEmpty() && this.inputAccessPlans.size() != this.inputNodeIds.size()) {
            throw new IllegalArgumentException("inputAccessPlans size " + this.inputAccessPlans.size()
                    + " does not match inputNodeIds size " + this.inputNodeIds.size());
        }
        this.iterationPlan = iterationPlan;
        this.layoutKind = layoutKind;
        this.storageKind = storageKind;
        this.kernelId = kernelId;
        this.kernelRunner = Cpu1ElementwiseKernelDispatch.kernelFor(kernelId);
        this.launchPolicy = launchPolicy;
        this.hasScalarParameter = hasScalarParameter;
        this.scalarParameterF32 = scalarParameterF32;
        this.scalarParameterF64 = scalarParameterF64;
        this.dispatchDecision = dispatchDecision;
        this.outputAccessPlan = outputAccessPlan == null ? defaultAccessPlan(iterationPlan) : outputAccessPlan;
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

    public List<Cpu1StorageAccessPlan> inputAccessPlans() {
        return inputAccessPlans;
    }

    public Cpu1StorageAccessPlan inputAccessPlan(int index) {
        return inputAccessPlans.get(index);
    }

    public Cpu1StorageAccessPlan outputAccessPlan() {
        return outputAccessPlan;
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

    public Cpu1ElementwiseKernelId kernelId() {
        return kernelId;
    }

    public Cpu1ElementwiseRangeRunner kernelRunner() {
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

    public Cpu1DispatchDecision dispatchDecision() {
        if (dispatchDecision == null) {
            throw new IllegalStateException("This cpu1 prepared unit does not expose a dispatch decision.");
        }
        return dispatchDecision;
    }

    private static List<DataType> repeatedInputDataTypes(List<Integer> inputNodeIds, DataType dataType) {
        if (inputNodeIds == null) {
            throw new IllegalArgumentException("inputNodeIds cannot be null");
        }
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        return inputNodeIds.stream().map(ignored -> dataType).toList();
    }

    private static List<Cpu1StorageAccessPlan> defaultInputAccessPlans(int inputCount, Cpu1IterationPlan iterationPlan) {
        List<Cpu1StorageAccessPlan> plans = new ArrayList<>(inputCount);
        Cpu1StorageAccessPlan plan = defaultAccessPlan(iterationPlan);
        for (int i = 0; i < inputCount; i++) {
            plans.add(plan);
        }
        return List.copyOf(plans);
    }

    private static Cpu1StorageAccessPlan defaultAccessPlan(Cpu1IterationPlan iterationPlan) {
        int[] shape = iterationPlan.shape();
        return new Cpu1StorageAccessPlan(
                Cpu1StorageAccessKind.DENSE_CONTIGUOUS,
                shape,
                TensorMetadata.computeStrides(shape),
                0,
                iterationPlan.elementCount(),
                null
        );
    }
}
