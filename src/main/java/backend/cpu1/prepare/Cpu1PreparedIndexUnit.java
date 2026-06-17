package backend.cpu1.prepare;

import backend.cpu1.kernels.index.Cpu1IndexKernel;
import backend.cpu1.kernels.index.Cpu1IndexKernelDispatch;
import backend.cpu1.kernels.index.Cpu1IndexKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import operations.index.ScatterReduction;
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
    private final int updateNodeId;
    private final Operation.OpType opType;
    private final ScatterReduction reduction;
    private final DataType valueDataType;
    private final DataType indexDataType;
    private final Cpu1StorageKind storageKind;
    private final Cpu1IndexKernelId kernelId;
    private final Cpu1IndexKernel kernel;
    private final int dimension;
    private final int axisSize;
    private final int innerSize;
    private final int outerSize;
    private final int indexElementCount;
    private final int indexAxisSize;
    private final int batchDims;
    private final int tupleRank;
    private final int prefixRank;
    private final int tupleStride;
    private final int updateElementCount;
    private final int[] gatherNdInputShape;
    private final int[] gatherNdInputStrides;
    private final int[] gatherNdIndicesDenseStrides;
    private final int[] gatherNdOutputShape;
    private final int[] gatherNdOutputDenseStrides;
    private final int outputElementCount;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1LaunchPolicy launchPolicy;
    private final Cpu1StorageAccessPlan inputAccessPlan;
    private final Cpu1StorageAccessPlan indexAccessPlan;
    private final Cpu1StorageAccessPlan updateAccessPlan;
    private final Cpu1StorageAccessPlan outputAccessPlan;

    public Cpu1PreparedIndexUnit(
            int nodeId,
            int inputNodeId,
            int indexNodeId,
            int updateNodeId,
            Operation.OpType opType,
            ScatterReduction reduction,
            DataType valueDataType,
            DataType indexDataType,
            Cpu1StorageKind storageKind,
            Cpu1IndexKernelId kernelId,
            int dimension,
            int axisSize,
            int innerSize,
            int outerSize,
            int indexElementCount,
            int indexAxisSize,
            int batchDims,
            int tupleRank,
            int prefixRank,
            int tupleStride,
            int updateElementCount,
            int[] gatherNdInputShape,
            int[] gatherNdInputStrides,
            int[] gatherNdIndicesDenseStrides,
            int[] gatherNdOutputShape,
            int[] gatherNdOutputDenseStrides,
            int outputElementCount,
            Cpu1LaunchConfig launchConfig,
            Cpu1LaunchPolicy launchPolicy,
            Cpu1StorageAccessPlan inputAccessPlan,
            Cpu1StorageAccessPlan indexAccessPlan,
            Cpu1StorageAccessPlan updateAccessPlan,
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
        if (updateNodeId < -1) {
            throw new IllegalArgumentException("updateNodeId cannot be less than -1");
        }
        if (dimension < 0) {
            throw new IllegalArgumentException("dimension cannot be negative");
        }
        requireNonNegative(axisSize, "axisSize");
        requireNonNegative(innerSize, "innerSize");
        requireNonNegative(outerSize, "outerSize");
        requireNonNegative(indexElementCount, "indexElementCount");
        requireNonNegative(indexAxisSize, "indexAxisSize");
        requireNonNegative(batchDims, "batchDims");
        requireNonNegative(tupleRank, "tupleRank");
        requireNonNegative(prefixRank, "prefixRank");
        requireNonNegative(tupleStride, "tupleStride");
        requireNonNegative(updateElementCount, "updateElementCount");
        requireNonNegative(outputElementCount, "outputElementCount");
        this.nodeId = nodeId;
        this.inputNodeId = inputNodeId;
        this.indexNodeId = indexNodeId;
        this.updateNodeId = updateNodeId;
        this.opType = Objects.requireNonNull(opType, "opType cannot be null");
        this.reduction = Objects.requireNonNull(reduction, "reduction cannot be null");
        this.valueDataType = Objects.requireNonNull(valueDataType, "valueDataType cannot be null");
        this.indexDataType = Objects.requireNonNull(indexDataType, "indexDataType cannot be null");
        this.storageKind = Objects.requireNonNull(storageKind, "storageKind cannot be null");
        this.kernelId = Objects.requireNonNull(kernelId, "kernelId cannot be null");
        this.kernel = Cpu1IndexKernelDispatch.kernelFor(kernelId);
        this.dimension = dimension;
        this.axisSize = axisSize;
        this.innerSize = innerSize;
        this.outerSize = outerSize;
        this.indexElementCount = indexElementCount;
        this.indexAxisSize = indexAxisSize;
        this.batchDims = batchDims;
        this.tupleRank = tupleRank;
        this.prefixRank = prefixRank;
        this.tupleStride = tupleStride;
        this.updateElementCount = updateElementCount;
        this.gatherNdInputShape = copyShape(gatherNdInputShape, "gatherNdInputShape");
        this.gatherNdInputStrides = copyShape(gatherNdInputStrides, "gatherNdInputStrides");
        this.gatherNdIndicesDenseStrides = copyShape(gatherNdIndicesDenseStrides, "gatherNdIndicesDenseStrides");
        this.gatherNdOutputShape = copyShape(gatherNdOutputShape, "gatherNdOutputShape");
        this.gatherNdOutputDenseStrides = copyShape(gatherNdOutputDenseStrides, "gatherNdOutputDenseStrides");
        this.outputElementCount = outputElementCount;
        this.launchConfig = Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        this.launchPolicy = Objects.requireNonNull(launchPolicy, "launchPolicy cannot be null");
        this.inputAccessPlan = Objects.requireNonNull(inputAccessPlan, "inputAccessPlan cannot be null");
        this.indexAccessPlan = Objects.requireNonNull(indexAccessPlan, "indexAccessPlan cannot be null");
        this.updateAccessPlan = updateAccessPlan;
        this.outputAccessPlan = Objects.requireNonNull(outputAccessPlan, "outputAccessPlan cannot be null");
        if (hasUpdateInput() && this.updateAccessPlan == null) {
            throw new IllegalArgumentException("updateAccessPlan cannot be null for " + opType);
        }
        if (!hasUpdateInput() && this.updateAccessPlan != null) {
            throw new IllegalArgumentException("updateAccessPlan must be null for " + opType);
        }
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

    public int updateNodeId() {
        return updateNodeId;
    }

    public boolean hasUpdateInput() {
        return updateNodeId >= 0;
    }

    public List<Integer> inputNodeIds() {
        if (hasUpdateInput()) {
            return List.of(inputNodeId, indexNodeId, updateNodeId);
        }
        return List.of(inputNodeId, indexNodeId);
    }

    public Operation.OpType opType() {
        return opType;
    }

    public ScatterReduction reduction() {
        return reduction;
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

    public int indexElementCount() {
        return indexElementCount;
    }

    public int indexAxisSize() {
        return indexAxisSize;
    }

    public int batchDims() {
        return batchDims;
    }

    public int tupleRank() {
        return tupleRank;
    }

    public int prefixRank() {
        return prefixRank;
    }

    public int tupleStride() {
        return tupleStride;
    }

    public int updateElementCount() {
        return updateElementCount;
    }

    public int[] gatherNdInputShape() {
        return gatherNdInputShape.clone();
    }

    public int[] gatherNdInputStrides() {
        return gatherNdInputStrides.clone();
    }

    public int[] gatherNdIndicesDenseStrides() {
        return gatherNdIndicesDenseStrides.clone();
    }

    public int[] gatherNdOutputShape() {
        return gatherNdOutputShape.clone();
    }

    public int[] gatherNdOutputDenseStrides() {
        return gatherNdOutputDenseStrides.clone();
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

    public Cpu1StorageAccessPlan updateAccessPlan() {
        return updateAccessPlan;
    }

    public Cpu1StorageAccessPlan outputAccessPlan() {
        return outputAccessPlan;
    }

    private static void requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " cannot be negative: " + value);
        }
    }

    private static int[] copyShape(int[] values, String name) {
        if (values == null) {
            throw new IllegalArgumentException(name + " cannot be null");
        }
        return values.clone();
    }
}
