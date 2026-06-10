package backend.cpu1.prepare;

import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.dtype.Cpu1DTypeKernel;
import backend.cpu1.kernels.dtype.Cpu1DTypeKernelDispatch;
import backend.cpu1.kernels.dtype.Cpu1DTypeKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

/**
 * Immutable prepare-time contract for one cpu1 dtype conversion node.
 */
public final class Cpu1PreparedDTypeUnit {
    private final int nodeId;
    private final int inputNodeId;
    private final Operation.OpType opType;
    private final DataType inputDataType;
    private final DataType outputDataType;
    private final int elementCount;
    private final Cpu1LayoutKind layoutKind;
    private final Cpu1StorageKind storageKind;
    private final Cpu1DTypeKernelId kernelId;
    private final Cpu1DTypeKernel kernel;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1LaunchPolicy launchPolicy;

    public Cpu1PreparedDTypeUnit(
            int nodeId,
            int inputNodeId,
            Operation.OpType opType,
            DataType inputDataType,
            DataType outputDataType,
            int elementCount,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1DTypeKernelId kernelId,
            Cpu1LaunchConfig launchConfig,
            Cpu1LaunchPolicy launchPolicy
    ) {
        if (nodeId < 0) {
            throw new IllegalArgumentException("nodeId cannot be negative");
        }
        if (inputNodeId < 0) {
            throw new IllegalArgumentException("inputNodeId cannot be negative");
        }
        if (opType == null) {
            throw new IllegalArgumentException("opType cannot be null");
        }
        if (inputDataType == null) {
            throw new IllegalArgumentException("inputDataType cannot be null");
        }
        if (outputDataType == null) {
            throw new IllegalArgumentException("outputDataType cannot be null");
        }
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount cannot be negative");
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
        if (launchConfig == null) {
            throw new IllegalArgumentException("launchConfig cannot be null");
        }
        if (launchPolicy == null) {
            throw new IllegalArgumentException("launchPolicy cannot be null");
        }
        this.nodeId = nodeId;
        this.inputNodeId = inputNodeId;
        this.opType = opType;
        this.inputDataType = inputDataType;
        this.outputDataType = outputDataType;
        this.elementCount = elementCount;
        this.layoutKind = layoutKind;
        this.storageKind = storageKind;
        this.kernelId = kernelId;
        this.kernel = Cpu1DTypeKernelDispatch.kernelFor(kernelId);
        this.launchConfig = launchConfig;
        this.launchPolicy = launchPolicy;
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

    public DataType inputDataType() {
        return inputDataType;
    }

    public DataType outputDataType() {
        return outputDataType;
    }

    public int elementCount() {
        return elementCount;
    }

    public Cpu1LayoutKind layoutKind() {
        return layoutKind;
    }

    public Cpu1StorageKind storageKind() {
        return storageKind;
    }

    public Cpu1DTypeKernelId kernelId() {
        return kernelId;
    }

    public Cpu1DTypeKernel kernel() {
        return kernel;
    }

    public Cpu1LaunchConfig launchConfig() {
        return launchConfig;
    }

    public Cpu1LaunchPolicy launchPolicy() {
        return launchPolicy;
    }
}
