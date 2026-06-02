package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1WorkspaceSpec;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.layout.Cpu1LayoutKernel;
import backend.cpu1.kernels.layout.Cpu1LayoutKernelDispatch;
import backend.cpu1.kernels.layout.Cpu1LayoutKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

import java.util.List;
import java.util.Objects;

/**
 * Immutable prepare-time contract for one cpu1 layout/view node.
 */
public final class Cpu1PreparedLayoutUnit {
    private final int nodeId;
    private final List<Integer> inputNodeIds;
    private final Operation.OpType opType;
    private final DataType dataType;
    private final Cpu1StorageKind storageKind;
    private final Cpu1LayoutKernelId kernelId;
    private final Cpu1LayoutKernel kernel;
    private final int materializeThreshold;
    private final Cpu1VectorizationKind vectorizationKind;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1WorkspaceSpec workspaceSpec;

    public Cpu1PreparedLayoutUnit(
            int nodeId,
            List<Integer> inputNodeIds,
            Operation.OpType opType,
            DataType dataType,
            Cpu1StorageKind storageKind,
            Cpu1LayoutKernelId kernelId,
            int materializeThreshold,
            Cpu1VectorizationKind vectorizationKind,
            Cpu1LaunchConfig launchConfig,
            Cpu1WorkspaceSpec workspaceSpec
    ) {
        if (nodeId < 0) {
            throw new IllegalArgumentException("nodeId cannot be negative");
        }
        this.nodeId = nodeId;
        this.inputNodeIds = List.copyOf(Objects.requireNonNull(inputNodeIds, "inputNodeIds cannot be null"));
        if (this.inputNodeIds.isEmpty()) {
            throw new IllegalArgumentException("inputNodeIds cannot be empty");
        }
        for (int inputNodeId : this.inputNodeIds) {
            if (inputNodeId < 0) {
                throw new IllegalArgumentException("inputNodeId cannot be negative");
            }
        }
        this.opType = Objects.requireNonNull(opType, "opType cannot be null");
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        this.storageKind = Objects.requireNonNull(storageKind, "storageKind cannot be null");
        this.kernelId = Objects.requireNonNull(kernelId, "kernelId cannot be null");
        this.kernel = Cpu1LayoutKernelDispatch.kernelFor(kernelId);
        if (materializeThreshold < 0) {
            throw new IllegalArgumentException("materializeThreshold cannot be negative");
        }
        this.materializeThreshold = materializeThreshold;
        this.vectorizationKind = Objects.requireNonNull(vectorizationKind, "vectorizationKind cannot be null");
        this.launchConfig = Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        this.workspaceSpec = Objects.requireNonNull(workspaceSpec, "workspaceSpec cannot be null");
    }

    public int nodeId() {
        return nodeId;
    }

    public List<Integer> inputNodeIds() {
        return inputNodeIds;
    }

    public int inputNodeId() {
        return inputNodeIds.getFirst();
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

    public Cpu1LayoutKernelId kernelId() {
        return kernelId;
    }

    public Cpu1LayoutKernel kernel() {
        return kernel;
    }

    public int materializeThreshold() {
        return materializeThreshold;
    }

    public Cpu1VectorizationKind vectorizationKind() {
        return vectorizationKind;
    }

    public Cpu1LaunchConfig launchConfig() {
        return launchConfig;
    }

    public Cpu1WorkspaceSpec workspaceSpec() {
        return workspaceSpec;
    }
}
