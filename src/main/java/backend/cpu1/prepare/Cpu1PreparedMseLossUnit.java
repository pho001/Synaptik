package backend.cpu1.prepare;

import backend.cpu1.kernels.loss.mse.Cpu1MseLossKernel;
import backend.cpu1.kernels.loss.mse.Cpu1MseLossKernelDispatch;
import backend.cpu1.kernels.loss.mse.Cpu1MseLossKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import operations.Operation;
import tensor.DataType;

import java.util.List;
import java.util.Objects;

public final class Cpu1PreparedMseLossUnit {
    private final int outputNodeId;
    private final int predictionNodeId;
    private final int targetNodeId;
    private final DataType dataType;
    private final Cpu1StorageKind storageKind;
    private final Cpu1MseLossKernelId kernelId;
    private final Cpu1MseLossKernel kernel;
    private final Operation.OpType reductionOpType;
    private final int elementCount;
    private final List<Integer> orderedNodeIds;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1ScratchBufferSpec scratchBufferSpec;

    public Cpu1PreparedMseLossUnit(
            int outputNodeId,
            int predictionNodeId,
            int targetNodeId,
            DataType dataType,
            Cpu1StorageKind storageKind,
            Cpu1MseLossKernelId kernelId,
            Operation.OpType reductionOpType,
            int elementCount,
            List<Integer> orderedNodeIds,
            Cpu1LaunchConfig launchConfig,
            Cpu1ScratchBufferSpec scratchBufferSpec
    ) {
        if (outputNodeId < 0 || predictionNodeId < 0 || targetNodeId < 0) {
            throw new IllegalArgumentException("node ids cannot be negative");
        }
        if (elementCount <= 0) {
            throw new IllegalArgumentException("elementCount must be positive: " + elementCount);
        }
        this.outputNodeId = outputNodeId;
        this.predictionNodeId = predictionNodeId;
        this.targetNodeId = targetNodeId;
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        this.storageKind = Objects.requireNonNull(storageKind, "storageKind cannot be null");
        this.kernelId = Objects.requireNonNull(kernelId, "kernelId cannot be null");
        this.kernel = Cpu1MseLossKernelDispatch.kernelFor(kernelId);
        this.reductionOpType = Objects.requireNonNull(reductionOpType, "reductionOpType cannot be null");
        this.elementCount = elementCount;
        this.orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        this.launchConfig = Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        this.scratchBufferSpec = Objects.requireNonNull(scratchBufferSpec, "scratchBufferSpec cannot be null");
    }

    public int outputNodeId() {
        return outputNodeId;
    }

    public int predictionNodeId() {
        return predictionNodeId;
    }

    public int targetNodeId() {
        return targetNodeId;
    }

    public DataType dataType() {
        return dataType;
    }

    public Cpu1StorageKind storageKind() {
        return storageKind;
    }

    public Cpu1MseLossKernelId kernelId() {
        return kernelId;
    }

    public Cpu1MseLossKernel kernel() {
        return kernel;
    }

    public Operation.OpType reductionOpType() {
        return reductionOpType;
    }

    public int elementCount() {
        return elementCount;
    }

    public List<Integer> orderedNodeIds() {
        return orderedNodeIds;
    }

    public Cpu1LaunchConfig launchConfig() {
        return launchConfig;
    }

    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return scratchBufferSpec;
    }
}
