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
    private final long reductionDivisor;
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
            long reductionDivisor,
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
        if (reductionDivisor <= 0) {
            throw new IllegalArgumentException("reductionDivisor must be positive: " + reductionDivisor);
        }
        this.outputNodeId = outputNodeId;
        this.predictionNodeId = predictionNodeId;
        this.targetNodeId = targetNodeId;
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        if (storageKind == null) {
            throw new IllegalArgumentException("storageKind cannot be null");
        }
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        if (reductionOpType == null) {
            throw new IllegalArgumentException("reductionOpType cannot be null");
        }
        if (launchConfig == null) {
            throw new IllegalArgumentException("launchConfig cannot be null");
        }
        if (scratchBufferSpec == null) {
            throw new IllegalArgumentException("scratchBufferSpec cannot be null");
        }
        this.dataType = dataType;
        this.storageKind = storageKind;
        this.kernelId = kernelId;
        this.kernel = Cpu1MseLossKernelDispatch.kernelFor(kernelId, storageKind);
        this.reductionOpType = reductionOpType;
        this.elementCount = elementCount;
        this.reductionDivisor = reductionDivisor;
        this.orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        this.launchConfig = launchConfig;
        this.scratchBufferSpec = scratchBufferSpec;
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

    public long reductionDivisor() {
        return reductionDivisor;
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
