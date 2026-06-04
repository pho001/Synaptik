package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.layout.Cpu1LayoutKernel;
import backend.cpu1.kernels.layout.Cpu1LayoutKernelDispatch;
import backend.cpu1.kernels.layout.Cpu1LayoutKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;
import tensor.options.Window2dOptions;

import java.util.List;

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
    private final Cpu1ScratchBufferSpec scratchBufferSpec;
    private final int axis;
    private final int[] padBefore;
    private final int[] padAfter;
    private final double padConstantValue;
    private final int unfoldAxis;
    private final int unfoldSize;
    private final int unfoldStep;
    private final Window2dOptions window2dOptions;

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
            Cpu1ScratchBufferSpec scratchBufferSpec,
            int axis,
            int[] padBefore,
            int[] padAfter,
            double padConstantValue,
            int unfoldAxis,
            int unfoldSize,
            int unfoldStep,
            Window2dOptions window2dOptions
    ) {
        if (nodeId < 0) {
            throw new IllegalArgumentException("nodeId cannot be negative");
        }
        if (inputNodeIds == null) {
            throw new IllegalArgumentException("inputNodeIds cannot be null");
        }
        if (opType == null) {
            throw new IllegalArgumentException("opType cannot be null");
        }
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        if (storageKind == null) {
            throw new IllegalArgumentException("storageKind cannot be null");
        }
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        if (vectorizationKind == null) {
            throw new IllegalArgumentException("vectorizationKind cannot be null");
        }
        if (launchConfig == null) {
            throw new IllegalArgumentException("launchConfig cannot be null");
        }
        if (scratchBufferSpec == null) {
            throw new IllegalArgumentException("scratchBufferSpec cannot be null");
        }
        this.nodeId = nodeId;
        this.inputNodeIds = List.copyOf(inputNodeIds);
        if (this.inputNodeIds.isEmpty()) {
            throw new IllegalArgumentException("inputNodeIds cannot be empty");
        }
        for (int inputNodeId : this.inputNodeIds) {
            if (inputNodeId < 0) {
                throw new IllegalArgumentException("inputNodeId cannot be negative");
            }
        }
        this.opType = opType;
        this.dataType = dataType;
        this.storageKind = storageKind;
        this.kernelId = kernelId;
        this.kernel = Cpu1LayoutKernelDispatch.kernelFor(kernelId);
        if (materializeThreshold < 0) {
            throw new IllegalArgumentException("materializeThreshold cannot be negative");
        }
        this.materializeThreshold = materializeThreshold;
        this.vectorizationKind = vectorizationKind;
        this.launchConfig = launchConfig;
        this.scratchBufferSpec = scratchBufferSpec;
        this.axis = axis;
        this.padBefore = padBefore == null ? new int[0] : padBefore.clone();
        this.padAfter = padAfter == null ? new int[0] : padAfter.clone();
        this.padConstantValue = padConstantValue;
        this.unfoldAxis = unfoldAxis;
        this.unfoldSize = unfoldSize;
        this.unfoldStep = unfoldStep;
        this.window2dOptions = window2dOptions;
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

    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return scratchBufferSpec;
    }

    public int axis() {
        return axis;
    }

    public int[] padBefore() {
        return padBefore.clone();
    }

    public int[] padAfter() {
        return padAfter.clone();
    }

    public double padConstantValue() {
        return padConstantValue;
    }

    public int unfoldAxis() {
        return unfoldAxis;
    }

    public int unfoldSize() {
        return unfoldSize;
    }

    public int unfoldStep() {
        return unfoldStep;
    }

    public Window2dOptions window2dOptions() {
        return window2dOptions;
    }
}
