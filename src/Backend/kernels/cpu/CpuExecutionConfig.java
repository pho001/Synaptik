package Backend.kernels.cpu;

import Config.backend.CpuKernelConfig;
import Operations.Operation;
import Tensor.Tensor;

public final class  CpuExecutionConfig {
    private final int vectorMinSize;
    private final int parallelMinSize;
    private final int parallelism;
    private final int chunksPerWorker;
    private final int minChunkSize;
    private final int contiguousMaterializeThreshold;

    public CpuExecutionConfig(
            int vectorMinSize,
            int parallelMinSize,
            int parallelism,
            int chunksPerWorker,
            int minChunkSize,
            int contiguousMaterializeThreshold
    ) {
        this.vectorMinSize = vectorMinSize;
        this.parallelMinSize = parallelMinSize;
        this.parallelism = parallelism;
        this.chunksPerWorker = chunksPerWorker;
        this.minChunkSize = minChunkSize;
        this.contiguousMaterializeThreshold = contiguousMaterializeThreshold;
    }

    public static CpuExecutionConfig defaults() {
        return fromKernelConfig(CpuKernelConfig.defaultsTraining());
    }

    public static CpuExecutionConfig fromKernelConfig(CpuKernelConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config cannot be null");
        }
        return new CpuExecutionConfig(
                config.vectorMinSize(),
                config.parallelMinSize(),
                config.parallelism(),
                config.chunksPerWorker(),
                config.minChunkSize(),
                config.contiguousMaterializeThreshold()
        );
    }

    public CpuExecutionMode modeFor(Operation op, Tensor node) {
        if (op == null || node == null || !op.isElementWise()) {
            return CpuExecutionMode.SCALAR;
        }
        int size = node.getFlatDataSize();
        if (size >= parallelMinSize) {
            if (size >= vectorMinSize) {
                return CpuExecutionMode.PARALLEL_VECTOR;
            }
            return CpuExecutionMode.PARALLEL;
        }
        if (size >= vectorMinSize) {
            return CpuExecutionMode.VECTOR;
        }
        return CpuExecutionMode.SCALAR;
    }

    public int plannedWorkers() {
        int configured = parallelism > 0 ? parallelism : Runtime.getRuntime().availableProcessors();
        return Math.max(1, configured);
    }

    public int computeChunkSize(int totalLength, int vectorWidth) {
        int length = Math.max(1, totalLength);
        int workers = plannedWorkers();
        int targets = Math.max(workers, workers * Math.max(1, chunksPerWorker));
        int candidate = (length + targets - 1) / targets;
        int chunk = Math.max(Math.max(1, minChunkSize), candidate);
        if (vectorWidth > 1) {
            int rem = chunk % vectorWidth;
            if (rem != 0) {
                chunk += (vectorWidth - rem);
            }
        }
        return chunk;
    }

    public int contiguousMaterializeThreshold() {
        return contiguousMaterializeThreshold;
    }

}
