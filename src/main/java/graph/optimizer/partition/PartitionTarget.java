package graph.optimizer.partition;

import backend.ComputeBackend;

public enum PartitionTarget {
    AUTO(null),
    NONE(null),
    CPU(ComputeBackend.CPU),
    GPU_METAL(ComputeBackend.GPU_METAL),
    GPU_CUDA(ComputeBackend.GPU_CUDA);

    private final ComputeBackend backend;

    PartitionTarget(ComputeBackend backend) {
        this.backend = backend;
    }

    public ComputeBackend backend() {
        return backend;
    }

    public boolean isAuto() {
        return this == AUTO;
    }

    public boolean isNone() {
        return this == NONE;
    }

    public static PartitionTarget fromBackend(ComputeBackend backend) {
        if (backend == null) {
            return NONE;
        }
        return switch (backend) {
            case CPU -> CPU;
            case GPU_METAL -> GPU_METAL;
            case GPU_CUDA -> GPU_CUDA;
            default -> NONE;
        };
    }
}
