package graph.optimizer.partition;

import backend.ComputeBackend;

public enum AcceleratorTarget {
    AUTO(null),
    NONE(null),
    GPU_METAL(ComputeBackend.GPU_METAL),
    GPU_CUDA(ComputeBackend.GPU_CUDA);

    private final ComputeBackend backend;

    AcceleratorTarget(ComputeBackend backend) {
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

    public static AcceleratorTarget fromBackend(ComputeBackend backend) {
        if (backend == null) {
            return NONE;
        }
        return switch (backend) {
            case GPU_METAL -> GPU_METAL;
            case GPU_CUDA -> GPU_CUDA;
            default -> NONE;
        };
    }
}
