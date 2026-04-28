package graph.optimizer.partition;

import backend.ComputeBackend;

/**
 * Backend target requested or selected for partition planning.
 */
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

    /**
     * Returns the compute backend represented by this target.
     *
     * @return backend, or {@code null} for {@link #AUTO} and {@link #NONE}
     */
    public ComputeBackend backend() {
        return backend;
    }

    /**
     * Returns whether this target should be resolved from graph backend intent.
     *
     * @return {@code true} for {@link #AUTO}
     */
    public boolean isAuto() {
        return this == AUTO;
    }

    /**
     * Returns whether partitioning is disabled or unsupported.
     *
     * @return {@code true} for {@link #NONE}
     */
    public boolean isNone() {
        return this == NONE;
    }

    /**
     * Converts a compute backend to a partition target.
     *
     * @param backend compute backend
     * @return matching partition target, or {@link #NONE} for {@code null} or unsupported backends
     */
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
