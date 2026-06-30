package config.compile;

import planning.partition.PartitionTarget;

/**
 * Public compile-policy backend target.
 */
public enum BackendTarget {
    CPU,
    GPU_METAL,
    GPU_CUDA;

    public boolean accelerator() {
        return this == GPU_METAL || this == GPU_CUDA;
    }

    public PartitionTarget toPartitionTarget() {
        return switch (this) {
            case CPU -> PartitionTarget.CPU;
            case GPU_METAL -> PartitionTarget.GPU_METAL;
            case GPU_CUDA -> PartitionTarget.GPU_CUDA;
        };
    }

    public static BackendTarget fromPartitionTarget(PartitionTarget target) {
        if (target == null) {
            return null;
        }
        return switch (target) {
            case CPU -> CPU;
            case GPU_METAL -> GPU_METAL;
            case GPU_CUDA -> GPU_CUDA;
            case AUTO, NONE -> null;
        };
    }
}
