package tuning.workload;

public enum WorkloadKind {
    GENERIC,
    MATMUL,
    CONV2D,
    NORMALIZATION,
    TRANSFORMER_HOT_PATH,
    POOL2D,
    LOSS
}
