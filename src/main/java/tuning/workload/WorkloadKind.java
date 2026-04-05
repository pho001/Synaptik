package tuning.workload;

public enum WorkloadKind {
    GENERIC,
    MATMUL,
    MLP_CLASSIFICATION,
    CONV2D,
    NORMALIZATION,
    TRANSFORMER_HOT_PATH,
    POOL2D,
    LOSS
}
