package tuning.workload;

public enum WorkloadKind {
    GENERIC,
    MATMUL,
    MLP_CLASSIFICATION,
    ABC_SEQUENCE_MATMUL,
    CONV2D,
    NORMALIZATION,
    TRANSFORMER_HOT_PATH,
    POOL2D,
    LOSS
}
