package tuning.workload;

public enum WorkloadKind {
    GENERIC,
    MATMUL,
    MLP_CLASSIFICATION,
    ABC_SEQUENCE_MATMUL,
    CONV2D,
    REDUCTION,
    NORMALIZATION,
    TRANSFORMER_HOT_PATH,
    POOL2D,
    LOSS,
    BOOL_COMPARE
}
