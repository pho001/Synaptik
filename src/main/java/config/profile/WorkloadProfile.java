package config.profile;

import java.util.Objects;

public record WorkloadProfile(
        WorkloadKind kind,
        int batch,
        int heads,
        int seqLen,
        int headDim,
        int valueDim,
        int ffHiddenDim,
        boolean causal
) {
    public WorkloadProfile {
        kind = Objects.requireNonNullElse(kind, WorkloadKind.NONE);
        if (kind == WorkloadKind.NONE) {
            batch = 0;
            heads = 0;
            seqLen = 0;
            headDim = 0;
            valueDim = 0;
            ffHiddenDim = 0;
            causal = false;
        } else if (batch <= 0 || heads <= 0 || seqLen <= 0 || headDim <= 0 || valueDim <= 0 || ffHiddenDim <= 0) {
            throw new IllegalArgumentException("Transformer hot-path workload dimensions must be > 0.");
        }
    }

    public static WorkloadProfile none() {
        return new WorkloadProfile(WorkloadKind.NONE, 0, 0, 0, 0, 0, 0, false);
    }

    public static WorkloadProfile transformerHotPathDefaults() {
        return new WorkloadProfile(WorkloadKind.TRANSFORMER_HOT_PATH, 8, 8, 128, 64, 64, 2048, true);
    }

    public int modelDim() {
        if (kind != WorkloadKind.TRANSFORMER_HOT_PATH) {
            throw new IllegalStateException("modelDim is only defined for transformer hot-path workloads.");
        }
        return heads * valueDim;
    }
}
