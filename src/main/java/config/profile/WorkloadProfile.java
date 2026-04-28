package config.profile;

import java.util.Objects;

/**
 * Optional workload-shape descriptor carried by execution profiles.
 *
 * <p>Most generic profiles use {@link #none()}. Specialized profiles can describe a transformer
 * hot-path workload so tuning and runtime policy can distinguish attention-like shapes from unrelated
 * matrix or elementwise workloads.</p>
 *
 * @param kind workload family
 * @param batch batch size for transformer hot-path workloads
 * @param heads attention head count
 * @param seqLen sequence length
 * @param headDim key/query head dimension
 * @param valueDim value head dimension
 * @param ffHiddenDim feed-forward hidden dimension
 * @param causal whether attention uses a causal mask
 */
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

    /**
     * Returns the absence of specialized workload metadata.
     *
     * @return generic workload profile with all dimensions set to zero
     */
    public static WorkloadProfile none() {
        return new WorkloadProfile(WorkloadKind.NONE, 0, 0, 0, 0, 0, 0, false);
    }

    /**
     * Returns a representative transformer hot-path profile used by tuning defaults.
     *
     * @return transformer hot-path descriptor with batch 8, 8 heads, sequence length 128, and model width 512
     */
    public static WorkloadProfile transformerHotPathDefaults() {
        return new WorkloadProfile(WorkloadKind.TRANSFORMER_HOT_PATH, 8, 8, 128, 64, 64, 2048, true);
    }

    /**
     * Returns the transformer model dimension implied by {@code heads * valueDim}.
     *
     * @return model dimension for transformer hot-path workloads
     * @throws IllegalStateException if this profile is not a transformer hot-path profile
     */
    public int modelDim() {
        if (kind != WorkloadKind.TRANSFORMER_HOT_PATH) {
            throw new IllegalStateException("modelDim is only defined for transformer hot-path workloads.");
        }
        return heads * valueDim;
    }
}
