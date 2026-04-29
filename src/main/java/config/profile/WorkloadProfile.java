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
        return transformerHotPathMedium();
    }

    /**
     * Returns the medium transformer hot-path profile used as the continuity baseline.
     *
     * @return transformer profile with batch 8, 8 heads, sequence length 128, and model width 512
     */
    public static WorkloadProfile transformerHotPathMedium() {
        return new WorkloadProfile(WorkloadKind.TRANSFORMER_HOT_PATH, 8, 8, 128, 64, 64, 2048, true);
    }

    /**
     * Returns a larger mixed transformer block profile for accelerator stress testing.
     *
     * @return transformer profile with batch 8, 12 heads, sequence length 256, and wider FFN
     */
    public static WorkloadProfile transformerHotPathLarge() {
        return new WorkloadProfile(WorkloadKind.TRANSFORMER_HOT_PATH, 8, 12, 256, 64, 64, 3072, true);
    }

    /**
     * Returns a long-sequence transformer profile that stresses attention and softmax work.
     *
     * @return transformer profile with batch 4, 8 heads, and sequence length 512
     */
    public static WorkloadProfile transformerHotPathLongSeq() {
        return new WorkloadProfile(WorkloadKind.TRANSFORMER_HOT_PATH, 4, 8, 512, 64, 64, 2048, true);
    }

    /**
     * Returns a feed-forward-heavy transformer profile that stresses projection matmuls.
     *
     * @return transformer profile with current medium attention shape and a 4096-wide FFN
     */
    public static WorkloadProfile transformerHotPathFfnHeavy() {
        return new WorkloadProfile(WorkloadKind.TRANSFORMER_HOT_PATH, 8, 8, 128, 64, 64, 4096, true);
    }

    /**
     * Returns an attention-heavy transformer profile with more heads and a longer sequence.
     *
     * @return transformer profile with batch 8, 16 heads, sequence length 256, and model width 1024
     */
    public static WorkloadProfile transformerHotPathAttentionHeavy() {
        return new WorkloadProfile(WorkloadKind.TRANSFORMER_HOT_PATH, 8, 16, 256, 64, 64, 2048, true);
    }

    /**
     * Resolves a stable preset name for known transformer hot-path dimensions.
     *
     * @return known preset id, {@code custom}, or {@code none}
     */
    public String transformerPresetName() {
        if (kind != WorkloadKind.TRANSFORMER_HOT_PATH) {
            return "none";
        }
        if (sameShape(this, transformerHotPathMedium())) {
            return "medium";
        }
        if (sameShape(this, transformerHotPathLarge())) {
            return "large";
        }
        if (sameShape(this, transformerHotPathLongSeq())) {
            return "long_seq";
        }
        if (sameShape(this, transformerHotPathFfnHeavy())) {
            return "ffn_heavy";
        }
        if (sameShape(this, transformerHotPathAttentionHeavy())) {
            return "attention_heavy";
        }
        return "custom";
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

    private static boolean sameShape(WorkloadProfile left, WorkloadProfile right) {
        return left.kind == right.kind
                && left.batch == right.batch
                && left.heads == right.heads
                && left.seqLen == right.seqLen
                && left.headDim == right.headDim
                && left.valueDim == right.valueDim
                && left.ffHiddenDim == right.ffHiddenDim
                && left.causal == right.causal;
    }
}
