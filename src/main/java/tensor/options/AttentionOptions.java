package tensor.options;

public record AttentionOptions(
        boolean causal,
        Double scaleOverride
) {
    public AttentionOptions {
        if (scaleOverride != null && !(scaleOverride > 0.0d)) {
            throw new IllegalArgumentException("Attention scaleOverride must be positive.");
        }
    }

    public static AttentionOptions defaults() {
        return new AttentionOptions(false, null);
    }

    public static AttentionOptions causalDefaults() {
        return new AttentionOptions(true, null);
    }

    public AttentionOptions withCausal(boolean causal) {
        return new AttentionOptions(causal, scaleOverride);
    }

    public AttentionOptions withScale(double scaleOverride) {
        return new AttentionOptions(causal, scaleOverride);
    }

    public double resolveScale(int headDim) {
        if (scaleOverride != null) {
            return scaleOverride;
        }
        return 1.0d / Math.sqrt(headDim);
    }
}
