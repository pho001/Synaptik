package backend.cpu.fused.numeric;

import backend.ApproxMode;
import config.runtime.ApproximationConfig;

/**
 * Prepared transcendental approximation contract for generated fused kernels.
 */
public enum FusedApproximationContract {
    STRICT(false, false),
    FAST_EXP(true, false),
    FAST_TANH(false, true),
    FAST_EXP_AND_TANH(true, true);

    private final boolean fastExp;
    private final boolean fastTanh;

    FusedApproximationContract(boolean fastExp, boolean fastTanh) {
        this.fastExp = fastExp;
        this.fastTanh = fastTanh;
    }

    public boolean useFastExp() {
        return fastExp;
    }

    public boolean useFastTanh() {
        return fastTanh;
    }

    public String signatureToken() {
        return name();
    }

    public static FusedApproximationContract from(ApproximationConfig config, boolean supportsBackward) {
        if (config == null || config.forceExactTranscendentals()) {
            return STRICT;
        }
        ApproxMode mode = config.approxMode();
        boolean fast = mode == ApproxMode.ALWAYS || (mode == ApproxMode.TRAINING_ONLY && supportsBackward);
        return fast ? FAST_EXP_AND_TANH : STRICT;
    }
}
