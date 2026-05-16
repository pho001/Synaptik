package config.runtime;

/**
 * Runtime policy for trainable BF16 parameters and optimizer state.
 */
public enum BFloat16TrainingPolicy {
    /**
     * Default safe mixed-precision policy: BF16 is allowed for activations/temporaries,
     * while trainable parameters and optimizer state stay in higher precision paths.
     */
    ACTIVATIONS_ONLY,

    /**
     * BF16 parameter compute copies are allowed, but optimizer updates require F32 master state.
     */
    PARAMS_WITH_F32_MASTER,

    /**
     * Explicit experimental mode that permits direct BF16 parameter updates with trace evidence.
     */
    PARAMS_BF16_EXPERIMENTAL
}
