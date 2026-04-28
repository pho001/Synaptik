package config.backend;

/**
 * Policy override for attention-specific matmul execution.
 *
 * <p>The runtime normally decides whether attention matmul should use the attention-specialized
 * microkernel path from calibrated thresholds. This enum lets tests or profiles force that decision
 * for diagnostics and controlled benchmarking.</p>
 */
public enum AttentionMatMulPolicy {
    /**
     * Let runtime dispatch choose the attention matmul path from calibrated policy.
     */
    AUTO,

    /**
     * Force the attention-specific matmul path when the operation is otherwise supported.
     */
    FORCE_ON,

    /**
     * Disable the attention-specific matmul path and use the regular matmul path.
     */
    FORCE_OFF
}
