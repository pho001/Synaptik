package config.runtime;

/**
 * Runtime policy for fused elementwise execution.
 *
 * <p>The current production primary backend is generated ASM. {@code allowBackendFallback} controls
 * whether execution may fall back to another supported path when the primary backend cannot execute a
 * fused region.</p>
 *
 * @param primaryBackend preferred fused execution backend; {@code null} becomes {@link FusedPrimaryBackend#ASM}
 * @param allowBackendFallback whether fallback execution is allowed when the primary backend is unsupported
 */
public record FusedExecutionPolicy(
        FusedPrimaryBackend primaryBackend,
        boolean allowBackendFallback
) {
    public FusedExecutionPolicy {
        primaryBackend = primaryBackend == null ? FusedPrimaryBackend.ASM : primaryBackend;
    }

    /**
     * @return fused execution defaults for training-capable execution
     */
    public static FusedExecutionPolicy defaultsTraining() {
        return new FusedExecutionPolicy(
                FusedPrimaryBackend.ASM,
                true
        );
    }

    /**
     * @return fused execution defaults for forward-only inference
     */
    public static FusedExecutionPolicy defaultsInference() {
        return new FusedExecutionPolicy(
                FusedPrimaryBackend.ASM,
                true
        );
    }

    /**
     * Returns a copy with a different primary backend.
     *
     * @param value replacement primary backend; {@code null} becomes {@link FusedPrimaryBackend#ASM}
     * @return updated policy
     */
    public FusedExecutionPolicy withPrimaryBackend(FusedPrimaryBackend value) {
        return new FusedExecutionPolicy(value, allowBackendFallback);
    }

    /**
     * Returns a copy with a different fallback setting.
     *
     * @param value replacement fallback setting
     * @return updated policy
     */
    public FusedExecutionPolicy withAllowBackendFallback(boolean value) {
        return new FusedExecutionPolicy(primaryBackend, value);
    }
}
