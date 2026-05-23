package config.runtime;

/**
 * Runtime policy for fused elementwise execution.
 *
 * @param allowBackendFallback whether fallback execution is allowed when generated ASM cannot execute a fused region
 */
public record FusedExecutionPolicy(boolean allowBackendFallback) {
    public static FusedExecutionPolicy defaultsTraining() {
        return new FusedExecutionPolicy(true);
    }

    public static FusedExecutionPolicy defaultsInference() {
        return new FusedExecutionPolicy(true);
    }

    public FusedExecutionPolicy withAllowBackendFallback(boolean value) {
        return new FusedExecutionPolicy(value);
    }
}
