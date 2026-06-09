package config.runtime;

/**
 * Runtime policy for fused elementwise execution.
 *
 * @param allowBackendFallback whether fallback execution is allowed when generated ASM cannot execute a fused region
 * @param useCpu1Elementwise whether CPU fused elementwise regions should prepare through the cpu1 fused path
 */
public record FusedExecutionPolicy(
        boolean allowBackendFallback,
        boolean useCpu1Elementwise
) {
    public FusedExecutionPolicy(boolean allowBackendFallback) {
        this(allowBackendFallback, false);
    }

    public static FusedExecutionPolicy defaultsTraining() {
        return new FusedExecutionPolicy(true, false);
    }

    public static FusedExecutionPolicy defaultsInference() {
        return new FusedExecutionPolicy(true, false);
    }

    public FusedExecutionPolicy withAllowBackendFallback(boolean value) {
        return new FusedExecutionPolicy(value, useCpu1Elementwise);
    }

    public FusedExecutionPolicy withUseCpu1Elementwise(boolean value) {
        return new FusedExecutionPolicy(allowBackendFallback, value);
    }
}
