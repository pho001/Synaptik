package config.runtime;

public record FusedExecutionPolicy(
        FusedPrimaryBackend primaryBackend,
        boolean allowBackendFallback
) {
    public FusedExecutionPolicy {
        primaryBackend = primaryBackend == null ? FusedPrimaryBackend.DIRECT_VECTOR : primaryBackend;
    }

    public static FusedExecutionPolicy defaultsTraining() {
        return new FusedExecutionPolicy(
                FusedPrimaryBackend.ASM,
                true
        );
    }

    public static FusedExecutionPolicy defaultsInference() {
        return new FusedExecutionPolicy(
                FusedPrimaryBackend.ASM,
                true
        );
    }

    public FusedExecutionPolicy withPrimaryBackend(FusedPrimaryBackend value) {
        return new FusedExecutionPolicy(value, allowBackendFallback);
    }

    public FusedExecutionPolicy withAllowBackendFallback(boolean value) {
        return new FusedExecutionPolicy(primaryBackend, value);
    }
}
