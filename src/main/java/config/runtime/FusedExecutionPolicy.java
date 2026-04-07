package config.runtime;

public record FusedExecutionPolicy(
        FusedPrimaryBackend primaryBackend,
        boolean allowBackendFallback,
        boolean preferDirectForCompareSelect
) {
    public FusedExecutionPolicy {
        primaryBackend = primaryBackend == null ? FusedPrimaryBackend.DIRECT_VECTOR : primaryBackend;
    }

    public static FusedExecutionPolicy defaultsTraining() {
        return new FusedExecutionPolicy(
                FusedPrimaryBackend.DIRECT_VECTOR,
                true,
                true
        );
    }

    public static FusedExecutionPolicy defaultsInference() {
        return new FusedExecutionPolicy(
                FusedPrimaryBackend.ASM,
                true,
                false
        );
    }

    public FusedExecutionPolicy withPrimaryBackend(FusedPrimaryBackend value) {
        return new FusedExecutionPolicy(value, allowBackendFallback, preferDirectForCompareSelect);
    }

    public FusedExecutionPolicy withAllowBackendFallback(boolean value) {
        return new FusedExecutionPolicy(primaryBackend, value, preferDirectForCompareSelect);
    }

    public FusedExecutionPolicy withPreferDirectForCompareSelect(boolean value) {
        return new FusedExecutionPolicy(primaryBackend, allowBackendFallback, value);
    }
}
