package config.runtime;

public record AcceleratorBackendConfig(
        boolean enabled,
        boolean requireRuntimeAvailability,
        long minimumEstimatedWork
) {
    public AcceleratorBackendConfig {
        minimumEstimatedWork = Math.max(0L, minimumEstimatedWork);
    }

    public static AcceleratorBackendConfig defaults() {
        return new AcceleratorBackendConfig(true, false, 0L);
    }

    public static AcceleratorBackendConfig disabled() {
        return new AcceleratorBackendConfig(false, false, 0L);
    }

    public AcceleratorBackendConfig withEnabled(boolean newEnabled) {
        return new AcceleratorBackendConfig(newEnabled, requireRuntimeAvailability, minimumEstimatedWork);
    }

    public AcceleratorBackendConfig withRequireRuntimeAvailability(boolean newRequireRuntimeAvailability) {
        return new AcceleratorBackendConfig(enabled, newRequireRuntimeAvailability, minimumEstimatedWork);
    }

    public AcceleratorBackendConfig withMinimumEstimatedWork(long newMinimumEstimatedWork) {
        return new AcceleratorBackendConfig(enabled, requireRuntimeAvailability, newMinimumEstimatedWork);
    }
}
