package config.profile;

import config.runtime.AcceleratorBackendConfig;

public record AcceleratorBackendPlatformProfile(
        boolean enabled,
        boolean requireRuntimeAvailability,
        long minimumEstimatedWork
) {
    public AcceleratorBackendPlatformProfile {
        minimumEstimatedWork = Math.max(0L, minimumEstimatedWork);
    }

    public static AcceleratorBackendPlatformProfile defaults() {
        return fromRuntimeConfig(AcceleratorBackendConfig.defaults());
    }

    public static AcceleratorBackendPlatformProfile fromRuntimeConfig(AcceleratorBackendConfig config) {
        AcceleratorBackendConfig resolved = config == null ? AcceleratorBackendConfig.defaults() : config;
        return new AcceleratorBackendPlatformProfile(
                resolved.enabled(),
                resolved.requireRuntimeAvailability(),
                resolved.minimumEstimatedWork()
        );
    }

    public AcceleratorBackendConfig toRuntimeConfig() {
        return new AcceleratorBackendConfig(enabled, requireRuntimeAvailability, minimumEstimatedWork);
    }
}
