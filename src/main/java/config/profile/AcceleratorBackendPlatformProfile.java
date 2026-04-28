package config.profile;

import config.runtime.AcceleratorBackendConfig;

/**
 * Calibrated policy for one accelerator backend on the current platform.
 *
 * <p>The profile records whether the backend is enabled, whether runtime availability must be proven
 * before selection, and the minimum estimated work at which the backend may be considered. Negative
 * work thresholds are normalized to {@code 0}.</p>
 *
 * @param enabled whether this accelerator backend may be selected
 * @param requireRuntimeAvailability whether selection must require a successful runtime availability check
 * @param minimumEstimatedWork minimum estimated operation work before this backend is eligible
 */
public record AcceleratorBackendPlatformProfile(
        boolean enabled,
        boolean requireRuntimeAvailability,
        long minimumEstimatedWork
) {
    public AcceleratorBackendPlatformProfile {
        minimumEstimatedWork = Math.max(0L, minimumEstimatedWork);
    }

    /**
     * Creates the default accelerator-backend profile from runtime defaults.
     *
     * @return default backend platform profile
     */
    public static AcceleratorBackendPlatformProfile defaults() {
        return fromRuntimeConfig(AcceleratorBackendConfig.defaults());
    }

    /**
     * Converts a runtime accelerator backend config into a persisted platform profile section.
     *
     * @param config runtime config; {@code null} uses runtime defaults
     * @return platform profile section with normalized threshold
     */
    public static AcceleratorBackendPlatformProfile fromRuntimeConfig(AcceleratorBackendConfig config) {
        AcceleratorBackendConfig resolved = config == null ? AcceleratorBackendConfig.defaults() : config;
        return new AcceleratorBackendPlatformProfile(
                resolved.enabled(),
                resolved.requireRuntimeAvailability(),
                resolved.minimumEstimatedWork()
        );
    }

    /**
     * Converts this platform profile section back into runtime config.
     *
     * @return runtime accelerator backend config
     */
    public AcceleratorBackendConfig toRuntimeConfig() {
        return new AcceleratorBackendConfig(enabled, requireRuntimeAvailability, minimumEstimatedWork);
    }
}
