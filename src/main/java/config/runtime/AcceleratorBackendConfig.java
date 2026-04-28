package config.runtime;

/**
 * Runtime selection policy for one accelerator backend.
 *
 * <p>This configuration does not load native libraries by itself. Backend selection reads it to decide
 * whether a backend is eligible, whether availability must be proven, and how large an operation should
 * be before the backend is considered. Negative work thresholds are normalized to {@code 0}.</p>
 *
 * @param enabled whether the backend is eligible
 * @param requireRuntimeAvailability whether selection must require a successful availability probe
 * @param minimumEstimatedWork minimum estimated work before this backend is eligible
 */
public record AcceleratorBackendConfig(
        boolean enabled,
        boolean requireRuntimeAvailability,
        long minimumEstimatedWork
) {
    public AcceleratorBackendConfig {
        minimumEstimatedWork = Math.max(0L, minimumEstimatedWork);
    }

    /**
     * @return default enabled backend policy
     */
    public static AcceleratorBackendConfig defaults() {
        return new AcceleratorBackendConfig(true, false, 0L);
    }

    /**
     * @return disabled backend policy
     */
    public static AcceleratorBackendConfig disabled() {
        return new AcceleratorBackendConfig(false, false, 0L);
    }

    /**
     * Returns a copy with a different enabled flag.
     *
     * @param newEnabled replacement enabled flag
     * @return updated config
     */
    public AcceleratorBackendConfig withEnabled(boolean newEnabled) {
        return new AcceleratorBackendConfig(newEnabled, requireRuntimeAvailability, minimumEstimatedWork);
    }

    /**
     * Returns a copy with a different runtime-availability requirement.
     *
     * @param newRequireRuntimeAvailability replacement availability requirement
     * @return updated config
     */
    public AcceleratorBackendConfig withRequireRuntimeAvailability(boolean newRequireRuntimeAvailability) {
        return new AcceleratorBackendConfig(enabled, newRequireRuntimeAvailability, minimumEstimatedWork);
    }

    /**
     * Returns a copy with a different minimum estimated work threshold.
     *
     * @param newMinimumEstimatedWork replacement threshold; negative values are normalized to {@code 0}
     * @return updated config
     */
    public AcceleratorBackendConfig withMinimumEstimatedWork(long newMinimumEstimatedWork) {
        return new AcceleratorBackendConfig(enabled, requireRuntimeAvailability, newMinimumEstimatedWork);
    }
}
