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
 * @param buffer native buffer-binding policy for this backend; {@code null} uses defaults
 */
public record AcceleratorBackendConfig(
        boolean enabled,
        boolean requireRuntimeAvailability,
        long minimumEstimatedWork,
        AcceleratorBufferConfig buffer
) {
    public AcceleratorBackendConfig {
        minimumEstimatedWork = Math.max(0L, minimumEstimatedWork);
        buffer = buffer == null ? AcceleratorBufferConfig.defaults() : buffer;
    }

    /**
     * Creates a backend policy using default buffer-binding settings.
     *
     * @param enabled whether the backend is eligible
     * @param requireRuntimeAvailability whether selection must require a successful availability probe
     * @param minimumEstimatedWork minimum estimated work before this backend is eligible
     */
    public AcceleratorBackendConfig(
            boolean enabled,
            boolean requireRuntimeAvailability,
            long minimumEstimatedWork
    ) {
        this(enabled, requireRuntimeAvailability, minimumEstimatedWork, AcceleratorBufferConfig.defaults());
    }

    /**
     * @return default enabled backend policy
     */
    public static AcceleratorBackendConfig defaults() {
        return new AcceleratorBackendConfig(true, false, 0L, AcceleratorBufferConfig.defaults());
    }

    /**
     * @return disabled backend policy
     */
    public static AcceleratorBackendConfig disabled() {
        return new AcceleratorBackendConfig(false, false, 0L, AcceleratorBufferConfig.disabled());
    }

    /**
     * Returns a copy with a different enabled flag.
     *
     * @param newEnabled replacement enabled flag
     * @return updated config
     */
    public AcceleratorBackendConfig withEnabled(boolean newEnabled) {
        return new AcceleratorBackendConfig(newEnabled, requireRuntimeAvailability, minimumEstimatedWork, buffer);
    }

    /**
     * Returns a copy with a different runtime-availability requirement.
     *
     * @param newRequireRuntimeAvailability replacement availability requirement
     * @return updated config
     */
    public AcceleratorBackendConfig withRequireRuntimeAvailability(boolean newRequireRuntimeAvailability) {
        return new AcceleratorBackendConfig(enabled, newRequireRuntimeAvailability, minimumEstimatedWork, buffer);
    }

    /**
     * Returns a copy with a different minimum estimated work threshold.
     *
     * @param newMinimumEstimatedWork replacement threshold; negative values are normalized to {@code 0}
     * @return updated config
     */
    public AcceleratorBackendConfig withMinimumEstimatedWork(long newMinimumEstimatedWork) {
        return new AcceleratorBackendConfig(enabled, requireRuntimeAvailability, newMinimumEstimatedWork, buffer);
    }

    /**
     * Returns a copy with a different buffer-binding policy.
     *
     * @param newBuffer replacement policy; {@code null} uses defaults
     * @return updated config
     */
    public AcceleratorBackendConfig withBuffer(AcceleratorBufferConfig newBuffer) {
        return new AcceleratorBackendConfig(enabled, requireRuntimeAvailability, minimumEstimatedWork, newBuffer);
    }
}
