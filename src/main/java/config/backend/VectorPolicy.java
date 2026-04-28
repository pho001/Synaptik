package config.backend;

/**
 * Generic vectorization override used by backend configuration surfaces.
 */
public enum VectorPolicy {
    /**
     * Let runtime dispatch choose scalar or vector execution from calibrated thresholds.
     */
    AUTO,
    /**
     * Force vector execution when a vector implementation exists.
     */
    FORCE_ON,
    /**
     * Disable vector execution and use scalar paths.
     */
    FORCE_OFF
}
