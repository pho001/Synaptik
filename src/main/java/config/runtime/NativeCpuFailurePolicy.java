package config.runtime;

/**
 * Failure policy for requested native CPU execution.
 */
public enum NativeCpuFailurePolicy {
    /**
     * Unsupported native CPU operations may fall back to the Java-array CPU path.
     */
    FALLBACK_TO_ARRAY,

    /**
     * Unsupported native CPU operations should fail instead of silently using the array path.
     */
    REQUIRE_NATIVE
}
