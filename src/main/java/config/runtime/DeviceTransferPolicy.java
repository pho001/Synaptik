package config.runtime;

/**
 * Runtime policy for host/device transfers that cross CPU storage residency boundaries.
 */
public enum DeviceTransferPolicy {
    /**
     * Existing behavior: Java array bridges are allowed when direct native/device transfer is unavailable.
     */
    ALLOW_ARRAY_BRIDGE,

    /**
     * Prefer direct native/device transfer when implemented, but allow a traced array bridge fallback.
     */
    PREFER_DIRECT,

    /**
     * Require a direct native/device transfer. Array bridge fallback is treated as an execution error.
     */
    REQUIRE_DIRECT;

    /**
     * Returns whether this policy permits Java array bridge fallback.
     *
     * @return true when fallback via CPU array is allowed
     */
    public boolean allowsArrayBridge() {
        return this != REQUIRE_DIRECT;
    }
}
