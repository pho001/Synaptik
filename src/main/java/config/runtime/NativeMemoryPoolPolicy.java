package config.runtime;

/**
 * Native CPU memory reuse policy.
 */
public enum NativeMemoryPoolPolicy {
    /**
     * Allocate each native CPU tensor storage independently and release it at run end.
     */
    DISABLED,

    /**
     * Reuse released native CPU blocks within one execute run.
     */
    PER_EXECUTION,

    /**
     * Reserved for a later wave; accepted in config but not pooled yet.
     */
    PER_PREPARED_EXECUTION
}
