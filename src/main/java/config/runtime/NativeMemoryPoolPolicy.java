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
     * Reuse released native CPU blocks across executions of one prepared plan.
     */
    PER_PREPARED_EXECUTION
}
