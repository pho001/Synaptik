package tensor;

/**
 * Policy for the convenience compute surface.
 *
 * NEVER:
 * - execute with the resolved default optimizer/runtime profile
 *
 * IF_MISSING:
 * - reuse a cached generic best-profile for this graph if present
 * - otherwise run graph autotune once and persist the winner
 *
 * FORCE:
 * - always rerun graph autotune for this graph before execution
 */
public enum AutotunePolicy {
    /** Never run autotuning from the convenience compute API. */
    NEVER,
    /** Reuse cached tuning results when present, otherwise tune once. */
    IF_MISSING,
    /** Always rerun autotuning before executing through convenience compute. */
    FORCE
}
