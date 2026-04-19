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
    NEVER,
    IF_MISSING,
    FORCE
}
