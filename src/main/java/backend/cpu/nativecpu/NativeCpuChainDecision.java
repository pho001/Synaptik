package backend.cpu.nativecpu;

/**
 * Prepare-time decision for a native CPU chain segment.
 */
public enum NativeCpuChainDecision {
    NONE,
    REQUIRED_NATIVE,
    AUTO_FAST_NATIVE,
    AUTO_REJECTED_SLOW_OP,
    UNSUPPORTED_OP,
    MATERIALIZATION_BOUNDARY
}
