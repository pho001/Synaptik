package tuning.benchmark.report;

/**
 * CUDA-specific classification for hot-path exits in parity planning.
 */
public enum CudaHotPathBlockerClass {
    V16_BLOCKER,
    ACCEPTED_CAPABILITY_GAP,
    FUTURE_SCOPE,
    REQUIRES_NATIVE_EVIDENCE
}
