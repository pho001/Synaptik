package backend.cuda;

/**
 * CUDA dtype role classification. Residency roles do not imply native CUDA compute support.
 */
public enum CudaDTypeRole {
    COMPUTE_INPUT,
    COMPUTE_OUTPUT,
    INDEX_INPUT,
    PREDICATE_INPUT,
    RESIDENCY_ONLY
}
