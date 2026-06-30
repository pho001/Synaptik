package trace.backend;

/**
 * Compute precision and backend metadata for a step.
 *
 * @param mode execution mode or precision mode label
 * @param storageType dtype used for storage
 * @param computeType dtype used for computation
 * @param backend backend name
 * @param accumulateType dtype used for accumulation
 */
public record ComputeTraceMetadata(
        String mode,
        String storageType,
        String computeType,
        String backend,
        String accumulateType
) {
    public ComputeTraceMetadata {
        mode = mode == null ? "" : mode;
        storageType = storageType == null ? "" : storageType;
        computeType = computeType == null ? "" : computeType;
        backend = backend == null ? "" : backend;
        accumulateType = accumulateType == null ? "" : accumulateType;
    }
}
