package config.profile;

/**
 * Calibrated thresholds for converting layout/view tensors into materialized contiguous buffers.
 *
 * <p>Materialization can improve downstream execution when strided or broadcasted access would be more
 * expensive than copying. The thresholds are split by generic contiguous conversion, cheap dtype-specific
 * elementwise paths, and {@code where} operations.</p>
 *
 * @param contiguousMaterializeThreshold generic contiguous materialization threshold
 * @param cheapF64MaterializeThreshold cheap FLOAT64 materialization threshold
 * @param cheapF32MaterializeThreshold cheap FLOAT32 materialization threshold
 * @param cheapBF16MaterializeThreshold cheap BF16 materialization threshold
 * @param whereMaterializeThreshold materialization threshold for where/select-style operations
 */
public record MaterializationPlatformProfile(
        int contiguousMaterializeThreshold,
        int cheapF64MaterializeThreshold,
        int cheapF32MaterializeThreshold,
        int cheapBF16MaterializeThreshold,
        int whereMaterializeThreshold
) {
    /**
     * Creates a materialization profile that uses one threshold for every materialization family.
     *
     * @param contiguousMaterializeThreshold threshold applied to all materialization categories
     */
    public MaterializationPlatformProfile(int contiguousMaterializeThreshold) {
        this(
                contiguousMaterializeThreshold,
                contiguousMaterializeThreshold,
                contiguousMaterializeThreshold,
                contiguousMaterializeThreshold,
                contiguousMaterializeThreshold
        );
    }
}
