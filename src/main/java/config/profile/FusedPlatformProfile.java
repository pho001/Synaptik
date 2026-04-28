package config.profile;

/**
 * Calibrated runtime settings for fused elementwise execution.
 *
 * <p>Fusion joins compatible elementwise operations into one generated loop. This profile decides when
 * those fused loops are vectorized or parallelized and which generated ASM vector width is used for
 * cheap/non-cheap and contiguous/strided variants.</p>
 *
 * @param fusedCheapVectorMinSize minimum fused cheap-operation size before vector dispatch
 * @param fusedTranscendentalVectorMinSize minimum fused transcendental-operation size before vector dispatch
 * @param fusedCheapParallelMinSize minimum fused cheap-operation size before parallel dispatch
 * @param fusedTranscendentalParallelMinSize minimum fused transcendental-operation size before parallel dispatch
 * @param fusedCheapContiguousAsmVectorWidth generated ASM vector width for cheap contiguous fused loops
 * @param fusedCheapStridedAsmVectorWidth generated ASM vector width for cheap strided fused loops
 * @param fusedNonCheapContiguousAsmVectorWidth generated ASM vector width for non-cheap contiguous fused loops
 * @param fusedNonCheapStridedAsmVectorWidth generated ASM vector width for non-cheap strided fused loops
 */
public record FusedPlatformProfile(
        int fusedCheapVectorMinSize,
        int fusedTranscendentalVectorMinSize,
        int fusedCheapParallelMinSize,
        int fusedTranscendentalParallelMinSize,
        int fusedCheapContiguousAsmVectorWidth,
        int fusedCheapStridedAsmVectorWidth,
        int fusedNonCheapContiguousAsmVectorWidth,
        int fusedNonCheapStridedAsmVectorWidth
) {
    /**
     * Creates a fused profile that uses one ASM vector width for all fused loop categories.
     *
     * @param fusedCheapVectorMinSize cheap-operation vector threshold
     * @param fusedTranscendentalVectorMinSize transcendental-operation vector threshold
     * @param fusedCheapParallelMinSize cheap-operation parallel threshold
     * @param fusedTranscendentalParallelMinSize transcendental-operation parallel threshold
     * @param fusedAsmVectorWidth ASM vector width applied to all fused loop categories
     */
    public FusedPlatformProfile(
            int fusedCheapVectorMinSize,
            int fusedTranscendentalVectorMinSize,
            int fusedCheapParallelMinSize,
            int fusedTranscendentalParallelMinSize,
            int fusedAsmVectorWidth
    ) {
        this(
                fusedCheapVectorMinSize,
                fusedTranscendentalVectorMinSize,
                fusedCheapParallelMinSize,
                fusedTranscendentalParallelMinSize,
                fusedAsmVectorWidth,
                fusedAsmVectorWidth,
                fusedAsmVectorWidth,
                fusedAsmVectorWidth
        );
    }

    /**
     * Returns the legacy aggregate ASM width value.
     *
     * <p>New calibration code stores separate widths per fused-loop category. This accessor exposes the
     * cheap contiguous width for older consumers that still need a single value.</p>
     *
     * @return cheap contiguous ASM vector width
     */
    public int fusedAsmVectorWidth() {
        return fusedCheapContiguousAsmVectorWidth;
    }
}
