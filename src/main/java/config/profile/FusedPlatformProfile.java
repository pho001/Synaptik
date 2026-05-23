package config.profile;

/**
 * Calibrated runtime settings for fused elementwise execution.
 *
 * @param fusedCheapVectorMinSize minimum fused cheap-operation size before vector dispatch
 * @param fusedTranscendentalVectorMinSize minimum fused transcendental-operation size before vector dispatch
 * @param fusedCheapParallelMinSize minimum fused cheap-operation size before parallel dispatch
 * @param fusedTranscendentalParallelMinSize minimum fused transcendental-operation size before parallel dispatch
 * @param fusedAsmVectorWidth calibrated preferred ASM vector width for fused loops
 */
public record FusedPlatformProfile(
        int fusedCheapVectorMinSize,
        int fusedTranscendentalVectorMinSize,
        int fusedCheapParallelMinSize,
        int fusedTranscendentalParallelMinSize,
        int fusedAsmVectorWidth
) {
}
