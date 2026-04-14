package config.profile;

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

    public int fusedAsmVectorWidth() {
        return fusedCheapContiguousAsmVectorWidth;
    }
}
