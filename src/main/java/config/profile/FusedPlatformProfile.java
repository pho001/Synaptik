package config.profile;

public record FusedPlatformProfile(
        int fusedCheapVectorMinSize,
        int fusedTranscendentalVectorMinSize,
        int fusedCheapParallelMinSize,
        int fusedTranscendentalParallelMinSize
) {}
