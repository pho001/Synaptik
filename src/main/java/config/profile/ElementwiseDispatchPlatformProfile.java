package config.profile;

public record ElementwiseDispatchPlatformProfile(
        int cheapVectorMinSize,
        int transcendentalVectorMinSize,
        int cheapParallelMinSize,
        int transcendentalParallelMinSize
) {}
