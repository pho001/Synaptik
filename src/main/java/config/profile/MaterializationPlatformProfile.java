package config.profile;

public record MaterializationPlatformProfile(
        int contiguousMaterializeThreshold,
        int cheapF64MaterializeThreshold,
        int cheapF32MaterializeThreshold,
        int cheapBF16MaterializeThreshold,
        int whereMaterializeThreshold
) {
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
