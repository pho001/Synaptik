package config.optimizer;

public record CseConfig(
        boolean strictSafety
) {
    public static CseConfig strictDefaults() {
        return new CseConfig(true);
    }

    public static CseConfig aggressiveDefaults() {
        return new CseConfig(false);
    }
}
