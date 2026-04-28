package config.optimizer;

/**
 * Common subexpression elimination safety configuration.
 *
 * <p>Strict CSE only merges expressions when conservative semantic checks pass. Aggressive CSE allows a
 * wider set of equivalent-looking graph nodes and is intended for inference-oriented profiles where the
 * optimizer can be less conservative.</p>
 *
 * @param strictSafety whether CSE must use strict safety checks
 */
public record CseConfig(
        boolean strictSafety
) {
    /**
     * @return strict CSE defaults
     */
    public static CseConfig strictDefaults() {
        return new CseConfig(true);
    }

    /**
     * @return aggressive CSE defaults
     */
    public static CseConfig aggressiveDefaults() {
        return new CseConfig(false);
    }
}
