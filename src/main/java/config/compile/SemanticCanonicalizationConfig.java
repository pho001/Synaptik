package config.compile;

import config.optimizer.RewriteConfig;

/**
 * Required semantic canonicalization policy for graph construction.
 */
public record SemanticCanonicalizationConfig(
        boolean enabled,
        RewriteConfig rewrite
) {
    public SemanticCanonicalizationConfig {
        rewrite = rewrite == null ? RewriteConfig.defaults() : rewrite;
    }

    public static SemanticCanonicalizationConfig defaults() {
        return new SemanticCanonicalizationConfig(true, RewriteConfig.defaults());
    }

    public static SemanticCanonicalizationConfig disabled() {
        return new SemanticCanonicalizationConfig(false, RewriteConfig.defaults());
    }
}
