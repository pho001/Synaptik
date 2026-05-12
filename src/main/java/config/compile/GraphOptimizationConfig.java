package config.compile;

import config.optimizer.CseConfig;
import config.optimizer.RewriteConfig;

/**
 * Backend-neutral graph rewrite and cleanup policy.
 */
public record GraphOptimizationConfig(
        boolean algebraicRewrite,
        boolean constantFolding,
        boolean commonSubexpressionElimination,
        boolean deadCodeElimination,
        boolean optionalLowering,
        RewriteConfig rewrite,
        CseConfig cse
) {
    public GraphOptimizationConfig {
        rewrite = rewrite == null ? RewriteConfig.defaults() : rewrite;
        cse = cse == null ? CseConfig.strictDefaults() : cse;
    }

    public static GraphOptimizationConfig trainingDefaults() {
        return new GraphOptimizationConfig(
                true,
                true,
                true,
                true,
                true,
                RewriteConfig.defaults(),
                CseConfig.strictDefaults()
        );
    }

    public static GraphOptimizationConfig inferenceDefaults() {
        return new GraphOptimizationConfig(
                true,
                true,
                true,
                true,
                true,
                RewriteConfig.defaults(),
                CseConfig.aggressiveDefaults()
        );
    }

    public static GraphOptimizationConfig noGraphOptimization() {
        return new GraphOptimizationConfig(
                false,
                false,
                false,
                false,
                false,
                RewriteConfig.defaults(),
                CseConfig.strictDefaults()
        );
    }

    public static GraphOptimizationConfig stages(
            boolean algebraicRewrite,
            boolean constantFolding,
            boolean commonSubexpressionElimination,
            boolean deadCodeElimination,
            boolean optionalLowering
    ) {
        return new GraphOptimizationConfig(
                algebraicRewrite,
                constantFolding,
                commonSubexpressionElimination,
                deadCodeElimination,
                optionalLowering,
                RewriteConfig.defaults(),
                CseConfig.strictDefaults()
        );
    }

    public GraphOptimizationConfig withRewrite(RewriteConfig newRewrite) {
        return new GraphOptimizationConfig(
                algebraicRewrite,
                constantFolding,
                commonSubexpressionElimination,
                deadCodeElimination,
                optionalLowering,
                newRewrite,
                cse
        );
    }

    public GraphOptimizationConfig withCse(CseConfig newCse) {
        return new GraphOptimizationConfig(
                algebraicRewrite,
                constantFolding,
                commonSubexpressionElimination,
                deadCodeElimination,
                optionalLowering,
                rewrite,
                newCse
        );
    }
}
