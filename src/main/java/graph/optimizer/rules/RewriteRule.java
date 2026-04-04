package graph.optimizer.rules;

import config.optimizer.RewriteConfig;

public final class RewriteRule extends graph.optimizer.rewrite.RewriteRule {
    public RewriteRule() {
        super(RewriteConfig.defaults());
    }

    public RewriteRule(RewriteConfig config) {
        super(config);
    }
}
