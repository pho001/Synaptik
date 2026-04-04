package graph.optimizer.rules;

public final class RewriteRule extends graph.optimizer.rewrite.RewriteRule {
    public RewriteRule() {
        super(java.util.List.of(
                new graph.optimizer.rewrite.AlgebraicRewrite(),
                new graph.optimizer.rewrite.LinearLoweringRewrite()
        ));
    }
}
