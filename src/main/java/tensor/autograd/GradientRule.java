package tensor.autograd;

/**
 * Typed backward rule used to construct semantic gradient graph nodes.
 */
@FunctionalInterface
public interface GradientRule {
    void apply(GradientContext context);
}
