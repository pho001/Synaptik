package io.github.pho001.synaptik.config.compile;

/**
 * Records whether a later compiler may apply optional semantics-preserving graph optimizations.
 *
 * <p>Disabling optional optimization does not suppress graph capture, ordering, inference,
 * validation, mandatory canonical representation, mode-required autograd, publication binding,
 * backend-neutral planning, preparation, or execution. Enabling it permits a compiler-owned
 * standard pipeline without making its passes, order, internal graph shape, or implementation
 * strategy part of this public contract.</p>
 *
 * <p>This value grants no permission for approximate mathematics, changed numerical semantics,
 * backend-specific fusion, preparation, or execution behavior. Equality, hashing, and
 * diagnostic text follow ordinary record semantics over the primitive permission.</p>
 *
 * @param optionalOptimizationsEnabled {@code true} to permit a later compiler's standard optional
 *     semantics-preserving optimization pipeline; {@code false} to request that such optional
 *     work be skipped
 */
public record GraphOptimizationConfig(boolean optionalOptimizationsEnabled) {
    /**
     * Creates an optimization configuration retaining the exact primitive permission.
     *
     * @param optionalOptimizationsEnabled {@code true} to permit a later compiler's standard
     *     optional semantics-preserving optimization pipeline; {@code false} to request that such
     *     optional work be skipped
     */
    public GraphOptimizationConfig(boolean optionalOptimizationsEnabled) {
        this.optionalOptimizationsEnabled = optionalOptimizationsEnabled;
    }

    /**
     * Creates a configuration that requests skipping optional semantics-preserving graph
     * optimizations.
     *
     * @return a new disabled optimization configuration; never {@code null}
     */
    public static GraphOptimizationConfig disabled() {
        return new GraphOptimizationConfig(false);
    }

    /**
     * Creates a configuration that permits the compiler's standard optional semantics-preserving
     * optimization pipeline.
     *
     * @return a new standard optimization configuration; never {@code null}
     */
    public static GraphOptimizationConfig standard() {
        return new GraphOptimizationConfig(true);
    }

    /**
     * Reports whether a later compiler may apply its standard optional semantics-preserving graph
     * optimization pipeline.
     *
     * @return {@code true} when standard optional optimization is permitted; {@code false} when
     *     optional optimization should be skipped
     */
    public boolean optionalOptimizationsEnabled() {
        return optionalOptimizationsEnabled;
    }
}
