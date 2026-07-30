package io.github.pho001.synaptik.prepare.analysis;

/**
 * Analyzes one validated planned partition using one concrete backend.
 *
 * <p>An implementation deterministically lowers the partition and selects a supported route and
 * private configuration from the complete {@link PrepareContext}. It must perform no measurement,
 * tuning search, cache mutation, physical allocation, or executable construction. The returned
 * analysis must retain the exact {@link PrepareContext#partition()} reference, an immutable opaque
 * plan for later finalization, and every exact shared buffer/workspace requirement.</p>
 *
 * <p>The generic roles preserve the association between one backend's explicit immutable inputs
 * and its selected opaque plan without casts, {@code Object}, or a shared parameter map. Shared
 * Prepare does not inspect either role.</p>
 *
 * @param <I> concrete backend-owned immutable analysis-input role
 * @param <P> corresponding concrete backend-owned immutable selected-plan role
 */
public interface BackendPartitionPreparer<
        I extends BackendAnalysisInputs,
        P extends BackendPreparationPlan> {
    /**
     * Deterministically analyzes one complete validated partition projection.
     *
     * @param context non-null complete immutable context; its exact partition reference must be
     *     returned in the analysis
     * @return non-null immutable analysis containing the exact context partition, an opaque plan,
     *     and exact shared requirements; never an executable, assigned slot, or physical resource
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if the backend cannot realize the supplied partition or
     *     complete inputs with one supported deterministic plan
     */
    BackendPartitionAnalysis<P> analyze(PrepareContext<I> context);
}
