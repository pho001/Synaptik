package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparer;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import java.util.Objects;

/**
 * Associates one partition position with its typed backend analysis inputs and collaborations.
 *
 * <p>The immutable value retains all three exact references. It contains no compiler aggregate,
 * backend discovery, mutable state, or physical resource.</p>
 *
 * @param <I> concrete backend-owned immutable analysis-input role
 * @param <P> corresponding concrete backend-owned immutable selected-plan role
 * @param backendInputs exact non-null backend inputs used during analysis
 * @param preparer exact non-null typed analysis collaboration
 * @param finalizer exact non-null typed finalization collaboration
 */
public record PartitionPreparation<
        I extends BackendAnalysisInputs,
        P extends BackendPreparationPlan>(
        I backendInputs,
        BackendPartitionPreparer<I, P> preparer,
        BackendPartitionFinalizer<P> finalizer) {
    /**
     * Validates one positional backend preparation.
     *
     * @param backendInputs exact non-null immutable backend inputs to retain
     * @param preparer exact non-null preparer to retain
     * @param finalizer exact non-null finalizer to retain
     * @throws NullPointerException if a component is null, checked in declaration order
     */
    public PartitionPreparation {
        Objects.requireNonNull(backendInputs, "backendInputs");
        Objects.requireNonNull(preparer, "preparer");
        Objects.requireNonNull(finalizer, "finalizer");
    }
}
