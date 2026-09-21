package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;

/**
 * Finalizes one typed backend analysis after shared resource assignment is complete.
 *
 * <p>A concrete backend implements this collaboration without changing its selected route or
 * adding shared requirements. A finalizer owns every persistent resource it acquires until its
 * complete result returns successfully. If any {@link RuntimeException} or {@link Error} occurs
 * before that return, including while constructing the result, it must close all acquired
 * resource identities once in reverse acquisition order and attempt every close. The original
 * failure remains primary; distinct cleanup failures are suppressed in reverse-cleanup encounter
 * order, and the exact primary throwable is skipped to avoid self-suppression. A successful
 * return transfers the acquisition-ordered resources to shared Prepare.</p>
 *
 * @param <P> concrete backend-owned immutable selected-plan role
 */
public interface BackendPartitionFinalizer<P extends BackendPreparationPlan> {
    /**
     * Returns the immutable identity of the implementing backend.
     *
     * @return non-null backend identity matched to planned-partition ownership
     */
    BackendId backendId();

    /**
     * Constructs one immutable executable recipe and its persistent resources from a fully
     * validated assignment.
     *
     * @param finalization non-null typed analysis and complete shared assignment
     * @return non-null atomic result containing an executable that retains exactly the supplied
     *     memory plan and its acquisition-ordered persistent resources
     * @throws NullPointerException if {@code finalization} is {@code null}
     * @throws IllegalArgumentException if backend-private immutable state is incompatible with
     *     the validated finalization
     * @throws RuntimeException if finalization or finalizer-local rollback fails; the original
     *     finalization failure remains primary and distinct rollback failures are suppressed
     * @throws Error if finalization fails with an error; finalizer-local rollback follows the
     *     same primary and suppression rules
     */
    BackendPartitionFinalizationResult finalizePartition(
            BackendPartitionFinalization<P> finalization);
}
