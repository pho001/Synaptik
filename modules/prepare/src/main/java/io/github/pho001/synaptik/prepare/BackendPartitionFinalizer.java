package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;

/**
 * Finalizes one typed backend analysis after shared resource assignment is complete.
 *
 * <p>A concrete backend implements this collaboration without changing its selected route or
 * adding shared requirements. Finalization may construct immutable Java recipe state, but it does
 * not allocate or acquire closeable physical resources under the current contract.</p>
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
     * Constructs one immutable executable recipe from a fully validated assignment.
     *
     * @param finalization non-null typed analysis and complete shared assignment
     * @return non-null immutable executable retaining exactly the supplied memory plan
     * @throws IllegalArgumentException if backend-private immutable state is incompatible with
     *     the validated finalization
     */
    PreparedExecutable finalizePartition(BackendPartitionFinalization<P> finalization);
}
