package io.github.pho001.synaptik.runtime.run;

import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import java.util.List;

/**
 * Gives the CPU physical-provider fixture direct test-only access to the actual Runtime setup
 * owner. This package-local friendship avoids reflection, copied creation/rollback logic and any
 * production API or executor. No source roles, slots, resources or numerical work are invented.
 */
public final class CpuRunStateCreationTestAccess {
    /** Prevents construction of the stateless test bridge. */
    private CpuRunStateCreationTestAccess() {}

    /**
     * Delegates complete creation, ownership validation and rollback to the actual Runtime owner.
     * @param plan actual prepared representation recipe, non-null
     * @param callers actual borrowed caller representations in recipe order, non-null
     * @param geometry authenticated sizing for the recipe's exact memory plan, non-null
     * @return the real open run state, owning created resources and borrowing caller inputs
     * @throws NullPointerException if an argument, caller or creator result is null
     * @throws IllegalArgumentException if geometry, caller coverage or resource identities disagree
     * @throws RuntimeException if actual resource creation or rollback reports failure
     * @throws Error if actual resource creation or rollback reports an error
     */
    public static RunState create(PreparedRepresentationPlan plan,
            List<BufferRepresentation> callers, RunMemoryGeometry geometry) {
        return RunStateCreation.create(plan, callers, geometry);
    }
}
