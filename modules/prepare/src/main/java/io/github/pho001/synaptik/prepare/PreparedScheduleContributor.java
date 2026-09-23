package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.backend.contract.BackendId;

/**
 * Supplies one backend owner's assigned physical creation recipes to complete schedule assembly.
 *
 * <p>The collaboration is immutable and thread-safe. It neither assembles global schedule order
 * nor allocates a representation. Shared composition invokes it after every partition finalizes,
 * while Prepare still owns complete contribution validation and the surrounding persistent-resource
 * transaction.</p>
 */
public interface PreparedScheduleContributor {
    /**
     * Returns the exact backend identity represented by this contributor.
     *
     * @return non-null immutable backend identity
     */
    BackendId backendId();

    /**
     * Contributes every buffer-representation and workspace creator assigned to this backend.
     *
     * @param context exact non-null complete finalized schedule context
     * @return non-null immutable sparse contribution owned by {@link #backendId()}
     * @throws NullPointerException if {@code context} is null
     * @throws IllegalArgumentException if assigned facts are unsupported or inconsistent
     * @throws IllegalStateException if the backend integration has closed
     */
    PreparedScheduleContribution contribute(PreparedScheduleContext context);
}
