package io.github.pho001.synaptik.backend.cpu.internal.memory;

import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import java.util.Optional;

/**
 * Cold CPU-private workspace creation recipes retained by finalized executable children.
 * A creator describes fresh run-owned resources; obtaining it neither allocates a workspace
 * nor transfers a persistent prepared resource. Shared schedule assembly invokes the retained
 * creator before binding, and the resulting RunState owns the representation's close duty.
 */
public interface CpuWorkspaceRecipeProvider {
    /**
     * Looks up a specialized factory for a workspace assigned to this exact executable recipe.
     *
     * @param workspaceIndex zero-based workspace slot in the complete prepared memory plan,
     *     not a child-local selection position
     * @return non-null optional creator; empty when this recipe does not supply that slot,
     *     leaving ordinary workspace creation unchanged. A present creator borrows the
     *     executable's persistent resources and returns a fresh caller-owned representation.
     *     Invoking it rejects inconsistent geometry with IllegalArgumentException and fails with
     *     IllegalStateException when required native resources are closed
     */
    Optional<PreparedRepresentationPlan.WorkspaceCreator> workspaceCreator(int workspaceIndex);
}
