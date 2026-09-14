package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import java.util.Objects;

/**
 * Opaque immutable handle for a prepared recipe owned by one {@link AdvancedEngine}.
 *
 * <p>The handle may be shared safely by concurrent runs through its exact owning Engine. Each run
 * still receives isolated mutable Runtime state. The handle exposes neither its owner nor its
 * inward Runtime recipe publicly, has no independent close operation, and owns no closeable
 * state. Closing its Engine makes the handle unusable.</p>
 */
public final class AdvancedPreparedExecution {
    private final AdvancedEngine owner;
    private final PreparedExecution execution;

    /**
     * Retains one exact Engine owner and immutable Runtime recipe.
     *
     * @param owner non-null exact Engine that created and exclusively consumes this handle
     * @param execution non-null immutable prepared recipe retained without copying
     * @throws NullPointerException if either argument is null
     */
    AdvancedPreparedExecution(AdvancedEngine owner, PreparedExecution execution) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.execution = Objects.requireNonNull(execution, "execution");
    }

    /**
     * Returns the exact creating Engine for package-private identity validation.
     *
     * @return the non-null exact owner
     */
    AdvancedEngine owner() {
        return owner;
    }

    /**
     * Returns the retained Runtime recipe only to package-private orchestration.
     *
     * @return the non-null immutable prepared execution
     */
    PreparedExecution execution() {
        return execution;
    }
}
