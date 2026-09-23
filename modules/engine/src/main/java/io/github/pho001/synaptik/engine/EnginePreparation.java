package io.github.pho001.synaptik.engine;

import java.util.Objects;

/** Cold single-owner preparation result carrying the direct selected Engine adapter. */
record EnginePreparation(
        io.github.pho001.synaptik.runtime.execution.PreparedExecution execution,
        EngineBackendComposition adapter) {
    /**
     * Retains one inward recipe and its direct non-owning adapter reference.
     *
     * @param execution non-null inward Runtime owner
     * @param adapter non-null selected adapter owned by the surrounding Engine registry
     * @throws NullPointerException if either component is {@code null}
     */
    EnginePreparation {
        Objects.requireNonNull(execution, "execution");
        Objects.requireNonNull(adapter, "adapter");
    }
}
