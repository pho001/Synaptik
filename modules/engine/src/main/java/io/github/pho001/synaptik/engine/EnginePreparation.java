package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import java.util.List;
import java.util.Objects;

/**
 * Cold ordinary preparation result carrying direct adapters for each external boundary
 * occurrence.
 */
record EnginePreparation(
        PreparedExecution execution,
        List<EngineBackendComposition> inputAdapters,
        List<EngineBackendComposition> publicationAdapters) {
    /**
     * Retains one inward recipe plus input- and publication-order direct adapter snapshots.
     *
     * @param execution non-null inward Runtime owner
     * @param inputAdapters non-null exact adapter per caller-input occurrence
     * @param publicationAdapters non-null exact adapter per publication occurrence
     * @throws NullPointerException if a component or indexed adapter is null
     */
    EnginePreparation {
        Objects.requireNonNull(execution, "execution");
        inputAdapters = List.copyOf(inputAdapters);
        publicationAdapters = List.copyOf(publicationAdapters);
    }
}
