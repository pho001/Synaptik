package io.github.pho001.synaptik.engine;

import java.util.Objects;

/**
 * Immutable reusable ordinary prepared handle owned by one exact {@link Engine}.
 *
 * <p>Each run through this handle receives isolated Runtime state. The inward prepared recipe is
 * private, the handle owns no per-run resource, and its originating compile metadata remains
 * readable after Engine closure. The handle has no independent close lifecycle; Engine closure
 * prevents new runs without invalidating immutable metadata.</p>
 */
public final class PreparedExecution {
    private final Engine owner;
    private final CompiledGraph compiledGraph;
    private final io.github.pho001.synaptik.runtime.execution.PreparedExecution execution;

    /**
     * Retains the exact ordinary owner, compile handle, and immutable Runtime recipe.
     *
     * @param owner non-null exact Engine owner
     * @param compiledGraph non-null exact originating compile handle
     * @param execution non-null immutable inward prepared recipe
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the compiled handle has another owner
     */
    PreparedExecution(
            Engine owner,
            CompiledGraph compiledGraph,
            io.github.pho001.synaptik.runtime.execution.PreparedExecution execution) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.compiledGraph = Objects.requireNonNull(compiledGraph, "compiledGraph");
        this.execution = Objects.requireNonNull(execution, "execution");
        if (compiledGraph.owner() != owner) {
            throw new IllegalArgumentException("compiled graph belongs to another engine");
        }
    }

    /**
     * Returns the exact ordinary compile handle from which this recipe was prepared.
     *
     * @return the retained non-null originating handle by reference identity
     */
    public CompiledGraph compiledGraph() {
        return compiledGraph;
    }

    Engine owner() {
        return owner;
    }

    io.github.pho001.synaptik.runtime.execution.PreparedExecution execution() {
        return execution;
    }
}
