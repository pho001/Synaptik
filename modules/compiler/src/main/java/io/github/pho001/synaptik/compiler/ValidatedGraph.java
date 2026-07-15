package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import java.util.List;
import java.util.Objects;

/**
 * Immutable successful result of captured-graph semantic verification.
 *
 * <p>The result retains the exact accepted graph reference and owns an immutable ordered snapshot
 * of unresolved constraints. Acceptance does not bind dimensions, make the graph executable, or
 * create a public compile artifact.
 *
 * @param graph exact non-null accepted graph reference
 * @param constraints non-null ordered unresolved constraints, snapshot on construction
 */
record ValidatedGraph(CompiledGraphModel graph, List<DeferredGraphConstraint> constraints) {
    /**
     * Creates a result retaining the accepted graph and snapshotting unresolved constraints.
     *
     * @param graph exact non-null accepted graph reference
     * @param constraints non-null ordered unresolved constraints, without null elements
     * @throws NullPointerException if either argument or a constraint is null
     */
    ValidatedGraph {
        Objects.requireNonNull(graph, "graph");
        constraints = List.copyOf(Objects.requireNonNull(constraints, "constraints"));
    }
}
