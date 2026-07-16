package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.ValueId;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable successful result of captured-graph semantic verification.
 *
 * <p>The result retains the exact accepted graph reference and owns an immutable ordered snapshot
 * of unresolved constraints. Acceptance does not bind dimensions, make the graph executable, or
 * create a public compile artifact.
 *
 * @param constantGraph exact non-null accepted graph and compiler-owned source facts
 * @param constraints non-null ordered unresolved constraints, snapshot on construction
 */
record ValidatedGraph(
        CompileTimeConstantGraph constantGraph, List<DeferredGraphConstraint> constraints) {
    /**
     * Creates a result retaining the accepted graph and snapshotting unresolved constraints.
     *
     * @param constantGraph exact non-null accepted graph and source facts
     * @param constraints non-null ordered unresolved constraints, without null elements
     * @throws NullPointerException if either argument or a constraint is null
     */
    ValidatedGraph {
        Objects.requireNonNull(constantGraph, "constantGraph");
        constraints = List.copyOf(Objects.requireNonNull(constraints, "constraints"));
    }

    /**
     * Preserves same-package graph-only construction with no compile-time constant facts.
     *
     * @param graph exact non-null accepted graph reference
     * @param constraints non-null ordered unresolved constraints, snapshot on construction
     * @throws NullPointerException if either argument or a constraint is null
     */
    ValidatedGraph(CompiledGraphModel graph, List<DeferredGraphConstraint> constraints) {
        this(CompileTimeConstantGraph.withoutConstants(graph), constraints);
    }

    /**
     * Returns the accepted structural graph.
     *
     * @return exact non-null graph reference inside {@link #constantGraph()}
     */
    CompiledGraphModel graph() {
        return constantGraph.graph();
    }

    /**
     * Returns the immutable fixed-source splat facts.
     *
     * @return exact non-null immutable fact map inside {@link #constantGraph()}
     */
    Map<ValueId, CompileTimeConstantGraph.Splat> constants() {
        return constantGraph.constants();
    }

    /**
     * Derives caller-bindable sources in graph-input order.
     *
     * @return a new non-null immutable list excluding every fixed constant source
     */
    List<ValueId> bindableInputs() {
        return constantGraph.bindableInputs();
    }
}
