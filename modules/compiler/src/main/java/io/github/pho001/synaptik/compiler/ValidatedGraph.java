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
 * create a public compile artifact.</p>
 *
 * @param constantGraph exact non-null accepted graph and compiler-owned source facts
 * @param constraints non-null ordered unresolved constraints, snapshot on construction
 * @param derivatives non-null derivative-order metadata for the exact accepted graph
 */
record ValidatedGraph(
        CompileTimeConstantGraph constantGraph,
        List<DeferredGraphConstraint> constraints,
        DerivativeGraphMetadata derivatives) {
    /**
     * Creates a result retaining the accepted graph and snapshotting unresolved constraints.
     *
     * @param constantGraph exact non-null accepted graph and source facts
     * @param constraints non-null ordered unresolved constraints, without null elements
     * @param derivatives non-null derivative-order metadata for the exact accepted graph
     * @throws NullPointerException if an argument or constraint is null
     * @throws IllegalArgumentException if the metadata owns a different graph
     */
    ValidatedGraph {
        Objects.requireNonNull(constantGraph, "constantGraph");
        constraints = List.copyOf(Objects.requireNonNull(constraints, "constraints"));
        Objects.requireNonNull(derivatives, "derivatives");
        if (derivatives.graph() != constantGraph.graph()) {
            throw new IllegalArgumentException(
                    "derivatives graph must be the exact validated graph");
        }
    }

    /**
     * Creates a compatibility result whose derivative orders follow graph phases.
     *
     * @param constantGraph exact non-null accepted graph and source facts
     * @param constraints non-null ordered unresolved constraints
     */
    ValidatedGraph(
            CompileTimeConstantGraph constantGraph,
            List<DeferredGraphConstraint> constraints) {
        this(constantGraph, constraints, metadataFromPhases(constantGraph.graph()));
    }

    /**
     * Preserves same-package graph-only construction with no compile-time constant facts.
     *
     * @param graph exact non-null accepted graph reference
     * @param constraints non-null ordered unresolved constraints, snapshot on construction
     * @throws NullPointerException if either argument or a constraint is null
     */
    ValidatedGraph(CompiledGraphModel graph, List<DeferredGraphConstraint> constraints) {
        this(
                CompileTimeConstantGraph.withoutConstants(graph),
                constraints,
                metadataFromPhases(graph));
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

    private static DerivativeGraphMetadata metadataFromPhases(CompiledGraphModel graph) {
        java.util.LinkedHashMap<io.github.pho001.synaptik.model.graph.NodeId, Integer> orders =
                new java.util.LinkedHashMap<>();
        graph.nodes().forEach(node -> orders.put(
                node.id(),
                graph.nodePhases().get(node.id())
                                == io.github.pho001.synaptik.model.graph.GraphPhase.FORWARD
                        ? 0
                        : 1));
        return new DerivativeGraphMetadata(graph, orders);
    }
}
