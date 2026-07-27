package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.NodeId;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Exposes immutable successful-compile diagnostics without publishing internal predicates.
 *
 * <p>The public projection preserves final deferred-constraint order and deterministic diagnostic
 * text. The exact immutable internal constraints remain privately retained for later
 * compiler-owned binding validation. This type is not a trace payload, predicate language,
 * binding API, warning taxonomy, rejection result, or serialization format.</p>
 */
public final class CompileDiagnostics {
    private final List<DeferredGraphConstraint> constraints;
    private final List<DeferredConstraintDiagnostic> deferredConstraints;

    /**
     * Creates public diagnostic projections from exact final deferred constraints.
     *
     * @param constraints non-null ordered final constraints to snapshot without null elements
     * @throws NullPointerException if the list or an element is {@code null}
     */
    CompileDiagnostics(List<DeferredGraphConstraint> constraints) {
        Objects.requireNonNull(constraints, "constraints");
        List<DeferredGraphConstraint> retained = new ArrayList<>(constraints.size());
        List<DeferredConstraintDiagnostic> diagnostics = new ArrayList<>(constraints.size());
        for (int index = 0; index < constraints.size(); index++) {
            DeferredGraphConstraint constraint = Objects.requireNonNull(
                    constraints.get(index), "constraints[" + index + "]");
            retained.add(constraint);
            diagnostics.add(new DeferredConstraintDiagnostic(
                    constraint.nodeId(),
                    constraint.subject(),
                    constraint.predicate().toString()));
        }
        this.constraints = List.copyOf(retained);
        this.deferredConstraints = List.copyOf(diagnostics);
    }

    /**
     * Returns ordered public deferred-constraint diagnostics.
     *
     * @return immutable non-null ordered membership snapshot with one projection per exact
     *     retained internal constraint
     */
    public List<DeferredConstraintDiagnostic> deferredConstraints() {
        return deferredConstraints;
    }

    /**
     * Returns exact internal deferred constraints for same-package compiler validation.
     *
     * @return immutable non-null ordered snapshot retaining exact internal constraint references
     */
    List<DeferredGraphConstraint> constraints() {
        return constraints;
    }

    /**
     * Describes one deferred graph obligation without exposing its internal predicate type.
     *
     * @param nodeId non-null exact owning graph-node identity
     * @param subject non-null, nonblank semantic role
     * @param predicate non-null, nonblank deterministic diagnostic rendering
     */
    public record DeferredConstraintDiagnostic(
            NodeId nodeId,
            String subject,
            String predicate) {
        /**
         * Validates one immutable public diagnostic.
         *
         * @param nodeId non-null exact owning node identity
         * @param subject non-null, nonblank semantic role
         * @param predicate non-null, nonblank deterministic predicate rendering
         * @throws NullPointerException if a component is {@code null}
         * @throws IllegalArgumentException if {@code subject} or {@code predicate} is blank
         */
        public DeferredConstraintDiagnostic {
            Objects.requireNonNull(nodeId, "nodeId");
            Objects.requireNonNull(subject, "subject");
            Objects.requireNonNull(predicate, "predicate");
            if (subject.isBlank()) {
                throw new IllegalArgumentException("subject must not be blank");
            }
            if (predicate.isBlank()) {
                throw new IllegalArgumentException("predicate must not be blank");
            }
        }
    }
}
