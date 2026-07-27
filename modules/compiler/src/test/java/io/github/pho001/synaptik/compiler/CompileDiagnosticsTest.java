package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

final class CompileDiagnosticsTest {
    @Test
    void projectsOrderedDiagnosticsWhilePrivatelyRetainingExactConstraints() {
        NodeId nodeId = new NodeId(4);
        DeferredGraphConstraint constraint = new DeferredGraphConstraint(
                nodeId,
                "input Shape",
                new DimensionEqual(
                        new DynamicDimension("N"),
                        new DynamicDimension("M")));
        List<DeferredGraphConstraint> source = new ArrayList<>(List.of(constraint));

        CompileDiagnostics diagnostics = new CompileDiagnostics(source);
        source.clear();

        assertSame(constraint, diagnostics.constraints().getFirst());
        assertSame(nodeId, diagnostics.deferredConstraints().getFirst().nodeId());
        assertEquals(
                "input Shape",
                diagnostics.deferredConstraints().getFirst().subject());
        assertTrue(!diagnostics.deferredConstraints().getFirst().predicate().isBlank());
        assertThrows(
                UnsupportedOperationException.class,
                () -> diagnostics.deferredConstraints().clear());
    }

    @Test
    void validatesPublicDiagnosticComponentsAndBlankText() {
        NodeId nodeId = new NodeId(1);
        assertEquals(
                "nodeId",
                assertThrows(
                        NullPointerException.class,
                        () -> new CompileDiagnostics.DeferredConstraintDiagnostic(
                                null, null, null))
                        .getMessage());
        assertEquals(
                "subject must not be blank",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new CompileDiagnostics.DeferredConstraintDiagnostic(
                                nodeId, " ", "predicate"))
                        .getMessage());
        assertEquals(
                "predicate must not be blank",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new CompileDiagnostics.DeferredConstraintDiagnostic(
                                nodeId, "subject", " "))
                        .getMessage());
    }
}
