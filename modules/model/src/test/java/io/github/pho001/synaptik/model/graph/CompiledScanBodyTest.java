package io.github.pho001.synaptik.model.graph;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CompiledScanBodyTest {
    @Test
    void lengthNodeCountsExtraActualButNeverALengthsFormal() {
        var attrs = new ScanAttrs(4, 1, 1, 0, 1, 1, false, true, 1, List.of(java.util.OptionalInt.of(0)));
        var op = new Operation(ScanKind.SCAN, attrs);
        var b = body();
        var inputs = List.of(H, X, new ValueId(2));
        var outputs = List.of(new ValueId(3), new ValueId(4), new ValueId(5));
        var n = new CompiledNode(new NodeId(0), op, inputs, outputs, Optional.of(b));
        assertSame(b, n.scanBody().orElseThrow());
        assertEquals(2, b.graph().inputs().size());
        assertThrows(IllegalArgumentException.class, () -> new CompiledNode(n.id(), op, List.of(H, X), outputs, Optional.of(b)));
        var g = b.graph(); var extra = new ValueId(2);
        var values = new ArrayList<>(g.values()); values.add(new GraphValue(extra, g.values().getFirst().descriptor()));
        var wrong = new CompiledScanBody(new CompiledGraphModel(values, g.nodes(), List.of(H, X, extra), g.outputs(), g.nodePhases()),
                b.nextCarry(), b.emissions(), b.residuals(), b.demandedOutputIndices());
        assertThrows(IllegalArgumentException.class, () -> new CompiledNode(n.id(), op, inputs, outputs, Optional.of(wrong)));
    }
    private static final ValueId H = new ValueId(0), X = new ValueId(1);

    private static CompiledGraphModel graph() {
        TensorDescriptor d = new TensorDescriptor(DataType.FLOAT32, Shape.of(2), Optional.empty(), true);
        return new CompiledGraphModel(List.of(new GraphValue(H, d), new GraphValue(X, d)),
                List.of(), List.of(H, X), List.of(H), Map.of());
    }

    private static CompiledScanBody body() {
        return new CompiledScanBody(graph(), List.of(H), List.of(H), List.of(H), List.of(0, 2));
    }

    private static Operation scan() {
        return new Operation(ScanKind.SCAN, new ScanAttrs(4, 1, 1, 0, 1, 1, false));
    }

    private static CompiledNode node(CompiledScanBody body) {
        // Local numeric input/output IDs intentionally coincide with independently scoped IDs.
        return new CompiledNode(new NodeId(0), scan(), List.of(H, X), List.of(H, X, new ValueId(2)), Optional.of(body));
    }

    @Test
    void snapshotsRepeatableRolesAndDemandsWithoutDeduplicatingFormalInputs() {
        List<ValueId> roles = new ArrayList<>(List.of(H, H));
        List<Integer> demand = new ArrayList<>(List.of(1, 3));
        CompiledGraphModel graph = graph();
        CompiledScanBody b = new CompiledScanBody(graph, roles, List.of(H), List.of(H), demand);
        roles.clear();
        demand.clear();
        assertSame(graph, b.graph());
        assertEquals(List.of(H, H), b.nextCarry());
        assertEquals(List.of(H, X), b.graph().inputs()); // unused X remains at its formal position
        assertEquals(List.of(H), b.graph().outputs());
        assertEquals(List.of(1, 3), b.demandedOutputIndices());
        assertThrows(UnsupportedOperationException.class, () -> b.nextCarry().clear());
        assertThrows(UnsupportedOperationException.class, () -> b.demandedOutputIndices().clear());
    }

    @Test
    void rejectsNonlocalOrNonresultBindingsAndMissingOutputCoverage() {
        assertThrows(IllegalArgumentException.class, () -> new CompiledScanBody(graph(), List.of(X), List.of(), List.of(), List.of()));
        assertThrows(IllegalArgumentException.class, () -> new CompiledScanBody(graph(), List.of(new ValueId(99)), List.of(H), List.of(), List.of()));
        assertThrows(IllegalArgumentException.class, () -> new CompiledScanBody(graph(), List.of(), List.of(), List.of(H), List.of()));
    }

    @Test void emissionOnlyCompiledStructureKeepsNoCarryAndExactFormalCardinality() {
        var g = graph();
        var local = new CompiledGraphModel(List.of(g.values().getFirst()), List.of(), List.of(H), List.of(H), Map.of());
        var b = new CompiledScanBody(local, List.of(), List.of(H, H), List.of(), List.of(0, 1));
        var op = new Operation(ScanKind.SCAN, new ScanAttrs(0, 0, 1, 0, 2, 0, true));
        var node = new CompiledNode(new NodeId(0), op, List.of(X), List.of(H, X), Optional.of(b));
        assertTrue(node.scanBody().orElseThrow().nextCarry().isEmpty());
        assertEquals(List.of(H, H), b.emissions());
        assertThrows(IllegalArgumentException.class, () -> new CompiledNode(node.id(), op,
                List.of(H, X), node.outputs(), Optional.of(b)));
    }

    @Test
    void rejectsInvalidDemandOrderDuplicatesAndBounds() {
        for (List<Integer> demand : List.of(List.of(-1), List.of(3), List.of(1, 1), List.of(2, 0))) {
            assertThrows(IllegalArgumentException.class,
                    () -> new CompiledScanBody(graph(), List.of(H), List.of(H), List.of(H), demand));
        }
        assertTrue(new CompiledScanBody(graph(), List.of(H), List.of(), List.of(), List.of()).demandedOutputIndices().isEmpty());
    }

    @Test
    void validatesAttachmentKindsHeadersAndEveryNullBoundary() {
        CompiledScanBody b = body();
        CompiledNode n = node(b);
        assertSame(b, n.scanBody().orElseThrow());
        assertThrows(IllegalArgumentException.class, () -> new CompiledNode(n.id(), n.operation(), n.inputs(), n.outputs()));
        Operation ordinary = new Operation(UnaryElementwiseKind.TANH, NoOperationAttrs.INSTANCE);
        assertTrue(new CompiledNode(n.id(), ordinary, List.of(H), List.of(X)).scanBody().isEmpty());
        assertThrows(IllegalArgumentException.class, () -> new CompiledNode(n.id(), ordinary, List.of(H), List.of(X), Optional.of(b)));
        assertThrows(NullPointerException.class, () -> new CompiledNode(n.id(), n.operation(), n.inputs(), n.outputs(), null));
        assertThrows(IllegalArgumentException.class, () -> new CompiledNode(n.id(), new Operation(ScanKind.SCAN,
                new ScanAttrs(4, 1, 1, 1, 1, 1, false)), List.of(H, X, X), n.outputs(), Optional.of(b)));
        assertThrows(IllegalArgumentException.class, () -> new CompiledNode(n.id(), n.operation(), n.inputs(), List.of(H, X), Optional.of(b)));
        assertThrows(NullPointerException.class, () -> new CompiledScanBody(null, List.of(H), List.of(), List.of(), List.of()));
        assertThrows(NullPointerException.class, () -> new CompiledScanBody(graph(), null, List.of(), List.of(), List.of()));
        assertThrows(NullPointerException.class, () -> new CompiledScanBody(graph(), List.of(H), null, List.of(), List.of()));
        assertThrows(NullPointerException.class, () -> new CompiledScanBody(graph(), List.of(H), List.of(), null, List.of()));
        assertThrows(NullPointerException.class, () -> new CompiledScanBody(graph(), List.of(H), List.of(), List.of(), null));
        assertThrows(NullPointerException.class, () -> new CompiledScanBody(graph(), List.of(H), Arrays.asList((ValueId) null), List.of(), List.of()));
        assertThrows(NullPointerException.class, () -> new CompiledScanBody(graph(), List.of(H), List.of(), List.of(), Arrays.asList((Integer) null)));
    }

    @Test
    void rejectsNestedBodiesButAllowsIndependentScopedNumberReuse() {
        CompiledNode n = node(body());
        TensorDescriptor d = graph().values().getFirst().descriptor();
        ValueId out = new ValueId(2), unused = new ValueId(3);
        CompiledNode inner = new CompiledNode(new NodeId(0), n.operation(), List.of(H, X),
                List.of(out, unused, new ValueId(4)), n.scanBody());
        CompiledGraphModel outer = new CompiledGraphModel(List.of(new GraphValue(H, d), new GraphValue(X, d),
                new GraphValue(out, d), new GraphValue(unused, d), new GraphValue(new ValueId(4), d)),
                List.of(inner), List.of(H, X), List.of(out), Map.of(inner.id(), GraphPhase.FORWARD));
        assertThrows(IllegalArgumentException.class,
                () -> new CompiledScanBody(outer, List.of(out), List.of(), List.of(), List.of()));
    }

}
