package io.github.pho001.synaptik.planning.capability;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.CompiledScanBody;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class ScanActivityPlanTest {
    private static final TensorDescriptor VALUE =
            new TensorDescriptor(DataType.FLOAT32, Shape.of(2), Optional.empty(), false);
    private static final NodeId ADD = new NodeId(7);
    private static final List<List<Integer>> SCOPES = List.of(List.of(0), List.of(1), List.of(0, 1));

    @Test
    void snapshotsNestedMembershipAndPreservesDeclarationRoots() {
        CompiledGraphModel graph = graph(BinaryArithmeticKind.ADD);
        List<Integer> union = new ArrayList<>(List.of(1, 0));
        List<List<Integer>> scopes = new ArrayList<>(List.of(List.of(0), List.of(1), union));
        Map<NodeId, Integer> nodes = new LinkedHashMap<>(Map.of(ADD, 2));
        Map<NodeId, ScanActivityPlan.Merge> merges = new LinkedHashMap<>(
                Map.of(ADD, new ScanActivityPlan.Merge(0, 1)));
        List<ScanActivityPlan.Carry> carries = new ArrayList<>(
                List.of(new ScanActivityPlan.Carry(0, false, 2)));
        ScanActivityPlan plan = new ScanActivityPlan(graph, carries, List.of(1), scopes, nodes, merges, List.of());
        union.clear();
        scopes.clear();
        nodes.clear();
        merges.clear();
        carries.clear();
        assertSame(graph, plan.graph());
        assertEquals(SCOPES, plan.scopes());
        assertEquals(List.of(new ScanActivityPlan.Carry(0, false, 2)), plan.carries());
        assertThrows(UnsupportedOperationException.class, () -> plan.scopes().get(2).clear());
        assertThrows(UnsupportedOperationException.class, () -> plan.scopes().clear());
        assertThrows(UnsupportedOperationException.class, () -> plan.nodeScopes().clear());
        assertThrows(UnsupportedOperationException.class, () -> plan.merges().clear());
    }

    @Test
    void rejectsIncompleteForeignAndInvalidScopeMembership() {
        CompiledGraphModel graph = graph(BinaryArithmeticKind.ADD);
        assertThrows(IllegalArgumentException.class, () -> plan(graph, SCOPES, Map.of(), Map.of()));
        assertThrows(IllegalArgumentException.class,
                () -> plan(graph, SCOPES, Map.of(new NodeId(99), 2), Map.of()));
        assertThrows(IllegalArgumentException.class,
                () -> plan(graph, SCOPES, Map.of(ADD, 3), Map.of()));
        assertThrows(IllegalArgumentException.class,
                () -> plan(graph, List.of(List.of(0), List.of(1), List.of(0, 0)), Map.of(ADD, 2), Map.of()));
        assertThrows(IllegalArgumentException.class,
                () -> plan(graph, List.of(List.of(0), List.of(1), List.of(2)), Map.of(ADD, 2), Map.of()));
        assertThrows(IllegalArgumentException.class,
                () -> plan(graph, SCOPES, Map.of(ADD, 2), Map.of(new NodeId(99), new ScanActivityPlan.Merge(0, 1))));
    }

    @Test
    void rejectsForgedMergeShapesUnionAndUnavailableOrdinaryOperands() {
        assertThrows(IllegalArgumentException.class,
                () -> plan(graph(BinaryArithmeticKind.SUB), SCOPES, Map.of(ADD, 2), Map.of(ADD, new ScanActivityPlan.Merge(0, 1))));
        assertThrows(IllegalArgumentException.class,
                () -> plan(graph(BinaryArithmeticKind.ADD), SCOPES, Map.of(ADD, 0), Map.of(ADD, new ScanActivityPlan.Merge(0, 1))));
        assertThrows(IllegalArgumentException.class,
                () -> plan(graph(BinaryArithmeticKind.ADD), SCOPES, Map.of(ADD, 2), Map.of()));
        assertThrows(IllegalArgumentException.class,
                () -> plan(graph(BinaryArithmeticKind.ADD), SCOPES, Map.of(ADD, 2), Map.of(ADD, new ScanActivityPlan.Merge(1, 0))));
    }

    @Test
    void rejectsOrdinaryBroadcastAndPromotionAtRawOptionalMergeBoundary() {
        TensorDescriptor scalar =
                new TensorDescriptor(DataType.FLOAT32, Shape.scalar(), Optional.empty(), false);
        TensorDescriptor wide =
                new TensorDescriptor(DataType.FLOAT64, VALUE.shape(), Optional.empty(), false);
        Map<NodeId, Integer> nodes = Map.of(ADD, 2);
        Map<NodeId, ScanActivityPlan.Merge> merges = Map.of(ADD, new ScanActivityPlan.Merge(0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> plan(graph(BinaryArithmeticKind.ADD, scalar, VALUE, VALUE), SCOPES, nodes, merges));
        assertThrows(IllegalArgumentException.class,
                () -> plan(graph(BinaryArithmeticKind.ADD, VALUE, scalar, VALUE), SCOPES, nodes, merges));
        assertThrows(IllegalArgumentException.class,
                () -> plan(graph(BinaryArithmeticKind.ADD, VALUE, wide, wide), SCOPES, nodes, merges));
        assertThrows(IllegalArgumentException.class,
                () -> plan(graph(BinaryArithmeticKind.ADD, wide, VALUE, wide), SCOPES, nodes, merges));
        CompiledGraphModel exact = graph(BinaryArithmeticKind.ADD, wide, wide, wide);
        assertSame(exact, plan(exact, SCOPES, nodes, merges).graph());
    }

    @Test
    void rejectsDuplicateAndOutOfRangeDeclarationPositions() {
        CompiledGraphModel graph = graph(BinaryArithmeticKind.ADD);
        assertThrows(IllegalArgumentException.class, () -> new ScanActivityPlan(graph,
                List.of(new ScanActivityPlan.Carry(0, true, 2), new ScanActivityPlan.Carry(0, false, 2)),
                List.of(1), SCOPES, Map.of(ADD, 2), Map.of(), List.of()));
        assertThrows(IllegalArgumentException.class, () -> new ScanActivityPlan(graph,
                List.of(new ScanActivityPlan.Carry(0, true, 2)), List.of(1, 1), SCOPES,
                Map.of(ADD, 2), Map.of(), List.of()));
        assertThrows(IllegalArgumentException.class, () -> new ScanActivityPlan(graph,
                List.of(new ScanActivityPlan.Carry(0, true, 2)), List.of(2), SCOPES,
                Map.of(ADD, 2), Map.of(), List.of()));
        assertThrows(IllegalArgumentException.class, () -> new ScanActivityPlan(graph,
                List.of(new ScanActivityPlan.Carry(0, true, 2)), List.of(1), SCOPES,
                Map.of(ADD, 2), Map.of(), List.of(new ScanActivityPlan.Accumulator(0, 2))));
    }

    @Test
    void queryRequiresTheExactAttachedBodyNotAnEqualGraph() {
        CompiledGraphModel graph = graph(BinaryArithmeticKind.ADD);
        ScanActivityPlan plan = plan(graph, SCOPES, Map.of(ADD, 2), Map.of(ADD, new ScanActivityPlan.Merge(0, 1)));
        CompiledScanBody body = new CompiledScanBody(graph, graph.outputs(), graph.outputs(), List.of(), List.of(0));
        Operation operation = new Operation(ScanKind.SCAN, new ScanAttrs(3, 1, 1, 0, 1, 0, false));
        OperationCapabilityQuery query = new OperationCapabilityQuery(operation, List.of(VALUE, VALUE),
                List.of(VALUE, VALUE), Optional.of(body), Optional.of(plan));
        assertSame(plan, query.scanActivityPlan().orElseThrow());
        CompiledGraphModel equalGraph = graph(BinaryArithmeticKind.ADD);
        assertEquals(graph, equalGraph);
        CompiledScanBody foreign = new CompiledScanBody(equalGraph, equalGraph.outputs(), equalGraph.outputs(), List.of(), List.of(0));
        assertThrows(IllegalArgumentException.class, () -> new OperationCapabilityQuery(operation,
                query.inputs(), query.outputs(), Optional.of(foreign), Optional.of(plan)));
        assertThrows(IllegalArgumentException.class, () -> new OperationCapabilityQuery(operation,
                query.inputs(), query.outputs(), Optional.empty(), Optional.of(plan)));
        assertThrows(NullPointerException.class, () -> new OperationCapabilityQuery(operation,
                query.inputs(), query.outputs(), Optional.of(body), null));
        ScanActivityPlan wrongNext = new ScanActivityPlan(graph,
                List.of(new ScanActivityPlan.Carry(0, true, 0)), plan.seedFormals(), plan.scopes(),
                plan.nodeScopes(), plan.merges(), plan.accumulators());
        assertThrows(IllegalArgumentException.class, () -> new OperationCapabilityQuery(operation,
                query.inputs(), query.outputs(), Optional.of(body), Optional.of(wrongNext)));
        Operation wrongHeader = new Operation(ScanKind.SCAN, new ScanAttrs(3, 0, 2, 0, 2, 0, false));
        assertThrows(IllegalArgumentException.class, () -> new OperationCapabilityQuery(wrongHeader,
                query.inputs(), query.outputs(), Optional.of(body), Optional.of(plan)));
    }

    private static ScanActivityPlan plan(CompiledGraphModel graph, List<List<Integer>> scopes,
            Map<NodeId, Integer> nodes, Map<NodeId, ScanActivityPlan.Merge> merges) {
        return new ScanActivityPlan(graph, List.of(new ScanActivityPlan.Carry(0, true, 2)),
                List.of(1), scopes, nodes, merges, List.of());
    }

    private static CompiledGraphModel graph(BinaryArithmeticKind kind) {
        return graph(kind, VALUE, VALUE, VALUE);
    }

    private static CompiledGraphModel graph(BinaryArithmeticKind kind, TensorDescriptor leftDescriptor,
            TensorDescriptor rightDescriptor, TensorDescriptor resultDescriptor) {
        ValueId left = new ValueId(0), right = new ValueId(1), result = new ValueId(2);
        CompiledNode node = new CompiledNode(ADD, new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(left, right), List.of(result));
        return new CompiledGraphModel(List.of(new GraphValue(left, leftDescriptor),
                new GraphValue(right, rightDescriptor), new GraphValue(result, resultDescriptor)),
                List.of(node), List.of(left, right), List.of(result),
                Map.of(ADD, GraphPhase.BACKWARD));
    }
}
