package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.CompiledScanBody;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.planning.capability.ScanActivityPlan;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class DerivativeGraphMetadataTest {
    private static final TensorDescriptor VALUE = descriptor(DataType.FLOAT32, 2);
    private static final TensorDescriptor HISTORY = descriptor(DataType.FLOAT32, 3, 2);
    private static final TensorDescriptor INTEGER = descriptor(DataType.INT32, 2);
    private static final NodeId SCAN = new NodeId(99);
    private static final NodeId LOCAL_ADD = new NodeId(0);

    @Test
    void snapshotsContainingMapAndPreservesTheGenuineUnannotatedConstructor() {
        Fixture fixture = fixture();
        LinkedHashMap<NodeId, ScanActivityPlan> activity = new LinkedHashMap<>(Map.of(SCAN, fixture.plan()));
        DerivativeGraphMetadata metadata = new DerivativeGraphMetadata(
                fixture.graph(), orders(fixture.graph()), activity);
        activity.clear();
        assertSame(fixture.plan(), metadata.scanActivityByNode().get(SCAN));
        assertThrows(UnsupportedOperationException.class, () -> metadata.scanActivityByNode().clear());
        assertTrue(new DerivativeGraphMetadata(fixture.graph(), orders(fixture.graph()))
                .scanActivityByNode().isEmpty());
        assertThrows(IllegalArgumentException.class, () -> new DerivativeGraphMetadata(
                fixture.graph(), orders(fixture.graph()), Map.of(new NodeId(404), fixture.plan())));
        assertThrows(IllegalArgumentException.class, () -> new DerivativeGraphMetadata(
                fixture.graph(), orders(fixture.graph()), Map.of(new NodeId(10), fixture.plan())));
        assertThrows(IllegalArgumentException.class, () -> new DerivativeGraphMetadata(
                fixture.graph(), orders(fixture.graph()), Map.of(SCAN, fixture().plan())));
    }

    @Test
    void everyOuterOptimizationRetainsExactLocalActivityAcrossRenumbering() {
        Fixture fixture = fixture();
        DerivativeGraphMetadata metadata = metadata(fixture);
        GraphCanonicalization.Result canonical = GraphCanonicalization.canonicalize(metadata);
        assertTransport(canonical.derivatives(), fixture.plan());
        assertFalse(canonical.derivatives().scanActivityByNode().containsKey(SCAN));
        ForwardExactArithmeticRewriting.Result rewritten = ForwardExactArithmeticRewriting.rewrite(metadata);
        assertTransport(rewritten.derivatives(), fixture.plan());
        ForwardCommonSubexpressionElimination.Result common =
                ForwardCommonSubexpressionElimination.DerivativeAware.eliminate(metadata);
        assertTransport(common.derivatives(), fixture.plan());
        ForwardConstantFolding.Result folded = ForwardConstantFolding.fold(fixture.constants(), metadata);
        assertTransport(folded.derivatives(), fixture.plan());
        ForwardDeadCodeElimination.Result dead = ForwardDeadCodeElimination.DerivativeAware.eliminate(
                fixture.constants(), metadata);
        assertTransport(dead.derivatives(), fixture.plan());
    }

    @Test
    void demandRefreshAndSameIdDescriptorClosuresRetainExactActivity() {
        Fixture fixture = fixture();
        ValidatedGraph input = CapturedGraphInference.inferAndValidate(fixture.constants(), metadata(fixture));
        ValidatedGraph demanded = ScanBodyInference.refreshDemand(input);
        CompiledScanBody demandedBody = demanded.graph().nodes().stream()
                .filter(node -> node.id().equals(SCAN)).findFirst().orElseThrow().scanBody().orElseThrow();
        assertEquals(List.of(0, 1), demandedBody.demandedOutputIndices());
        assertEquals(List.of(new ValueId(0), new ValueId(2)), demandedBody.residuals());
        assertTransport(demanded.derivatives(), fixture.plan());
        ValidatedGraph closed = StaticResultLogicalLayoutClosure.close(input);
        assertNotSame(input.graph(), closed.graph());
        assertTransport(closed.derivatives(), fixture.plan());
        ValueId publication = new ValueId(100);
        List<GraphValue> publishedValues = new ArrayList<>(fixture.graph().values());
        publishedValues.add(new GraphValue(publication, VALUE));
        List<ValueId> publishedInputs = new ArrayList<>(fixture.graph().inputs());
        publishedInputs.add(publication);
        List<ValueId> publishedOutputs = new ArrayList<>(fixture.graph().outputs());
        publishedOutputs.add(publication);
        CompiledGraphModel publishedGraph = new CompiledGraphModel(publishedValues, fixture.graph().nodes(),
                publishedInputs, publishedOutputs, fixture.graph().nodePhases());
        Map<ValueId, CompileTimeConstantGraph.Splat> constants = new LinkedHashMap<>(fixture.constants().constants());
        constants.put(publication, new CompileTimeConstantGraph.Splat(ScalarValue.float32(1)));
        ValidatedGraph published = CapturedGraphInference.inferAndValidate(
                new CompileTimeConstantGraph(publishedGraph, constants),
                new DerivativeGraphMetadata(publishedGraph, orders(publishedGraph), Map.of(SCAN, fixture.plan())));
        ValidatedGraph publishedClosed = PublishedCompileTimeConstantDescriptorClosure.close(published);
        assertNotSame(published.graph(), publishedClosed.graph());
        assertTransport(publishedClosed.derivatives(), fixture.plan());

        ValueId convolution = new ValueId(101);
        List<GraphValue> convolutionValues = new ArrayList<>(fixture.graph().values());
        convolutionValues.add(new GraphValue(convolution, descriptor(DataType.FLOAT32, 1, 1, 1, 2)));
        ValueId image = new ValueId(102), kernel = new ValueId(103);
        convolutionValues.add(new GraphValue(image, canonical(DataType.FLOAT32, 1, 1, 1, 2)));
        convolutionValues.add(new GraphValue(kernel, canonical(DataType.FLOAT32, 1, 1, 1, 1)));
        List<ValueId> convolutionInputs = new ArrayList<>(fixture.graph().inputs());
        convolutionInputs.add(image);
        convolutionInputs.add(kernel);
        List<CompiledNode> convolutionNodes = new ArrayList<>(fixture.graph().nodes());
        CompiledNode conv = new CompiledNode(new NodeId(100),
                new Operation(Conv2dKind.CONV2D, Conv2dAttrs.defaults()),
                List.of(image, kernel), List.of(convolution));
        convolutionNodes.add(conv);
        Map<NodeId, GraphPhase> phases = new LinkedHashMap<>(fixture.graph().nodePhases());
        phases.put(conv.id(), GraphPhase.BACKWARD);
        List<ValueId> convolutionOutputs = new ArrayList<>(fixture.graph().outputs());
        convolutionOutputs.add(convolution);
        CompiledGraphModel convolutionGraph = new CompiledGraphModel(convolutionValues, convolutionNodes,
                convolutionInputs, convolutionOutputs, phases);
        ValidatedGraph convolutionInput = CapturedGraphInference.inferAndValidate(
                new CompileTimeConstantGraph(convolutionGraph, fixture.constants().constants()),
                new DerivativeGraphMetadata(convolutionGraph, orders(convolutionGraph), Map.of(SCAN, fixture.plan())));
        ValidatedGraph convolutionClosed = ConvolutionLogicalLayoutClosure.close(convolutionInput);
        assertNotSame(convolutionInput.graph(), convolutionClosed.graph());
        assertTransport(convolutionClosed.derivatives(), fixture.plan());
    }

    @Test
    void removingAnUndemandedContainingScanRemovesOnlyItsActivity() {
        Fixture fixture = fixture();
        CompiledGraphModel graph = fixture.graph();
        CompiledGraphModel noScanOutput = new CompiledGraphModel(graph.values(), graph.nodes(),
                graph.inputs(), List.of(graph.inputs().getFirst()), graph.nodePhases());
        DerivativeGraphMetadata metadata = new DerivativeGraphMetadata(
                noScanOutput, orders(noScanOutput), Map.of(SCAN, fixture.plan()));
        ForwardDeadCodeElimination.Result dead = ForwardDeadCodeElimination.DerivativeAware.eliminate(
                new CompileTimeConstantGraph(noScanOutput, fixture.constants().constants()), metadata);
        assertTrue(dead.derivatives().scanActivityByNode().isEmpty());
        assertTrue(dead.constantGraph().graph().nodes().isEmpty());
    }

    @Test
    void containingBindingsRejectForgedNextScopeAndOptionalTemporalAdd() {
        Fixture fixture = fixture();
        ScanActivityPlan original = fixture.plan();
        ScanActivityPlan wrongNext = new ScanActivityPlan(original.graph(),
                List.of(new ScanActivityPlan.Carry(0, true, 0)), original.seedFormals(),
                original.scopes(), original.nodeScopes(), original.merges(), original.accumulators());
        assertThrows(IllegalArgumentException.class, () -> new DerivativeGraphMetadata(
                fixture.graph(), orders(fixture.graph()), Map.of(SCAN, wrongNext)));

        ValueId old = new ValueId(0), seed = new ValueId(1), next = new ValueId(2);
        CompiledNode add = new CompiledNode(LOCAL_ADD,
                new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE), List.of(old, seed), List.of(next));
        CompiledGraphModel local = new CompiledGraphModel(List.of(
                new GraphValue(old, VALUE), new GraphValue(seed, VALUE), new GraphValue(next, VALUE)),
                List.of(add), List.of(old, seed), List.of(next), Map.of(LOCAL_ADD, GraphPhase.BACKWARD));
        List<List<Integer>> scopes = List.of(List.of(), List.of(0));
        ScanActivityPlan temporal = new ScanActivityPlan(local, List.of(), List.of(1), scopes,
                Map.of(LOCAL_ADD, 1), Map.of(), List.of(new ScanActivityPlan.Accumulator(0, 1)));
        CompiledScanBody body = new CompiledScanBody(local, List.of(next), List.of(), List.of(), List.of(0));
        CompiledNode scan = new CompiledNode(SCAN,
                new Operation(ScanKind.SCAN, new ScanAttrs(3, 1, 1, 0, 0, 0, true)),
                List.of(old, seed), List.of(next), Optional.of(body));
        CompiledGraphModel outer = new CompiledGraphModel(List.of(
                new GraphValue(old, VALUE), new GraphValue(seed, HISTORY), new GraphValue(next, VALUE)),
                List.of(scan), List.of(old, seed), List.of(next), Map.of(SCAN, GraphPhase.BACKWARD));
        assertSame(temporal, new DerivativeGraphMetadata(outer, orders(outer), Map.of(SCAN, temporal))
                .scanActivityByNode().get(SCAN));
        ScanActivityPlan optionalTemporal = new ScanActivityPlan(local, List.of(), List.of(1), scopes,
                Map.of(LOCAL_ADD, 1), Map.of(LOCAL_ADD, new ScanActivityPlan.Merge(0, 1)),
                List.of(new ScanActivityPlan.Accumulator(0, 1)));
        assertThrows(IllegalArgumentException.class, () -> new DerivativeGraphMetadata(
                outer, orders(outer), Map.of(SCAN, optionalTemporal)));
    }

    private static void assertTransport(DerivativeGraphMetadata metadata, ScanActivityPlan plan) {
        CompiledNode scan = metadata.graph().nodes().stream()
                .filter(node -> node.scanBody().isPresent()).findFirst().orElseThrow();
        assertSame(plan, metadata.scanActivityByNode().get(scan.id()));
        assertSame(plan.graph(), scan.scanBody().orElseThrow().graph());
        assertEquals(Map.of(LOCAL_ADD, new ScanActivityPlan.Merge(0, 1)), plan.merges());
        assertEquals(List.of(List.of(0), List.of(1), List.of(0, 1)), plan.scopes());
    }

    private static DerivativeGraphMetadata metadata(Fixture fixture) {
        return new DerivativeGraphMetadata(fixture.graph(), orders(fixture.graph()), Map.of(SCAN, fixture.plan()));
    }

    private static LinkedHashMap<NodeId, Integer> orders(CompiledGraphModel graph) {
        LinkedHashMap<NodeId, Integer> orders = new LinkedHashMap<>();
        graph.nodes().forEach(node -> orders.put(node.id(), 1));
        return orders;
    }

    private static Fixture fixture() {
        Tensor sourceH = TensorFactory.create(canonical(DataType.FLOAT32, 2));
        Tensor sourceX = TensorFactory.create(canonical(DataType.FLOAT32, 2));
        Tensor sourceSum = sourceH.add(sourceX);
        ScanBody sourceBody = new ScanBody(List.of(sourceH), List.of(sourceX), List.of(),
                List.of(sourceSum), List.of(sourceSum));
        var sourceScan = Scan.apply(List.of(TensorFactory.create(canonical(DataType.FLOAT32, 2))),
                List.of(TensorFactory.create(canonical(DataType.FLOAT32, 3, 2))),
                List.of(), sourceBody, true);
        CompiledGraphModel captured = GraphCapture.capture(
                List.of(sourceScan.finalCarry().getFirst(), sourceScan.outputs().getFirst()));
        CompiledNode capturedScan = captured.nodes().getFirst();
        CompiledScanBody body = capturedScan.scanBody().orElseThrow();
        CompiledGraphModel local = body.graph();
        ValueId h = new ValueId(0), x = new ValueId(1);
        ScanActivityPlan plan = new ScanActivityPlan(local, List.of(new ScanActivityPlan.Carry(0, true, 2)),
                List.of(1), List.of(List.of(0), List.of(1), List.of(0, 1)), Map.of(LOCAL_ADD, 2),
                Map.of(LOCAL_ADD, new ScanActivityPlan.Merge(0, 1)), List.of());
        List<TensorDescriptor> capturedPorts = capturedScan.outputs().stream()
                .map(id -> captured.values().stream().filter(value -> value.id().equals(id))
                        .findFirst().orElseThrow().descriptor()).toList();
        ValueId integer = new ValueId(3), neg = new ValueId(4), duplicate = new ValueId(5);
        ValueId integralNeg = new ValueId(6), result = new ValueId(7), emission = new ValueId(8);
        ValueId savedH = new ValueId(9), savedSum = new ValueId(10);
        CompiledNode first = unary(10, h, neg);
        CompiledNode second = unary(11, h, duplicate);
        CompiledNode fold = new CompiledNode(new NodeId(12),
                new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                List.of(integer, integer), List.of(integralNeg));
        CompiledNode scan = new CompiledNode(SCAN,
                capturedScan.operation(),
                List.of(h, x), List.of(result, emission, savedH, savedSum), Optional.of(body));
        List<CompiledNode> nodes = List.of(first, second, fold, scan);
        LinkedHashMap<NodeId, GraphPhase> phases = new LinkedHashMap<>();
        nodes.forEach(node -> phases.put(node.id(), GraphPhase.BACKWARD));
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(new GraphValue(h, VALUE), new GraphValue(x, HISTORY), new GraphValue(integer, INTEGER),
                        new GraphValue(neg, VALUE), new GraphValue(duplicate, VALUE), new GraphValue(integralNeg, INTEGER),
                        new GraphValue(result, capturedPorts.get(0)),
                        new GraphValue(emission, capturedPorts.get(1)),
                        new GraphValue(savedH, capturedPorts.get(2)),
                        new GraphValue(savedSum, capturedPorts.get(3))),
                nodes, List.of(h, x, integer), List.of(result, emission), phases);
        return new Fixture(graph, new CompileTimeConstantGraph(
                graph, Map.of(integer, new CompileTimeConstantGraph.Splat(ScalarValue.int32(4)))), plan);
    }

    private static CompiledNode unary(long id, ValueId input, ValueId output) {
        return new CompiledNode(new NodeId(id), new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                List.of(input), List.of(output));
    }

    private static TensorDescriptor descriptor(DataType type, long... dimensions) {
        return new TensorDescriptor(type, Shape.of(dimensions), Optional.empty(), false);
    }

    private static TensorDescriptor canonical(DataType type, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private record Fixture(CompiledGraphModel graph, CompileTimeConstantGraph constants, ScanActivityPlan plan) {}
}
