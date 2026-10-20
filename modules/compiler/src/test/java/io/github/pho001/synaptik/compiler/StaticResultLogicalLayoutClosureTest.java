package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertAll;
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
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class StaticResultLogicalLayoutClosureTest {
    @Test
    void closesAllowlistedComputationAndExplicitContiguousResultsOnly() {
        Shape shape = Shape.of(2, 3);
        CompiledNode add = node(10, BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE,
                List.of(0, 1), List.of(2));
        CompiledNode contiguous = node(11, ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE,
                List.of(2), List.of(3));
        CompiledGraphModel graph = graph(
                List.of(unresolved(shape), unresolved(shape), unresolved(shape), unresolved(shape)),
                List.of(add, contiguous), List.of(0, 1), List.of(3));
        ValidatedGraph source = validated(
                graph, Map.of(), Map.of(id(0), new TensorId(100), id(1), new TensorId(101)));

        ValidatedGraph result = StaticResultLogicalLayoutClosure.close(source);

        assertAll(
                () -> assertNotSame(source, result),
                () -> assertTrue(result.graph().values().get(0).descriptor().layout().isEmpty()),
                () -> assertTrue(result.graph().values().get(1).descriptor().layout().isEmpty()),
                () -> assertEquals(LayoutDescriptor.contiguous(shape),
                        result.graph().values().get(2).descriptor().layout().orElseThrow()),
                () -> assertEquals(LayoutDescriptor.contiguous(shape),
                        result.graph().values().get(3).descriptor().layout().orElseThrow()),
                () -> assertSame(add, result.graph().nodes().get(0)),
                () -> assertSame(contiguous, result.graph().nodes().get(1)));
    }

    @Test
    void derivesExactAffineViewsIncludingCoordinateSliceAndScalarSelect() {
        Shape matrix = Shape.of(2, 3);
        LayoutDescriptor offset = LayoutDescriptor.of(matrix, new long[] {3, 1}, 5, true);
        LayoutDescriptor zeroStride = LayoutDescriptor.of(
                Shape.of(1, 3), new long[] {0, 1}, 0, true);
        List<CompiledNode> nodes = List.of(
                node(20, AxisTransformKind.PERMUTE, new PermutationAttrs(List.of(1, 0)),
                        List.of(0), List.of(2)),
                node(21, ShapeTransformKind.RESHAPE, new TargetShapeAttrs(Shape.of(6)),
                        List.of(0), List.of(3)),
                node(22, ShapeTransformKind.EXPAND, new TargetShapeAttrs(matrix),
                        List.of(1), List.of(4)),
                node(23, SliceKind.SLICE,
                        new SliceAttrs(List.of(0L), List.of(2L), List.of(1), List.of(1L)),
                        List.of(0), List.of(5)),
                node(24, SelectKind.SELECT, new SelectAttrs(0, 1),
                        List.of(0), List.of(6)));
        CompiledGraphModel graph = graph(
                List.of(
                        resolved(matrix, offset),
                        resolved(Shape.of(1, 3), zeroStride),
                        unresolved(Shape.of(3, 2)),
                        unresolved(Shape.of(6)),
                        unresolved(matrix),
                        unresolved(Shape.of(2, 2)),
                        unresolved(Shape.of(3))),
                nodes, List.of(0, 1), List.of(2, 3, 4, 5, 6));
        ValidatedGraph source = validated(
                graph, Map.of(), Map.of(id(0), new TensorId(200), id(1), new TensorId(201)));

        ValidatedGraph result = StaticResultLogicalLayoutClosure.close(source);

        assertNotSame(source, result);
        assertEquals(offset, result.graph().values().get(0).descriptor().layout().orElseThrow());
        assertEquals(zeroStride,
                result.graph().values().get(1).descriptor().layout().orElseThrow());
        assertEquals(
                LayoutDescriptor.of(Shape.of(3, 2), new long[] {1, 3}, 5, true),
                result.graph().values().get(2).descriptor().layout().orElseThrow());
        assertEquals(
                LayoutDescriptor.of(Shape.of(6), new long[] {1}, 5, true),
                result.graph().values().get(3).descriptor().layout().orElseThrow());
        assertEquals(
                LayoutDescriptor.of(matrix, new long[] {0, 1}, 0, true),
                result.graph().values().get(4).descriptor().layout().orElseThrow());
        assertEquals(LayoutDescriptor.of(Shape.of(2, 2), new long[] {3, 1}, 5, true),
                result.graph().values().get(5).descriptor().layout().orElseThrow());
        assertEquals(LayoutDescriptor.of(Shape.of(3), new long[] {1}, 8, true),
                result.graph().values().get(6).descriptor().layout().orElseThrow());
    }

    @Test
    void propagatesMaterializedMatmulThroughRnnSelectAndDownstreamPermute() {
        Shape batch = Shape.of(3, 2, 2);
        Shape matrix = Shape.of(2, 2);
        List<CompiledNode> nodes = List.of(
                node(40, MatmulKind.MATMUL, NoOperationAttrs.INSTANCE, List.of(0, 1), List.of(2)),
                node(41, SelectKind.SELECT, new SelectAttrs(0, 2), List.of(2), List.of(3)),
                node(42, AxisTransformKind.PERMUTE, new PermutationAttrs(List.of(1, 0)),
                        List.of(3), List.of(4)));
        CompiledGraphModel graph = graph(
                List.of(unresolved(batch), unresolved(matrix), unresolved(batch),
                        unresolved(matrix), unresolved(matrix)),
                nodes, List.of(0, 1), List.of(4));
        ValidatedGraph source = validated(graph, Map.of(),
                Map.of(id(0), new TensorId(400), id(1), new TensorId(401)));

        ValidatedGraph result = StaticResultLogicalLayoutClosure.close(source);

        assertEquals(LayoutDescriptor.contiguous(batch),
                result.graph().values().get(2).descriptor().layout().orElseThrow());
        assertEquals(LayoutDescriptor.of(matrix, new long[] {2, 1}, 8, true),
                result.graph().values().get(3).descriptor().layout().orElseThrow());
        assertEquals(LayoutDescriptor.of(matrix, new long[] {1, 2}, 8, true),
                result.graph().values().get(4).descriptor().layout().orElseThrow());
        assertSame(result, StaticResultLogicalLayoutClosure.close(result));
    }

    @Test
    void propagatesMatmulAndConcatGateSlicesWithoutCanonicalizingTheirRowStrides() {
        for (int width : List.of(6, 8)) {
            for (boolean concat : List.of(false, true)) {
                Shape matrix = Shape.of(2, 2);
                Shape projection = Shape.of(2, width);
                Shape secondInput = Shape.of(2, width - 2);
                List<CompiledNode> nodes = List.of(
                        node(50, concat ? TensorCompositionKind.CONCAT : MatmulKind.MATMUL,
                                concat ? new CompositionAxisAttrs(1) : NoOperationAttrs.INSTANCE,
                                List.of(0, 1), List.of(2)),
                        node(51, SliceKind.SLICE,
                                new SliceAttrs(List.of(2L), List.of(2L), List.of(1), List.of(1L)),
                                List.of(2), List.of(3)),
                        node(52, AxisTransformKind.PERMUTE, new PermutationAttrs(List.of(1, 0)),
                                List.of(3), List.of(4)));
                CompiledGraphModel graph = graph(
                        List.of(unresolved(matrix), unresolved(concat ? secondInput : projection),
                                unresolved(projection), unresolved(matrix), unresolved(matrix)),
                        nodes, List.of(0, 1), List.of(4));
                ValidatedGraph result = StaticResultLogicalLayoutClosure.close(validated(
                        graph, Map.of(), Map.of(id(0), new TensorId(500), id(1), new TensorId(501))));

                assertEquals(LayoutDescriptor.of(matrix, new long[] {width, 1}, 2, true),
                        result.graph().values().get(3).descriptor().layout().orElseThrow());
                assertEquals(LayoutDescriptor.of(matrix, new long[] {1, width}, 2, true),
                        result.graph().values().get(4).descriptor().layout().orElseThrow());
            }
        }
    }

    @Test
    void retainsNonunitAndZeroStridesAndOffsetsThroughSliceThenSelect() {
        for (long rowStride : List.of(0L, 11L)) {
            Shape matrix = Shape.of(3, 5);
            Shape sliced = Shape.of(3, 2);
            CompiledGraphModel graph = graph(
                    List.of(resolved(matrix,
                                    LayoutDescriptor.of(matrix, new long[] {rowStride, 2}, 7, true)),
                            unresolved(sliced), unresolved(Shape.of(2))),
                    List.of(
                            node(60, SliceKind.SLICE,
                                    new SliceAttrs(List.of(1L), List.of(2L), List.of(1), List.of(2L)),
                                    List.of(0), List.of(1)),
                            node(61, SelectKind.SELECT, new SelectAttrs(0, 2), List.of(1), List.of(2))),
                    List.of(0), List.of(2));
            ValidatedGraph result = StaticResultLogicalLayoutClosure.close(validated(
                    graph, Map.of(), Map.of(id(0), new TensorId(600))));

            assertEquals(LayoutDescriptor.of(sliced, new long[] {rowStride, 4}, 9, true),
                    result.graph().values().get(1).descriptor().layout().orElseThrow());
            assertEquals(LayoutDescriptor.of(Shape.of(2), new long[] {4}, 9 + 2 * rowStride, true),
                    result.graph().values().get(2).descriptor().layout().orElseThrow());
        }
    }

    @Test
    void leavesNegativeEmptyUnknownDynamicAndAlreadyResolvedExtractionsUntouched() {
        Shape matrix = Shape.of(2, 3);
        Shape dynamic = Shape.ofDimensions(new DynamicDimension("N"), new StaticDimension(3));
        LayoutDescriptor canonical = LayoutDescriptor.contiguous(matrix);
        CompiledGraphModel graph = graph(
                List.of(resolved(matrix, canonical), unresolved(matrix), unresolved(dynamic),
                        unresolved(matrix), unresolved(Shape.of(2, 0)), unresolved(Shape.of(3)),
                        unresolved(Shape.of(3)), resolved(Shape.of(3),
                                LayoutDescriptor.of(Shape.of(3), new long[] {2}, 4, true)),
                        resolved(Shape.of(0, 3), LayoutDescriptor.contiguous(Shape.of(0, 3))),
                        unresolved(Shape.of(0))),
                List.of(
                        node(70, SliceKind.SLICE,
                                new SliceAttrs(List.of(2L), List.of(3L), List.of(1), List.of(-1L)),
                                List.of(0), List.of(3)),
                        node(71, SliceKind.SLICE,
                                new SliceAttrs(List.of(0L), List.of(0L), List.of(1), List.of(1L)),
                                List.of(0), List.of(4)),
                        node(72, SelectKind.SELECT, new SelectAttrs(0, 1), List.of(1), List.of(5)),
                        node(73, SelectKind.SELECT, new SelectAttrs(0, 1), List.of(2), List.of(6)),
                        node(74, SelectKind.SELECT, new SelectAttrs(0, 1), List.of(0), List.of(7)),
                        node(75, SelectKind.SELECT, new SelectAttrs(1, 1), List.of(8), List.of(9))),
                List.of(0, 1, 2, 8), List.of(3, 4, 5, 6, 7, 9));
        ValidatedGraph source = validated(graph, Map.of(), Map.of(
                id(0), new TensorId(700), id(1), new TensorId(701),
                id(2), new TensorId(702), id(8), new TensorId(708)));

        assertSame(source, StaticResultLogicalLayoutClosure.close(source));
    }

    @Test
    void preservesCapturedMetadataAndRejectsInferredDtypeOrShapeMismatch() {
        Shape matrix = Shape.of(2, 3);
        Shape vector = Shape.of(3);
        CompiledNode select = node(80, SelectKind.SELECT, new SelectAttrs(0, 1),
                List.of(0), List.of(1));
        for (TensorDescriptor output : List.of(
                new TensorDescriptor(DataType.FLOAT64, vector, Optional.empty(), true),
                unresolved(Shape.of(2)))) {
            ValidatedGraph source = validated(graph(
                    List.of(resolved(matrix, LayoutDescriptor.contiguous(matrix)), output),
                    List.of(select), List.of(0), List.of(1)), Map.of(),
                    Map.of(id(0), new TensorId(800)));
            assertSame(source, StaticResultLogicalLayoutClosure.close(source));
        }
        CompiledGraphModel graph = graph(
                List.of(resolved(matrix, LayoutDescriptor.contiguous(matrix)),
                        new TensorDescriptor(DataType.FLOAT32, vector, Optional.empty(), true),
                        unresolved(Shape.scalar())),
                List.of(select), List.of(0, 2), List.of(1));
        graph = new CompiledGraphModel(graph.values(), graph.nodes(), graph.inputs(), graph.outputs(),
                Map.of(select.id(), GraphPhase.BACKWARD));
        var constraint = new DeferredGraphConstraint(select.id(), "retained obligation",
                new DimensionAtLeast(new DynamicDimension("M"), 1));
        ValidatedGraph source = new ValidatedGraph(new CompileTimeConstantGraph(graph,
                Map.of(id(2), new CompileTimeConstantGraph.Splat(ScalarValue.float32(1))),
                Map.of(id(0), new TensorId(800))), List.of(constraint),
                new DerivativeGraphMetadata(graph, Map.of(select.id(), 2)));

        ValidatedGraph result = StaticResultLogicalLayoutClosure.close(source);

        assertEquals(LayoutDescriptor.of(vector, new long[] {1}, 3, true),
                result.graph().values().get(1).descriptor().layout().orElseThrow());
        assertTrue(result.graph().values().get(1).descriptor().requiresGrad());
        assertSame(source.graph().values().getFirst(), result.graph().values().getFirst());
        assertSame(select, result.graph().nodes().getFirst());
        assertSame(select.operation(), result.graph().nodes().getFirst().operation());
        assertEquals(source.graph().values().stream().map(GraphValue::id).toList(),
                result.graph().values().stream().map(GraphValue::id).toList());
        assertEquals(source.graph().inputs(), result.graph().inputs());
        assertEquals(source.graph().outputs(), result.graph().outputs());
        assertEquals(source.graph().nodePhases(), result.graph().nodePhases());
        assertEquals(source.constants(), result.constants());
        assertEquals(source.constantGraph().bindableTensorIds(), result.constantGraph().bindableTensorIds());
        assertSame(constraint, result.constraints().getFirst());
        assertEquals(source.derivatives().derivativeOrderByNode(), result.derivatives().derivativeOrderByNode());
        assertSame(result.graph(), result.derivatives().graph());
    }

    @Test
    void reportsCheckedSliceStrideOverflowWithNodeAndOutputContext() {
        Shape singleton = Shape.of(1);
        CompiledNode slice = node(90, SliceKind.SLICE,
                new SliceAttrs(List.of(0L), List.of(1L), List.of(0), List.of(Long.MAX_VALUE)),
                List.of(0), List.of(1));
        ValidatedGraph source = validated(graph(
                List.of(resolved(singleton, LayoutDescriptor.of(singleton, new long[] {2}, 0, true)),
                        unresolved(singleton)), List.of(slice), List.of(0), List.of(1)),
                Map.of(), Map.of(id(0), new TensorId(900)));

        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> StaticResultLogicalLayoutClosure.close(source));

        assertTrue(failure.getMessage().contains(slice.id().toString()));
        assertTrue(failure.getMessage().contains(id(1).toString()));
        assertTrue(failure.getCause() instanceof ArithmeticException);
    }

    @Test
    void derivesRankEditAfterClosingItsMaterializedInputInTheSamePass() {
        Shape matrix = Shape.of(1, 3);
        Shape vector = Shape.of(3);
        CompiledNode materialized = node(
                25,
                UnaryElementwiseKind.NEG,
                NoOperationAttrs.INSTANCE,
                List.of(0),
                List.of(1));
        CompiledNode squeeze = node(
                26,
                AxisTransformKind.SQUEEZE,
                new AxisTransformAttrs(0),
                List.of(1),
                List.of(2));
        CompiledGraphModel graph = graph(
                List.of(resolved(matrix, LayoutDescriptor.contiguous(matrix)),
                        unresolved(matrix),
                        unresolved(vector)),
                List.of(materialized, squeeze),
                List.of(0),
                List.of(2));

        ValidatedGraph result = StaticResultLogicalLayoutClosure.close(validated(
                graph, Map.of(), Map.of(id(0), new TensorId(202))));

        assertEquals(
                LayoutDescriptor.contiguous(matrix),
                result.graph().values().get(1).descriptor().layout().orElseThrow());
        assertEquals(
                LayoutDescriptor.of(vector, new long[] {1}, 0, true),
                result.graph().values().get(2).descriptor().layout().orElseThrow());
    }

    @Test
    void retriesStaticCropAfterClosingMaterializedAndBroadcastInputs() {
    Shape row = Shape.of(1, 3);
    Shape matrix = Shape.of(2, 3);
    Shape cropShape = Shape.of(1, 2);
    List<CompiledNode> nodes =
        List.of(
            node(27, UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE, List.of(0), List.of(1)),
            node(
                28,
                SliceKind.SLICE,
                new CropToShapeAttrs(cropShape, Shape.of(1, 1)),
                List.of(1),
                List.of(2)),
            node(
                29,
                ShapeTransformKind.EXPAND,
                new TargetShapeAttrs(matrix),
                List.of(3),
                List.of(4)),
            node(
                30,
                SliceKind.SLICE,
                new CropToShapeAttrs(cropShape, Shape.of(1, 1)),
                List.of(4),
                List.of(5)));
    CompiledGraphModel graph =
        graph(
            List.of(
                resolved(matrix, LayoutDescriptor.contiguous(matrix)),
                unresolved(matrix),
                unresolved(cropShape),
                unresolved(row),
                unresolved(matrix),
                unresolved(cropShape)),
            nodes,
            List.of(0, 3),
            List.of(2, 5));
    ValidatedGraph result =
        StaticResultLogicalLayoutClosure.close(
            validated(
                graph,
                Map.of(id(3), new CompileTimeConstantGraph.Splat(ScalarValue.float32(2.0f))),
                Map.of(id(0), new TensorId(203))));

    assertAll(
        () ->
            assertEquals(
                LayoutDescriptor.of(cropShape, new long[] {3, 1}, 4, true),
                result.graph().values().get(2).descriptor().layout().orElseThrow()),
        () ->
            assertEquals(
                LayoutDescriptor.of(cropShape, new long[] {0, 1}, 1, true),
                result.graph().values().get(5).descriptor().layout().orElseThrow()));
  }

  @Test
  void closesStaticSplatsButPreservesBindableInputsAndDynamicResults() {
        Shape staticShape = Shape.of(2, 3);
        Shape dynamicShape = Shape.ofDimensions(
                new StaticDimension(2), new DynamicDimension("N"));
        CompiledNode staticNeg = node(30, UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE,
                List.of(0), List.of(3));
        CompiledNode dynamicNeg = node(31, UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE,
                List.of(2), List.of(4));
        CompiledGraphModel graph = graph(
                List.of(
                        unresolved(staticShape),
                        unresolved(staticShape),
                        unresolved(dynamicShape),
                        unresolved(staticShape),
                        unresolved(dynamicShape)),
                List.of(staticNeg, dynamicNeg), List.of(0, 1, 2), List.of(3, 4));
        ValidatedGraph source = validated(
                graph,
                Map.of(id(1), new CompileTimeConstantGraph.Splat(ScalarValue.float32(2.0f))),
                Map.of(id(0), new TensorId(300), id(2), new TensorId(301)));

        ValidatedGraph result = StaticResultLogicalLayoutClosure.close(source);

        assertAll(
                () -> assertNotSame(source, result),
                () -> assertTrue(result.graph().values().get(0).descriptor().layout().isEmpty()),
                () -> assertEquals(LayoutDescriptor.contiguous(staticShape),
                        result.graph().values().get(1).descriptor().layout().orElseThrow()),
                () -> assertTrue(result.graph().values().get(2).descriptor().layout().isEmpty()),
                () -> assertEquals(LayoutDescriptor.contiguous(staticShape),
                        result.graph().values().get(3).descriptor().layout().orElseThrow()),
                () -> assertTrue(result.graph().values().get(4).descriptor().layout().isEmpty()),
                () -> assertEquals(source.constantGraph().constants(),
                        result.constantGraph().constants()),
                () -> assertEquals(source.constantGraph().bindableTensorIds(),
                        result.constantGraph().bindableTensorIds()),
                () -> assertFalse(result.graph().values().get(1).descriptor().requiresGrad()));
    }

    private static ValidatedGraph validated(
            CompiledGraphModel graph,
            Map<ValueId, CompileTimeConstantGraph.Splat> constants,
            Map<ValueId, TensorId> bindable) {
        Map<NodeId, Integer> derivatives = new LinkedHashMap<>();
        graph.nodes().forEach(node -> derivatives.put(node.id(), 0));
        return new ValidatedGraph(
                new CompileTimeConstantGraph(graph, constants, bindable),
                List.of(),
                new DerivativeGraphMetadata(graph, derivatives));
    }

    private static CompiledGraphModel graph(
            List<TensorDescriptor> descriptors,
            List<CompiledNode> nodes,
            List<Integer> inputs,
            List<Integer> outputs) {
        List<GraphValue> values = new ArrayList<>();
        for (int index = 0; index < descriptors.size(); index++) {
            values.add(new GraphValue(id(index), descriptors.get(index)));
        }
        Map<NodeId, GraphPhase> phases = new LinkedHashMap<>();
        nodes.forEach(node -> phases.put(node.id(), GraphPhase.FORWARD));
        return new CompiledGraphModel(
                values,
                nodes,
                inputs.stream().map(StaticResultLogicalLayoutClosureTest::id).toList(),
                outputs.stream().map(StaticResultLogicalLayoutClosureTest::id).toList(),
                phases);
    }

    private static CompiledNode node(
            long nodeId,
            OperationKind kind,
            OperationAttrs attrs,
            List<Integer> inputs,
            List<Integer> outputs) {
        return new CompiledNode(
                new NodeId(nodeId),
                new Operation(kind, attrs),
                inputs.stream().map(StaticResultLogicalLayoutClosureTest::id).toList(),
                outputs.stream().map(StaticResultLogicalLayoutClosureTest::id).toList());
    }

    private static TensorDescriptor unresolved(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape, Optional.empty(), false);
    }

    private static TensorDescriptor resolved(Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(DataType.FLOAT32, shape, Optional.of(layout), false);
    }

    private static ValueId id(long value) {
        return new ValueId(value);
    }
}
