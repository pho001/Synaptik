package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
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
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
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
    void closesActualTransposeMatmulGradientsToTheirOriginalLeafShapes() {
        for (DataType type : List.of(
                DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16)) {
            Tensor left = tensor(type, true, Shape.of(6));
            Tensor right = tensor(type, true, Shape.of(15));
            Tensor product = left.reshape(3, 2).permute(1, 0)
                    .matmul(right.reshape(5, 3).permute(1, 0));
            Tensor seed = tensor(type, false, Shape.of(2, 5));
            var stage = new FunctionalGradientRequest.Stage(
                    List.of(new FunctionalGradientRequest.ForwardTensorReference(product)),
                    List.of(Optional.of(seed)), List.of(left, right), false,
                    FunctionalGradientRequest.DisconnectedPolicy.ERROR);
            var compilation = GraphCompiler.compile(
                    CompileMode.FORWARD_AND_BACKWARD, List.of(product),
                    Optional.of(new FunctionalGradientRequest(List.of(stage))),
                    CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled());
            var descriptors = new LinkedHashMap<ValueId, TensorDescriptor>();
            compilation.validatedGraph().graph().values()
                    .forEach(value -> descriptors.put(value.id(), value.descriptor()));
            var outputs = compilation.validatedGraph().graph().outputs();
            assertEquals(LayoutDescriptor.contiguous(left.descriptor().shape()),
                    descriptors.get(outputs.get(1)).layout().orElseThrow(), type.name());
            assertEquals(LayoutDescriptor.contiguous(right.descriptor().shape()),
                    descriptors.get(outputs.get(2)).layout().orElseThrow(), type.name());
        }
    }

    @Test
    void closesCapturedMaterializedAffineComputedCropChainWithoutChangingGraphAssociations() {
        Shape matrix = Shape.of(3, 4);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT64, matrix,
                Optional.of(LayoutDescriptor.of(matrix, new long[] {7, 1}, 5, true)), true));
        Tensor materialized = input.neg();
        Tensor affine = materialized.permute(1, 0);
        Tensor computed = affine.neg();
        Tensor crop = computed.cropToShape(Shape.of(2, 2), Shape.of(1, 1));
        Tensor finalView = crop.permute(1, 0);
        ValidatedGraph source = CapturedGraphInference.inferAndValidate(GraphCapture.capture(
                List.of(materialized, affine, computed, crop, finalView)));

        ValidatedGraph result = StaticResultLogicalLayoutClosure.close(source);
        Map<ValueId, TensorDescriptor> descriptors = new LinkedHashMap<>();
        result.graph().values().forEach(value -> descriptors.put(value.id(), value.descriptor()));
        List<LayoutDescriptor> expected = List.of(
                LayoutDescriptor.contiguous(matrix),
                LayoutDescriptor.of(Shape.of(4, 3), new long[] {1, 4}, 0, true),
                LayoutDescriptor.contiguous(Shape.of(4, 3)),
                LayoutDescriptor.of(Shape.of(2, 2), new long[] {3, 1}, 4, true),
                LayoutDescriptor.of(Shape.of(2, 2), new long[] {1, 3}, 4, true));
        for (int index = 0; index < expected.size(); index++) {
            TensorDescriptor descriptor = descriptors.get(result.graph().outputs().get(index));
            assertEquals(expected.get(index), descriptor.layout().orElseThrow());
            assertEquals(DataType.FLOAT64, descriptor.dataType());
            assertTrue(descriptor.requiresGrad());
        }
        for (int index = 0; index < source.graph().nodes().size(); index++) {
            assertSame(source.graph().nodes().get(index), result.graph().nodes().get(index));
        }
        assertSame(source.graph().values().getFirst(), result.graph().values().getFirst());
        assertEquals(source.graph().inputs(), result.graph().inputs());
        assertEquals(source.graph().outputs(), result.graph().outputs());
        assertEquals(source.graph().nodePhases(), result.graph().nodePhases());
        assertSame(result, StaticResultLogicalLayoutClosure.close(result));
    }

    @Test
    void distinguishesCapturedReshapeViewFromNoncontiguousMaterialization() {
        Tensor input = tensor(DataType.FLOAT32, true, Shape.of(2, 3));
        Tensor computed = input.neg();
        Tensor view = computed.reshape(6);
        Tensor materialized = computed.permute(1, 0).reshape(6);
        ValidatedGraph source = CapturedGraphInference.inferAndValidate(
                GraphCapture.capture(List.of(view, materialized)));

        ValidatedGraph result = StaticResultLogicalLayoutClosure.close(source);
        Map<ValueId, TensorDescriptor> descriptors = new LinkedHashMap<>();
        result.graph().values().forEach(value -> descriptors.put(value.id(), value.descriptor()));
        assertEquals(LayoutDescriptor.of(Shape.of(6), new long[] {1}, 0, true),
                descriptors.get(result.graph().outputs().getFirst()).layout().orElseThrow());
        assertEquals(LayoutDescriptor.contiguous(Shape.of(6)),
                descriptors.get(result.graph().outputs().getLast()).layout().orElseThrow());
    }

    @Test
    void materializesUnprovedReshapeWithoutReinterpretingItsSourceAndClosesDownstreamView() {
        Shape sourceShape = Shape.of(3, 2);
        Shape vector = Shape.of(6);
        TensorDescriptor input = resolved(
                sourceShape, LayoutDescriptor.of(sourceShape, new long[] {1, 3}, 5, true));
        Operation reshape = new Operation(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(vector));
        assertTrue(LayoutInference.infer(reshape, List.of(input))
                .outputs().getFirst().layout().isEmpty(), "raw source inference remains view-only");
        var source = validated(graph(
                List.of(input, unresolved(vector), unresolved(Shape.of(1, 6))),
                List.of(
                        node(101, ShapeTransformKind.RESHAPE, reshape.attrs(), List.of(0), List.of(1)),
                        node(102, AxisTransformKind.EXPAND_DIMS, new AxisTransformAttrs(0),
                                List.of(1), List.of(2))),
                List.of(0), List.of(1, 2)), Map.of(), Map.of(id(0), new TensorId(310)));

        var result = StaticResultLogicalLayoutClosure.close(source);
        assertEquals(LayoutDescriptor.contiguous(vector),
                result.graph().values().get(1).descriptor().layout().orElseThrow());
        assertEquals(LayoutDescriptor.of(Shape.of(1, 6), new long[] {6, 1}, 0, true),
                result.graph().values().get(2).descriptor().layout().orElseThrow());
    }

    @Test
    void selectsCanonicalScalarAndEmptyMaterializationsUsingOrdinaryStaticGeometry() {
        List<Shape> sources = List.of(Shape.of(1, 1), Shape.of(0, 3));
        List<Shape> targets = List.of(Shape.scalar(), Shape.of(0));
        List<long[]> strides = List.of(new long[] {7, 3}, new long[] {7, 2});
        for (int index = 0; index < sources.size(); index++) {
            Shape input = sources.get(index), output = targets.get(index);
            var source = validated(graph(
                    List.of(resolved(input, LayoutDescriptor.of(input, strides.get(index), 4, true)),
                            unresolved(output)),
                    List.of(node(103, ShapeTransformKind.RESHAPE, new TargetShapeAttrs(output),
                            List.of(0), List.of(1))),
                    List.of(0), List.of(1)), Map.of(), Map.of(id(0), new TensorId(311)));
            assertEquals(LayoutDescriptor.contiguous(output), StaticResultLogicalLayoutClosure
                    .close(source).graph().values().get(1).descriptor().layout().orElseThrow());
        }
    }

    @Test
    void doesNotInventMaterializationGeometryFromMissingOrDynamicSourceLayouts() {
        for (Shape input : List.of(Shape.of(3, 2),
                Shape.ofDimensions(new DynamicDimension("N"), new StaticDimension(2)))) {
            Shape output = Shape.of(6);
            var source = validated(graph(
                    List.of(unresolved(input), unresolved(output)),
                    List.of(node(104, ShapeTransformKind.RESHAPE, new TargetShapeAttrs(output),
                            List.of(0), List.of(1))),
                    List.of(0), List.of(1)), Map.of(), Map.of(id(0), new TensorId(312)));
            assertTrue(StaticResultLogicalLayoutClosure.close(source)
                    .graph().values().get(1).descriptor().layout().isEmpty());
        }
    }

    @Test
    void rejectsOverflowingCanonicalStrideEvenWhenTheMaterializedElementCountIsZero() {
        Shape input = Shape.of(0, 3);
        Shape output = Shape.of(0, Long.MAX_VALUE, Long.MAX_VALUE);
        var source = validated(graph(
                List.of(resolved(input, LayoutDescriptor.of(input, new long[] {7, 2}, 0, true)),
                        unresolved(output)),
                List.of(node(105, ShapeTransformKind.RESHAPE, new TargetShapeAttrs(output),
                        List.of(0), List.of(1))),
                List.of(0), List.of(1)), Map.of(), Map.of(id(0), new TensorId(313)));
        var failure = assertThrows(IllegalArgumentException.class,
                () -> StaticResultLogicalLayoutClosure.close(source));
        assertTrue(failure.getCause() instanceof ArithmeticException);
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

    private static Tensor tensor(DataType type, boolean requiresGrad, Shape shape) {
        return TensorFactory.create(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad));
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
