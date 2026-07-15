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
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class ForwardExactArithmeticRewritingTest {
    @Test
    void rejectsNullWithTheSpecifiedMessage() {
        assertEquals("graph", assertThrows(NullPointerException.class,
                () -> ForwardExactArithmeticRewriting.rewrite(null)).getMessage());
    }

    @Test
    void rewritesEveryDuplicateMinimumAndMaximumNumericDomain() {
        for (BinaryArithmeticKind kind
                : List.of(BinaryArithmeticKind.MIN, BinaryArithmeticKind.MAX)) {
            for (DataType dataType : numericTypes()) {
                TensorDescriptor descriptor = descriptor(dataType, Shape.of(2), false);
                Operation operation = operation(kind);
                CompiledGraphModel graph = validatedGraph(
                        List.of(descriptor, descriptor),
                        List.of(node(41, operation, List.of(8L, 8L), List.of(17L))),
                        List.of(8L),
                        List.of(8L),
                        List.of(GraphPhase.FORWARD));

                CompiledGraphModel result = ForwardExactArithmeticRewriting.rewrite(graph);

                assertAll(kind + " " + dataType,
                        () -> assertNotSame(graph, result),
                        () -> assertTrue(result.nodes().isEmpty()),
                        () -> assertEquals(List.of(new ValueId(0)), result.inputs()),
                        () -> assertEquals(List.of(new ValueId(0)), result.outputs()),
                        () -> assertSame(descriptor, result.values().getFirst().descriptor()),
                        () -> assertSame(result,
                                CapturedGraphInference.inferAndValidate(result).graph()));
            }
        }
    }

    @Test
    void rewritesEverySelectedTypedScalarRow() {
        for (ScalarRewriteCase selected : selectedScalarCases()) {
            TensorDescriptor descriptor =
                    descriptor(selected.value().dataType(), Shape.of(3), false);
            Operation operation = scalarOperation(selected.kind(), selected.value());
            CompiledGraphModel graph = validatedGraph(
                    List.of(descriptor, descriptor),
                    List.of(node(9, operation, List.of(4L), List.of(12L))),
                    List.of(4L),
                    List.of(4L),
                    List.of(GraphPhase.FORWARD));

            CompiledGraphModel result = ForwardExactArithmeticRewriting.rewrite(graph);

            assertAll(selected.toString(),
                    () -> assertNotSame(graph, result),
                    () -> assertTrue(result.nodes().isEmpty()),
                    () -> assertEquals(List.of(new ValueId(0)), result.inputs()),
                    () -> assertEquals(List.of(new ValueId(0)), result.outputs()),
                    () -> assertSame(descriptor, result.values().getFirst().descriptor()));
        }
    }

    @Test
    void usesEarlierRemappingToExposeALaterDuplicateExtremaInTheSameScan() {
        TensorDescriptor descriptor = descriptor(DataType.FLOAT32, Shape.of(2), false);
        Operation multiply = scalarOperation(
                ScalarElementwiseKind.MUL, ScalarValue.float32(1.0f));
        Operation minimum = operation(BinaryArithmeticKind.MIN);
        CompiledGraphModel graph = validatedGraph(
                List.of(descriptor, descriptor, descriptor),
                List.of(
                        node(20, multiply, List.of(5L), List.of(11L)),
                        node(30, minimum, List.of(11L, 5L), List.of(14L))),
                List.of(5L),
                List.of(5L),
                List.of(GraphPhase.FORWARD, GraphPhase.FORWARD));

        CompiledGraphModel result = ForwardExactArithmeticRewriting.rewrite(graph);

        assertAll(
                () -> assertTrue(result.nodes().isEmpty()),
                () -> assertEquals(List.of(new ValueId(0)), result.inputs()),
                () -> assertEquals(List.of(new ValueId(0)), result.outputs()),
                () -> assertEquals(2, graph.nodes().size()));
    }

    @Test
    void requiresExactDescriptorEqualityAndNoGradientEligibility() {
        Shape shape = Shape.of(2);
        TensorDescriptor resolvedInput = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        TensorDescriptor unresolvedOutput = descriptor(DataType.FLOAT32, shape, false);
        Operation multiply = scalarOperation(
                ScalarElementwiseKind.MUL, ScalarValue.float32(1.0f));
        CompiledGraphModel layoutMismatch = validatedGraph(
                List.of(resolvedInput, unresolvedOutput),
                List.of(node(0, multiply, List.of(0L), List.of(1L))),
                List.of(0L),
                List.of(0L),
                List.of(GraphPhase.FORWARD));

        TensorDescriptor gradient = descriptor(DataType.FLOAT32, shape, true);
        CompiledGraphModel gradientEligible = validatedGraph(
                List.of(gradient, gradient),
                List.of(node(0, multiply, List.of(0L), List.of(1L))),
                List.of(0L),
                List.of(0L),
                List.of(GraphPhase.FORWARD));
        CompiledGraphModel gradientEligibleExtrema = validatedGraph(
                List.of(gradient, gradient),
                List.of(node(0, operation(BinaryArithmeticKind.MIN),
                        List.of(0L, 0L), List.of(1L))),
                List.of(0L),
                List.of(0L),
                List.of(GraphPhase.FORWARD));

        assertAll(
                () -> assertSame(layoutMismatch,
                        ForwardExactArithmeticRewriting.rewrite(layoutMismatch)),
                () -> assertSame(gradientEligible,
                        ForwardExactArithmeticRewriting.rewrite(gradientEligible)),
                () -> assertSame(gradientEligibleExtrema,
                        ForwardExactArithmeticRewriting.rewrite(gradientEligibleExtrema)));
    }

    @Test
    void excludesGraphOutputAndBackwardOccurrences() {
        TensorDescriptor descriptor = descriptor(DataType.FLOAT64, Shape.of(2), false);
        Operation multiply = scalarOperation(
                ScalarElementwiseKind.MUL, ScalarValue.float64(1.0d));
        CompiledGraphModel graphOutput = validatedGraph(
                List.of(descriptor, descriptor),
                List.of(node(0, multiply, List.of(0L), List.of(1L))),
                List.of(0L),
                List.of(1L),
                List.of(GraphPhase.FORWARD));
        CompiledGraphModel backward = validatedGraph(
                List.of(descriptor, descriptor),
                List.of(node(0, multiply, List.of(0L), List.of(1L))),
                List.of(0L),
                List.of(0L),
                List.of(GraphPhase.BACKWARD));

        assertAll(
                () -> assertSame(graphOutput,
                        ForwardExactArithmeticRewriting.rewrite(graphOutput)),
                () -> assertSame(backward,
                        ForwardExactArithmeticRewriting.rewrite(backward)));
    }

    @Test
    void retainsTheCompleteAdjacentScalarValueAndRuleMatrix() {
        for (ScalarRewriteCase retained : retainedScalarCases()) {
            TensorDescriptor descriptor =
                    descriptor(retained.value().dataType(), Shape.of(2), false);
            CompiledGraphModel graph = validatedGraph(
                    List.of(descriptor, descriptor),
                    List.of(node(0, scalarOperation(retained.kind(), retained.value()),
                            List.of(0L), List.of(1L))),
                    List.of(0L),
                    List.of(0L),
                    List.of(GraphPhase.FORWARD));

            assertSame(graph, ForwardExactArithmeticRewriting.rewrite(graph),
                    retained.toString());
        }
    }

    @Test
    void retainsCancellationTensorShapedAndDifferentInputExtremaRules() {
        TensorDescriptor floating = descriptor(DataType.FLOAT32, Shape.of(2), false);
        for (BinaryArithmeticKind kind : List.of(
                BinaryArithmeticKind.ADD,
                BinaryArithmeticKind.SUB,
                BinaryArithmeticKind.MUL,
                BinaryArithmeticKind.DIV,
                BinaryArithmeticKind.POW,
                BinaryArithmeticKind.MIN,
                BinaryArithmeticKind.MAX)) {
            List<Long> inputs = kind == BinaryArithmeticKind.SUB
                            || kind == BinaryArithmeticKind.DIV
                    ? List.of(0L, 0L)
                    : List.of(0L, 1L);
            int valueCount = inputs.equals(List.of(0L, 0L)) ? 2 : 3;
            List<TensorDescriptor> descriptors = new ArrayList<>();
            for (int index = 0; index < valueCount; index++) {
                descriptors.add(floating);
            }
            long output = valueCount - 1L;
            CompiledGraphModel graph = validatedGraph(
                    descriptors,
                    List.of(node(0, operation(kind), inputs, List.of(output))),
                    inputs.equals(List.of(0L, 0L)) ? List.of(0L) : List.of(0L, 1L),
                    List.of(0L),
                    List.of(GraphPhase.FORWARD));

            assertSame(graph, ForwardExactArithmeticRewriting.rewrite(graph), kind.toString());
        }
    }

    @Test
    void retainsScalarBoundsClampUnaryReductionMixedTypeAndMultiOutputFamilies() {
        TensorDescriptor floating = descriptor(DataType.FLOAT32, Shape.of(4), false);
        List<CompiledGraphModel> retained = List.of(
                singleInputInternalGraph(
                        scalarOperation(ScalarElementwiseKind.MIN,
                                ScalarValue.float32(Float.POSITIVE_INFINITY)),
                        floating, floating),
                singleInputInternalGraph(
                        scalarOperation(ScalarElementwiseKind.MAX,
                                ScalarValue.float32(Float.NEGATIVE_INFINITY)),
                        floating, floating),
                singleInputInternalGraph(
                        new Operation(ScalarElementwiseKind.CLAMP, new ClampRangeAttrs(
                                ScalarValue.float32(-1.0f), ScalarValue.float32(1.0f))),
                        floating, floating),
                singleInputInternalGraph(
                        operation(UnaryElementwiseKind.RECIPROCAL), floating, floating));
        for (CompiledGraphModel graph : retained) {
            assertSame(graph, ForwardExactArithmeticRewriting.rewrite(graph));
        }

        TensorDescriptor scalar = descriptor(DataType.FLOAT32, Shape.scalar(), false);
        CompiledGraphModel reduction = validatedGraph(
                List.of(floating, scalar),
                List.of(node(0, operation(AggregateReductionKind.MIN),
                        List.of(0L), List.of(1L))),
                List.of(0L),
                List.of(0L),
                List.of(GraphPhase.FORWARD));
        assertSame(reduction, ForwardExactArithmeticRewriting.rewrite(reduction));

        TensorDescriptor float64 = descriptor(DataType.FLOAT64, Shape.of(4), false);
        CompiledGraphModel mixed = validatedGraph(
                List.of(floating, float64, float64),
                List.of(node(0, operation(BinaryArithmeticKind.MIN),
                        List.of(0L, 1L), List.of(2L))),
                List.of(0L, 1L),
                List.of(0L),
                List.of(GraphPhase.FORWARD));
        assertSame(mixed, ForwardExactArithmeticRewriting.rewrite(mixed));

        TensorDescriptor selected = descriptor(DataType.FLOAT32, Shape.of(2), false);
        TensorDescriptor indices = descriptor(DataType.INT64, Shape.of(2), false);
        CompiledGraphModel multiOutput = validatedGraph(
                List.of(floating, selected, indices),
                List.of(node(0, new Operation(
                                TopKKind.TOP_K, new TopKAttrs(0, 2, true, true)),
                        List.of(0L), List.of(1L, 2L))),
                List.of(0L),
                List.of(0L),
                List.of(GraphPhase.FORWARD));
        assertSame(multiOutput, ForwardExactArithmeticRewriting.rewrite(multiOutput));
    }

    @Test
    void rebuildsDeterministicallyWithoutMutationAndRetainsExactSemanticReferences() {
        TensorDescriptor firstDescriptor = descriptor(DataType.INT64, Shape.of(2), false);
        TensorDescriptor secondDescriptor = descriptor(DataType.INT64, Shape.of(3), false);
        Operation rewrite = scalarOperation(
                ScalarElementwiseKind.ADD, ScalarValue.int64(0L));
        Operation retainedForward = scalarOperation(
                ScalarElementwiseKind.MUL, ScalarValue.int64(2L));
        Operation retainedBackward = scalarOperation(
                ScalarElementwiseKind.SUB, ScalarValue.int64(3L));
        CompiledGraphModel graph = validatedGraph(
                List.of(
                        firstDescriptor,
                        secondDescriptor,
                        firstDescriptor,
                        firstDescriptor,
                        secondDescriptor),
                List.of(
                        node(70, rewrite, List.of(10L), List.of(31L)),
                        node(80, retainedForward, List.of(31L), List.of(44L)),
                        node(90, retainedBackward, List.of(20L), List.of(55L))),
                List.of(10L, 20L),
                List.of(44L, 55L),
                List.of(GraphPhase.FORWARD, GraphPhase.FORWARD, GraphPhase.BACKWARD));

        CompiledGraphModel first = ForwardExactArithmeticRewriting.rewrite(graph);
        CompiledGraphModel second = ForwardExactArithmeticRewriting.rewrite(graph);

        assertAll(
                () -> assertEquals(first, second),
                () -> assertNotSame(first, second),
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1)), first.inputs()),
                () -> assertEquals(List.of(new ValueId(2), new ValueId(3)), first.outputs()),
                () -> assertEquals(2, first.nodes().size()),
                () -> assertEquals(List.of(new ValueId(0)), first.nodes().get(0).inputs()),
                () -> assertEquals(List.of(new ValueId(1)), first.nodes().get(1).inputs()),
                () -> assertSame(retainedForward, first.nodes().get(0).operation()),
                () -> assertSame(retainedBackward, first.nodes().get(1).operation()),
                () -> assertSame(firstDescriptor, first.values().get(2).descriptor()),
                () -> assertSame(secondDescriptor, first.values().get(3).descriptor()),
                () -> assertSame(GraphPhase.FORWARD,
                        first.nodePhases().get(new NodeId(0))),
                () -> assertSame(GraphPhase.BACKWARD,
                        first.nodePhases().get(new NodeId(1))),
                () -> assertEquals(3, graph.nodes().size()),
                () -> assertEquals(List.of(new ValueId(44), new ValueId(55)), graph.outputs()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> first.nodes().clear()),
                () -> assertSame(first,
                        CapturedGraphInference.inferAndValidate(first).graph()));
    }

    private static List<DataType> numericTypes() {
        return List.of(
                DataType.BFLOAT16,
                DataType.FLOAT32,
                DataType.FLOAT64,
                DataType.INT32,
                DataType.INT64);
    }

    private static List<ScalarRewriteCase> selectedScalarCases() {
        List<ScalarRewriteCase> result = new ArrayList<>();
        for (ScalarValue value : List.of(
                ScalarValue.bfloat16Bits((short) 0x3F80),
                ScalarValue.float32(1.0f),
                ScalarValue.float64(1.0d),
                ScalarValue.int32(1),
                ScalarValue.int64(1L))) {
            result.add(new ScalarRewriteCase(ScalarElementwiseKind.MUL, value));
        }
        for (ScalarElementwiseKind kind
                : List.of(ScalarElementwiseKind.DIV, ScalarElementwiseKind.POW)) {
            for (ScalarValue value : List.of(
                    ScalarValue.bfloat16Bits((short) 0x3F80),
                    ScalarValue.float32(1.0f),
                    ScalarValue.float64(1.0d))) {
                result.add(new ScalarRewriteCase(kind, value));
            }
        }
        for (ScalarElementwiseKind kind
                : List.of(ScalarElementwiseKind.ADD, ScalarElementwiseKind.SUB)) {
            result.add(new ScalarRewriteCase(kind, ScalarValue.int32(0)));
            result.add(new ScalarRewriteCase(kind, ScalarValue.int64(0L)));
        }
        return List.copyOf(result);
    }

    private static List<ScalarRewriteCase> retainedScalarCases() {
        List<ScalarRewriteCase> result = new ArrayList<>();
        for (ScalarElementwiseKind kind
                : List.of(ScalarElementwiseKind.ADD, ScalarElementwiseKind.SUB)) {
            result.add(new ScalarRewriteCase(kind, ScalarValue.float64(0.0d)));
            result.add(new ScalarRewriteCase(kind, ScalarValue.float64(-0.0d)));
            result.add(new ScalarRewriteCase(kind, ScalarValue.float32(0.0f)));
            result.add(new ScalarRewriteCase(kind, ScalarValue.float32(-0.0f)));
        }
        for (ScalarValue value : List.of(
                ScalarValue.bfloat16Bits((short) 0xBF80),
                ScalarValue.bfloat16Bits((short) 0x0000),
                ScalarValue.bfloat16Bits((short) 0x3F7F),
                ScalarValue.bfloat16Bits((short) 0x3F81),
                ScalarValue.bfloat16Bits((short) 0x4000),
                ScalarValue.bfloat16Bits((short) 0x7FC0),
                ScalarValue.bfloat16Bits((short) 0x7F80),
                ScalarValue.bfloat16Bits((short) 0xFF80),
                ScalarValue.float32(-1.0f),
                ScalarValue.float32(0.0f),
                ScalarValue.float32(2.0f),
                ScalarValue.float32(Float.NaN),
                ScalarValue.float32(Float.POSITIVE_INFINITY),
                ScalarValue.float32(Float.NEGATIVE_INFINITY),
                ScalarValue.int32(-1),
                ScalarValue.int32(0),
                ScalarValue.int32(2),
                ScalarValue.int64(-1L),
                ScalarValue.int64(0L),
                ScalarValue.int64(2L))) {
            result.add(new ScalarRewriteCase(ScalarElementwiseKind.MUL, value));
        }
        for (ScalarElementwiseKind kind
                : List.of(ScalarElementwiseKind.DIV, ScalarElementwiseKind.POW)) {
            for (ScalarValue value : List.of(
                    ScalarValue.bfloat16Bits((short) 0xBF80),
                    ScalarValue.bfloat16Bits((short) 0x0000),
                    ScalarValue.bfloat16Bits((short) 0x3F7F),
                    ScalarValue.bfloat16Bits((short) 0x3F81),
                    ScalarValue.bfloat16Bits((short) 0x4000),
                    ScalarValue.bfloat16Bits((short) 0x7FC0),
                    ScalarValue.bfloat16Bits((short) 0x7F80),
                    ScalarValue.bfloat16Bits((short) 0xFF80),
                    ScalarValue.float64(-1.0d),
                    ScalarValue.float64(0.0d),
                    ScalarValue.float64(2.0d),
                    ScalarValue.float64(3.0d),
                    ScalarValue.float64(Double.NaN),
                    ScalarValue.float64(Double.POSITIVE_INFINITY),
                    ScalarValue.float64(Double.NEGATIVE_INFINITY))) {
                result.add(new ScalarRewriteCase(kind, value));
            }
        }
        result.add(new ScalarRewriteCase(ScalarElementwiseKind.ADD, ScalarValue.int32(1)));
        result.add(new ScalarRewriteCase(ScalarElementwiseKind.SUB, ScalarValue.int64(-1L)));
        return List.copyOf(result);
    }

    private static CompiledGraphModel singleInputInternalGraph(
            Operation operation, TensorDescriptor input, TensorDescriptor output) {
        return validatedGraph(
                List.of(input, output),
                List.of(node(0, operation, List.of(0L), List.of(1L))),
                List.of(0L),
                List.of(0L),
                List.of(GraphPhase.FORWARD));
    }

    private static CompiledGraphModel validatedGraph(
            List<TensorDescriptor> descriptors,
            List<CompiledNode> nodes,
            List<Long> inputs,
            List<Long> outputs,
            List<GraphPhase> phases) {
        List<GraphValue> values = new ArrayList<>(descriptors.size());
        List<ValueId> valueIds = collectValueIds(descriptors.size(), nodes, inputs);
        for (int index = 0; index < descriptors.size(); index++) {
            values.add(new GraphValue(valueIds.get(index), descriptors.get(index)));
        }
        Map<NodeId, GraphPhase> phaseMap = new java.util.LinkedHashMap<>();
        for (int index = 0; index < nodes.size(); index++) {
            phaseMap.put(nodes.get(index).id(), phases.get(index));
        }
        CompiledGraphModel graph = new CompiledGraphModel(
                values,
                nodes,
                inputs.stream().map(ValueId::new).toList(),
                outputs.stream().map(ValueId::new).toList(),
                phaseMap);
        return CapturedGraphInference.inferAndValidate(graph).graph();
    }

    private static List<ValueId> collectValueIds(
            int expectedSize, List<CompiledNode> nodes, List<Long> inputs) {
        List<ValueId> result = new ArrayList<>(expectedSize);
        for (Long input : inputs) {
            if (!result.contains(new ValueId(input))) {
                result.add(new ValueId(input));
            }
        }
        for (CompiledNode node : nodes) {
            for (ValueId input : node.inputs()) {
                if (!result.contains(input)) {
                    result.add(input);
                }
            }
            for (ValueId output : node.outputs()) {
                if (!result.contains(output)) {
                    result.add(output);
                }
            }
        }
        assertEquals(expectedSize, result.size());
        return result;
    }

    private static CompiledNode node(
            long id, Operation operation, List<Long> inputs, List<Long> outputs) {
        return new CompiledNode(
                new NodeId(id),
                operation,
                inputs.stream().map(ValueId::new).toList(),
                outputs.stream().map(ValueId::new).toList());
    }

    private static Operation operation(io.github.pho001.synaptik.model.operation.OperationKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }

    private static Operation scalarOperation(
            ScalarElementwiseKind kind, ScalarValue value) {
        return new Operation(kind, new ScalarValueAttrs(value));
    }

    private static TensorDescriptor descriptor(
            DataType dataType, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad);
    }

    private record ScalarRewriteCase(ScalarElementwiseKind kind, ScalarValue value) {}
}
