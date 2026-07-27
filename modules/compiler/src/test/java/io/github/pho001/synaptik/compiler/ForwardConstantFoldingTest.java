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
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class ForwardConstantFoldingTest {
    @Test
    void exposesOnlyThePackagePrivateStatelessFoldContract() throws Exception {
        var method = ForwardConstantFolding.class.getDeclaredMethod(
                "fold", CompileTimeConstantGraph.class);
        var constructor = ForwardConstantFolding.class.getDeclaredConstructor();
        assertAll(
                () -> assertTrue(Modifier.isFinal(ForwardConstantFolding.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(ForwardConstantFolding.class.getModifiers())),
                () -> assertEquals(0, ForwardConstantFolding.class.getDeclaredFields().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertSame(CompileTimeConstantGraph.class, method.getReturnType()),
                () -> assertEquals("constantGraph", assertThrows(NullPointerException.class,
                        () -> ForwardConstantFolding.fold(null)).getMessage()));
    }

    @Test
    void foldsCompleteBooleanTruthTables() {
        for (boolean value : List.of(false, true)) {
            assertEquals(!value, foldLogical(BooleanLogicalKind.NOT, value).booleanValue());
            for (boolean right : List.of(false, true)) {
                assertAll(
                        () -> assertEquals(value && right,
                                foldLogical(BooleanLogicalKind.AND, value, right).booleanValue()),
                        () -> assertEquals(value || right,
                                foldLogical(BooleanLogicalKind.OR, value, right).booleanValue()));
            }
        }
    }

    @Test
    void foldsIntegralArithmeticForEveryWidthPairWithSignedPromotionAndOverflow() {
        List<BinaryArithmeticKind> kinds = List.of(
                BinaryArithmeticKind.ADD,
                BinaryArithmeticKind.SUB,
                BinaryArithmeticKind.MUL,
                BinaryArithmeticKind.MIN,
                BinaryArithmeticKind.MAX);
        List<ScalarValue> lefts = List.of(
                ScalarValue.int32(Integer.MAX_VALUE), ScalarValue.int64(Long.MIN_VALUE));
        List<ScalarValue> rights = List.of(ScalarValue.int32(1), ScalarValue.int64(-1));

        for (BinaryArithmeticKind kind : kinds) {
            for (ScalarValue left : lefts) {
                for (ScalarValue right : rights) {
                    DataType promoted = left.dataType() == DataType.INT64
                                    || right.dataType() == DataType.INT64
                            ? DataType.INT64 : DataType.INT32;
                    ScalarValue actual = foldBinary(kind, left, right, promoted);
                    assertEquals(expectedArithmetic(kind, left, right, promoted), actual,
                            () -> kind + " " + left + " " + right);
                }
            }
        }
        assertAll(
                () -> assertEquals(Integer.MIN_VALUE,
                        foldBinary(
                                BinaryArithmeticKind.ADD,
                                ScalarValue.int32(Integer.MAX_VALUE),
                                ScalarValue.int32(1),
                                DataType.INT32).int32Value()),
                () -> assertEquals(Long.MAX_VALUE,
                        foldBinary(
                                BinaryArithmeticKind.SUB,
                                ScalarValue.int64(Long.MIN_VALUE),
                                ScalarValue.int64(1),
                                DataType.INT64).int64Value()));
    }

    @Test
    void foldsEverySignedIntegralComparisonForAllWidthPairs() {
        for (BinaryComparisonKind kind : BinaryComparisonKind.values()) {
            for (ScalarValue left : List.of(ScalarValue.int32(-1), ScalarValue.int64(Long.MAX_VALUE))) {
                for (ScalarValue right : List.of(ScalarValue.int32(0), ScalarValue.int64(-1))) {
                    long a = integralLong(left);
                    long b = integralLong(right);
                    boolean expected = switch (kind) {
                        case GREATER_THAN -> a > b;
                        case GREATER_OR_EQUAL -> a >= b;
                        case LESS_THAN -> a < b;
                        case LESS_OR_EQUAL -> a <= b;
                        case EQUAL -> a == b;
                        case NOT_EQUAL -> a != b;
                    };
                    assertEquals(expected, foldBinary(
                            kind, left, right, DataType.BOOL).booleanValue());
                }
            }
        }
    }

    @Test
    void foldsSelectedSameTypedIntegralScalarRowsInOperandOrder() {
        for (DataType type : List.of(DataType.INT32, DataType.INT64)) {
            for (ScalarElementwiseKind kind : List.of(
                    ScalarElementwiseKind.ADD,
                    ScalarElementwiseKind.SUB,
                    ScalarElementwiseKind.MUL,
                    ScalarElementwiseKind.MIN,
                    ScalarElementwiseKind.MAX)) {
                ScalarValue left = type == DataType.INT32
                        ? ScalarValue.int32(Integer.MIN_VALUE) : ScalarValue.int64(Long.MIN_VALUE);
                ScalarValue right = type == DataType.INT32
                        ? ScalarValue.int32(1) : ScalarValue.int64(1);
                ScalarValue result = foldScalar(kind, left, right);
                assertEquals(expectedScalar(kind, left, right), result);
            }
        }
        assertEquals(-5, foldScalar(
                ScalarElementwiseKind.SUB,
                ScalarValue.int32(2), ScalarValue.int32(7)).int32Value());
    }

    @Test
    void propagatesInOneScanAndAllocatesSyntheticSourcesBeforeRetainedOutputs() {
        TensorDescriptor descriptor = descriptor(DataType.INT32, Shape.of(2), Optional.empty());
        List<GraphValue> values = List.of(
                value(0, descriptor), value(1, descriptor), value(2, descriptor),
                value(3, descriptor), value(4, descriptor), value(5, descriptor));
        Operation add = operation(BinaryArithmeticKind.ADD);
        Operation mul = operation(BinaryArithmeticKind.MUL);
        Operation output = new Operation(
                ScalarElementwiseKind.ADD,
                new ScalarValueAttrs(ScalarValue.int32(0)));
        CompiledGraphModel graph = graph(
                values,
                List.of(
                        node(0, add, List.of(0L, 1L), List.of(3L)),
                        node(1, mul, List.of(3L, 2L), List.of(4L)),
                        node(2, output, List.of(4L), List.of(5L))),
                List.of(0L, 1L, 2L),
                List.of(5L),
                List.of(GraphPhase.FORWARD, GraphPhase.FORWARD, GraphPhase.FORWARD));
        CompileTimeConstantGraph source = sidecar(graph, Map.of(
                0L, ScalarValue.int32(2),
                1L, ScalarValue.int32(3),
                2L, ScalarValue.int32(4)));

        CompileTimeConstantGraph result = ForwardConstantFolding.fold(source);

        assertAll(
                () -> assertEquals(ids(0, 1, 2, 3, 4), result.graph().inputs()),
                () -> assertEquals(5, result.constants().size()),
                () -> assertEquals(5, result.constants().get(new ValueId(3)).value().int32Value()),
                () -> assertEquals(20, result.constants().get(new ValueId(4)).value().int32Value()),
                () -> assertEquals(1, result.graph().nodes().size()),
                () -> assertSame(output, result.graph().nodes().getFirst().operation()),
                () -> assertEquals(List.of(new ValueId(4)),
                        result.graph().nodes().getFirst().inputs()),
                () -> assertEquals(List.of(new ValueId(5)), result.graph().outputs()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> result.graph().inputs().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> result.constants().clear()));
    }

    @Test
    void retainsGraphOutputsMultiOutputFloatingAndUnselectedButFoldsBackward() {
        TensorDescriptor intDescriptor = descriptor(
                DataType.INT32, Shape.of(3), Optional.empty());
        TensorDescriptor floatDescriptor = descriptor(
                DataType.FLOAT32, Shape.of(3), Optional.empty());
        TensorDescriptor indexDescriptor = descriptor(
                DataType.INT64, Shape.of(1), Optional.empty());

        CompileTimeConstantGraph graphOutput = oneNodeSidecar(
                operation(BinaryArithmeticKind.ADD),
                List.of(ScalarValue.int32(1), ScalarValue.int32(2)),
                intDescriptor,
                GraphPhase.FORWARD,
                true);
        CompileTimeConstantGraph backward = oneNodeSidecar(
                operation(BinaryArithmeticKind.ADD),
                List.of(ScalarValue.int32(1), ScalarValue.int32(2)),
                intDescriptor,
                GraphPhase.BACKWARD,
                false);
        CompileTimeConstantGraph floating = oneNodeSidecar(
                operation(BinaryArithmeticKind.ADD),
                List.of(ScalarValue.float32(-0.0f),
                        ScalarValue.float32(Float.intBitsToFloat(0x7fc0_0042))),
                floatDescriptor,
                GraphPhase.FORWARD,
                false);
        CompileTimeConstantGraph scalarDiv = oneNodeSidecar(
                new Operation(ScalarElementwiseKind.DIV,
                        new ScalarValueAttrs(ScalarValue.int32(1))),
                List.of(ScalarValue.int32(8)),
                intDescriptor,
                GraphPhase.FORWARD,
                false);

        CompiledGraphModel multiGraph = graph(
                List.of(value(0, intDescriptor), value(1, intDescriptor), value(2, indexDescriptor)),
                List.of(node(0, new Operation(TopKKind.TOP_K, new TopKAttrs(1, 0L, true, true)),
                        List.of(0L), List.of(1L, 2L))),
                List.of(0L), List.of(1L), List.of(GraphPhase.FORWARD));
        CompileTimeConstantGraph multi = sidecar(
                multiGraph, Map.of(0L, ScalarValue.int32(4)));

        CompileTimeConstantGraph foldedBackward = ForwardConstantFolding.fold(backward);
        assertAll(
                () -> assertSame(graphOutput, ForwardConstantFolding.fold(graphOutput)),
                () -> assertNotSame(backward, foldedBackward),
                () -> assertEquals(1, foldedBackward.graph().nodes().size()),
                () -> assertEquals(
                        GraphPhase.FORWARD,
                        foldedBackward.graph().nodePhases().get(new NodeId(0))),
                () -> assertEquals(3, foldedBackward.constants().size()),
                () -> assertSame(floating, ForwardConstantFolding.fold(floating)),
                () -> assertSame(scalarDiv, ForwardConstantFolding.fold(scalarDiv)),
                () -> assertSame(multi, ForwardConstantFolding.fold(multi)));
    }

    @Test
    void foldsDynamicAndResolvedLayoutSplatsWithoutShapeExpansion() {
        TensorDescriptor dynamic = descriptor(
                DataType.INT64,
                Shape.ofDimensions(new DynamicDimension("batch")),
                Optional.empty());
        Shape staticShape = Shape.of(0, 128);
        TensorDescriptor resolved = descriptor(
                DataType.INT64,
                staticShape,
                Optional.of(LayoutDescriptor.contiguous(staticShape)));

        assertAll(
                () -> assertEquals(7L, foldScalar(
                        ScalarElementwiseKind.ADD,
                        ScalarValue.int64(3), ScalarValue.int64(4), dynamic).int64Value()),
                () -> assertEquals(7L, foldScalar(
                        ScalarElementwiseKind.ADD,
                        ScalarValue.int64(3), ScalarValue.int64(4), resolved).int64Value()));
    }

    private static ScalarValue foldLogical(BooleanLogicalKind kind, boolean... values) {
        List<ScalarValue> scalars = new ArrayList<>();
        for (boolean value : values) scalars.add(ScalarValue.bool(value));
        return foldOne(operation(kind), scalars,
                descriptor(DataType.BOOL, Shape.of(2), Optional.empty()));
    }

    private static ScalarValue foldBinary(
            OperationKind kind, ScalarValue left, ScalarValue right, DataType outputType) {
        return foldOne(operation(kind), List.of(left, right),
                descriptor(outputType, Shape.of(2), Optional.empty()));
    }

    private static ScalarValue foldScalar(
            ScalarElementwiseKind kind, ScalarValue left, ScalarValue right) {
        return foldScalar(kind, left, right,
                descriptor(left.dataType(), Shape.of(2), Optional.empty()));
    }

    private static ScalarValue foldScalar(
            ScalarElementwiseKind kind,
            ScalarValue left,
            ScalarValue right,
            TensorDescriptor descriptor) {
        return foldOne(new Operation(kind, new ScalarValueAttrs(right)), List.of(left), descriptor);
    }

    private static ScalarValue foldOne(
            Operation operation, List<ScalarValue> inputs, TensorDescriptor outputDescriptor) {
        CompileTimeConstantGraph source = oneNodeSidecar(
                operation, inputs, outputDescriptor, GraphPhase.FORWARD, false);
        CompileTimeConstantGraph folded = ForwardConstantFolding.fold(source);
        ValueId synthetic = folded.graph().inputs().get(inputs.size());
        return folded.constants().get(synthetic).value();
    }

    private static CompileTimeConstantGraph oneNodeSidecar(
            Operation operation,
            List<ScalarValue> inputValues,
            TensorDescriptor outputDescriptor,
            GraphPhase phase,
            boolean nodeIsGraphOutput) {
        List<GraphValue> values = new ArrayList<>();
        Map<Long, ScalarValue> constants = new LinkedHashMap<>();
        List<Long> inputs = new ArrayList<>();
        for (int index = 0; index < inputValues.size(); index++) {
            ScalarValue scalar = inputValues.get(index);
            TensorDescriptor inputDescriptor = descriptor(
                    scalar.dataType(), outputDescriptor.shape(), outputDescriptor.layout());
            values.add(value(index, inputDescriptor));
            inputs.add((long) index);
            constants.put((long) index, scalar);
        }
        long foldedOutput = inputValues.size();
        long graphOutput = nodeIsGraphOutput ? foldedOutput : foldedOutput + 1;
        values.add(value(foldedOutput, outputDescriptor));
        List<CompiledNode> nodes = new ArrayList<>();
        nodes.add(node(0, operation, inputs, List.of(foldedOutput)));
        List<GraphPhase> phases = new ArrayList<>();
        phases.add(phase);
        if (!nodeIsGraphOutput) {
            values.add(value(graphOutput, outputDescriptor));
            nodes.add(node(1, outputConsumer(outputDescriptor.dataType()),
                    List.of(foldedOutput), List.of(graphOutput)));
            phases.add(GraphPhase.FORWARD);
        }
        return sidecar(graph(values, nodes, inputs, List.of(graphOutput), phases), constants);
    }

    private static Operation outputConsumer(DataType dataType) {
        return switch (dataType) {
            case BOOL -> operation(BooleanLogicalKind.NOT);
            case INT32 -> new Operation(
                    ScalarElementwiseKind.ADD,
                    new ScalarValueAttrs(ScalarValue.int32(0)));
            case INT64 -> new Operation(
                    ScalarElementwiseKind.ADD,
                    new ScalarValueAttrs(ScalarValue.int64(0)));
            case FLOAT64, FLOAT32, BFLOAT16 -> operation(UnaryElementwiseKind.NEG);
        };
    }

    private static ScalarValue expectedArithmetic(
            BinaryArithmeticKind kind, ScalarValue left, ScalarValue right, DataType type) {
        if (type == DataType.INT32) {
            int a = left.int32Value();
            int b = right.int32Value();
            return switch (kind) {
                case ADD -> ScalarValue.int32(a + b);
                case SUB -> ScalarValue.int32(a - b);
                case MUL -> ScalarValue.int32(a * b);
                case MIN -> ScalarValue.int32(Math.min(a, b));
                case MAX -> ScalarValue.int32(Math.max(a, b));
                case DIV, POW -> throw new AssertionError();
            };
        }
        long a = integralLong(left);
        long b = integralLong(right);
        return switch (kind) {
            case ADD -> ScalarValue.int64(a + b);
            case SUB -> ScalarValue.int64(a - b);
            case MUL -> ScalarValue.int64(a * b);
            case MIN -> ScalarValue.int64(Math.min(a, b));
            case MAX -> ScalarValue.int64(Math.max(a, b));
            case DIV, POW -> throw new AssertionError();
        };
    }

    private static ScalarValue expectedScalar(
            ScalarElementwiseKind kind, ScalarValue left, ScalarValue right) {
        return expectedArithmetic(BinaryArithmeticKind.valueOf(kind.name()),
                left, right, left.dataType());
    }

    private static long integralLong(ScalarValue value) {
        return value.dataType() == DataType.INT32 ? value.int32Value() : value.int64Value();
    }

    private static CompileTimeConstantGraph sidecar(
            CompiledGraphModel graph, Map<Long, ScalarValue> values) {
        Map<ValueId, CompileTimeConstantGraph.Splat> facts = new LinkedHashMap<>();
        values.forEach((id, value) -> facts.put(
                new ValueId(id), new CompileTimeConstantGraph.Splat(value)));
        return new CompileTimeConstantGraph(graph, facts);
    }

    private static TensorDescriptor descriptor(
            DataType type, Shape shape, Optional<LayoutDescriptor> layout) {
        return new TensorDescriptor(type, shape, layout, false);
    }

    private static Operation operation(OperationKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }

    private static GraphValue value(long id, TensorDescriptor descriptor) {
        return new GraphValue(new ValueId(id), descriptor);
    }

    private static CompiledNode node(
            long id, Operation operation, List<Long> inputs, List<Long> outputs) {
        return new CompiledNode(
                new NodeId(id), operation,
                inputs.stream().map(ValueId::new).toList(),
                outputs.stream().map(ValueId::new).toList());
    }

    private static CompiledGraphModel graph(
            List<GraphValue> values,
            List<CompiledNode> nodes,
            List<Long> inputs,
            List<Long> outputs,
            List<GraphPhase> phases) {
        Map<NodeId, GraphPhase> phaseMap = new LinkedHashMap<>();
        for (int index = 0; index < nodes.size(); index++) {
            phaseMap.put(nodes.get(index).id(), phases.get(index));
        }
        return new CompiledGraphModel(
                values,
                nodes,
                inputs.stream().map(ValueId::new).toList(),
                outputs.stream().map(ValueId::new).toList(),
                phaseMap);
    }

    private static List<ValueId> ids(long... values) {
        return Arrays.stream(values).mapToObj(ValueId::new).toList();
    }
}
