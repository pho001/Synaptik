package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.*;
import io.github.pho001.synaptik.model.operation.elementwise.cast.*;
import io.github.pho001.synaptik.model.operation.elementwise.classification.*;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.*;
import io.github.pho001.synaptik.model.operation.elementwise.logical.*;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.*;
import io.github.pho001.synaptik.model.operation.elementwise.selection.*;
import io.github.pho001.synaptik.model.operation.elementwise.unary.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

final class ElementwiseInferenceTest {
    @Test
    void acceptsRepresentativeElementwiseInventory() {
        Tensor a = tensor(DataType.FLOAT32);
        Tensor b = tensor(DataType.BFLOAT16);
        List<Tensor> outputs = List.of(
                a.add(b), a.div(b), a.neg(), a.exp(),
                a.add(ScalarValue.float32(2)),
                a.clamp(ScalarValue.float32(-1), ScalarValue.float32(1)),
                a.greaterThan(b), a.isFinite(), a.cast(DataType.INT64));
        for (Tensor output : outputs) {
            assertDoesNotThrow(() -> CapturedGraphInference.inferAndValidate(
                    GraphCapture.capture(List.of(output))));
        }
        Tensor predicate = tensor(DataType.BOOL);
        assertDoesNotThrow(() -> CapturedGraphInference.inferAndValidate(
                GraphCapture.capture(List.of(predicate.logicalNot()))));
    }

    @Test
    void coversEveryKindAndAttributesVariant() {
        TensorDescriptor floating = descriptor(DataType.FLOAT32);
        TensorDescriptor bool = descriptor(DataType.BOOL);
        for (var kind : BinaryArithmeticKind.values()) {
            assertDoesNotThrow(() -> ElementwiseInference.infer(
                    new Operation(kind, NoOperationAttrs.INSTANCE),
                    List.of(floating, floating)));
        }
        for (var kind : UnaryElementwiseKind.values()) {
            assertDoesNotThrow(() -> ElementwiseInference.infer(
                    new Operation(kind, NoOperationAttrs.INSTANCE), List.of(floating)));
        }
        for (var kind : BinaryComparisonKind.values()) {
            assertDoesNotThrow(() -> ElementwiseInference.infer(
                    new Operation(kind, NoOperationAttrs.INSTANCE),
                    List.of(floating, floating)));
        }
        for (var kind : FloatingClassificationKind.values()) {
            assertDoesNotThrow(() -> ElementwiseInference.infer(
                    new Operation(kind, NoOperationAttrs.INSTANCE), List.of(floating)));
        }
        for (var kind : BooleanLogicalKind.values()) {
            assertDoesNotThrow(() -> ElementwiseInference.infer(
                    new Operation(kind, NoOperationAttrs.INSTANCE),
                    kind == BooleanLogicalKind.NOT ? List.of(bool) : List.of(bool, bool)));
        }
        for (var kind : ScalarElementwiseKind.values()) {
            Operation operation = kind == ScalarElementwiseKind.CLAMP
                    ? new Operation(kind, new ClampRangeAttrs(
                            ScalarValue.float32(-1), ScalarValue.float32(1)))
                    : new Operation(kind, new ScalarValueAttrs(ScalarValue.float32(1)));
            assertDoesNotThrow(() -> ElementwiseInference.infer(operation, List.of(floating)));
        }
        assertDoesNotThrow(() -> ElementwiseInference.infer(
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                List.of(bool, floating, floating)));
        assertDoesNotThrow(() -> ElementwiseInference.infer(
                new Operation(CastKind.CAST, new CastAttrs(DataType.INT64)),
                List.of(floating)));
    }

    @Test
    void rejectsManuallyConstructedInvalidElementwiseRuleCategories() {
        TensorDescriptor floating = descriptor(DataType.FLOAT32);
        TensorDescriptor integral = descriptor(DataType.INT32);
        TensorDescriptor bool = descriptor(DataType.BOOL);

        assertInvalid(
                new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                List.of(bool, bool), bool, "numeric");
        assertInvalid(
                new Operation(ScalarElementwiseKind.ADD,
                        new ScalarValueAttrs(ScalarValue.float64(1))),
                List.of(floating), floating, "scalar data type must match input");
        assertInvalid(
                new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE),
                List.of(integral), integral, "input must be floating");
        assertInvalid(
                new Operation(BinaryComparisonKind.EQUAL, NoOperationAttrs.INSTANCE),
                List.of(bool, bool), bool, "numeric");
        assertInvalid(
                new Operation(BooleanLogicalKind.AND, NoOperationAttrs.INSTANCE),
                List.of(floating, bool), bool, "input[0] must be BOOL");
        assertInvalid(
                new Operation(FloatingClassificationKind.IS_FINITE,
                        NoOperationAttrs.INSTANCE),
                List.of(integral), bool, "input must be floating");
        assertInvalid(
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                List.of(floating, floating, floating), floating,
                "condition must be BOOL");
        assertInvalid(
                new Operation(CastKind.CAST, new CastAttrs(DataType.INT64)),
                List.of(floating), descriptor(DataType.FLOAT32), "output[0]");
    }

    private static void assertInvalid(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output,
            String expectedDetail) {
        CompiledGraphModel graph = graph(operation, inputs, List.of(output));
        String message = assertThrows(IllegalArgumentException.class,
                () -> CapturedGraphInference.inferAndValidate(graph)).getMessage();
        assertAll(
                () -> assertTrue(message.startsWith("nodes[0] NodeId[value=0] ")),
                () -> assertTrue(message.contains(expectedDetail), message));
    }

    private static CompiledGraphModel graph(
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        List<GraphValue> values = new ArrayList<>();
        List<ValueId> inputIds = new ArrayList<>();
        List<ValueId> outputIds = new ArrayList<>();
        long id = 0;
        for (TensorDescriptor input : inputs) {
            ValueId valueId = new ValueId(id++);
            values.add(new GraphValue(valueId, input));
            inputIds.add(valueId);
        }
        for (TensorDescriptor output : outputs) {
            ValueId valueId = new ValueId(id++);
            values.add(new GraphValue(valueId, output));
            outputIds.add(valueId);
        }
        NodeId nodeId = new NodeId(0);
        return new CompiledGraphModel(
                values,
                List.of(new CompiledNode(nodeId, operation, inputIds, outputIds)),
                inputIds,
                outputIds,
                Map.of(nodeId, GraphPhase.FORWARD));
    }

    private static TensorDescriptor descriptor(DataType dataType) {
        return new TensorDescriptor(
                dataType, Shape.of(2), Optional.empty(), dataType.isFloating());
    }

    private static Tensor tensor(DataType dataType) {
        return TensorFactory.create(descriptor(dataType));
    }
}
