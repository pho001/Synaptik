package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

final class IndexingInferenceTest {
    @Test
    void acceptsIndexingFamilies() {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        Tensor indices = tensor(DataType.INT64, Shape.of(2), false);
        Tensor elementIndices = tensor(DataType.INT32, Shape.of(2, 4), false);
        Tensor updates = tensor(DataType.FLOAT32, Shape.of(2, 4), true);
        List<Tensor> outputs = List.of(
                data.select(1, 1), data.gather(indices, 1), indices.oneHot(4),
                data.gatherElements(elementIndices, 1),
                data.scatterElements(elementIndices, updates, 1));
        for (Tensor output : outputs) {
            assertDoesNotThrow(() -> CapturedGraphInference.inferAndValidate(
                    GraphCapture.capture(List.of(output))));
        }
    }

    @Test
    void coversEveryKindAndAttributesVariant() {
        TensorDescriptor data = descriptor(DataType.FLOAT32, Shape.of(2, 3), true);
        TensorDescriptor indices = descriptor(DataType.INT64, Shape.of(2), false);
        TensorDescriptor element = descriptor(DataType.INT32, Shape.of(2, 4), false);
        TensorDescriptor updates = descriptor(DataType.FLOAT32, Shape.of(2, 4), true);
        assertDoesNotThrow(() -> IndexingInference.infer(
                new Operation(SelectKind.SELECT, new SelectAttrs(1, 1)), List.of(data)));
        for (var kind : AxisGatherKind.values()) {
            assertDoesNotThrow(() -> IndexingInference.infer(
                    new Operation(kind, new IndexAxisAttrs(1)),
                    kind == AxisGatherKind.GATHER
                            ? List.of(data, indices) : List.of(data, element)));
        }
        assertDoesNotThrow(() -> IndexingInference.infer(
                new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(3)), List.of(indices)));
        TensorDescriptor tuples = descriptor(DataType.INT64, Shape.of(2, 1), false);
        TensorDescriptor ndUpdates = descriptor(DataType.FLOAT32, Shape.of(2, 3), true);
        assertDoesNotThrow(() -> IndexingInference.infer(
                new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(0)),
                List.of(data, tuples)));
        assertDoesNotThrow(() -> IndexingInference.infer(
                new Operation(ScatterNdKind.SCATTER_ND,
                        new ScatterNdAttrs(0, ScatterReduction.NONE)),
                List.of(data, tuples, ndUpdates)));
        for (var kind : AxisScatterKind.values()) {
            Operation operation = kind == AxisScatterKind.SCATTER_ADD
                    ? new Operation(kind, new IndexAxisAttrs(1))
                    : new Operation(kind,
                            new ScatterElementsAttrs(1, ScatterReduction.NONE));
            List<TensorDescriptor> inputs = kind == AxisScatterKind.SCATTER_ADD
                    ? List.of(data, indices,
                            descriptor(DataType.FLOAT32, Shape.of(2, 2), true))
                    : List.of(data, element, updates);
            assertDoesNotThrow(() -> IndexingInference.infer(operation, inputs));
        }
    }

    @Test
    void rejectsManuallyConstructedInvalidIndexingRuleCategories() {
        TensorDescriptor data = descriptor(DataType.FLOAT32, Shape.of(2, 3), true);
        TensorDescriptor floatingIndices =
                descriptor(DataType.FLOAT32, Shape.of(2), false);
        TensorDescriptor indices = descriptor(DataType.INT64, Shape.of(2), false);

        assertInvalid(
                new Operation(SelectKind.SELECT, new SelectAttrs(0, 2)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2), true)),
                descriptor(DataType.FLOAT32, Shape.scalar(), true),
                "constraint select index failed");
        assertInvalid(
                new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1)),
                List.of(data, floatingIndices), data, "indices must be INT32 or INT64");
        assertInvalid(
                new Operation(AxisGatherKind.GATHER_ELEMENTS, new IndexAxisAttrs(1)),
                List.of(data, descriptor(DataType.INT64, Shape.of(4, 2), false)),
                data, "gather-elements non-axis dimension mismatch");
        assertInvalid(
                new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(3)),
                List.of(floatingIndices),
                descriptor(DataType.BOOL, Shape.of(2, 3), false),
                "indices must be INT32 or INT64");
        assertInvalid(
                new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(0)),
                List.of(data, descriptor(DataType.INT64, Shape.of(2, 0), false)),
                data, "invalid tuple depth");
        assertInvalid(
                new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(1, ScatterReduction.NONE)),
                List.of(data, indices,
                        descriptor(DataType.FLOAT64, Shape.of(2), true)),
                data, "updates data type must match data");
        assertInvalid(
                new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(1)),
                List.of(data, indices,
                        descriptor(DataType.FLOAT32, Shape.of(2, 3), true)),
                data, "updates shape mismatch");
        assertInvalid(
                new Operation(ScatterNdKind.SCATTER_ND,
                        new ScatterNdAttrs(0, ScatterReduction.NONE)),
                List.of(data,
                        descriptor(DataType.INT64, Shape.of(2, 1), false),
                        descriptor(DataType.FLOAT32, Shape.of(2, 2), true)),
                data, "updates shape mismatch");
    }

    private static void assertInvalid(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output,
            String expectedDetail) {
        CompiledGraphModel graph = graph(operation, inputs, output);
        String message = assertThrows(IllegalArgumentException.class,
                () -> CapturedGraphInference.inferAndValidate(graph)).getMessage();
        assertAll(
                () -> assertTrue(message.startsWith("nodes[0] NodeId[value=0] ")),
                () -> assertTrue(message.contains(expectedDetail), message));
    }

    private static CompiledGraphModel graph(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output) {
        List<GraphValue> values = new ArrayList<>();
        List<ValueId> inputIds = new ArrayList<>();
        long id = 0;
        for (TensorDescriptor input : inputs) {
            ValueId valueId = new ValueId(id++);
            values.add(new GraphValue(valueId, input));
            inputIds.add(valueId);
        }
        ValueId outputId = new ValueId(id);
        values.add(new GraphValue(outputId, output));
        NodeId nodeId = new NodeId(0);
        return new CompiledGraphModel(
                values,
                List.of(new CompiledNode(
                        nodeId, operation, inputIds, List.of(outputId))),
                inputIds,
                List.of(outputId),
                Map.of(nodeId, GraphPhase.FORWARD));
    }

    private static TensorDescriptor descriptor(
            DataType dataType, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad);
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(descriptor(dataType, shape, requiresGrad));
    }
}
