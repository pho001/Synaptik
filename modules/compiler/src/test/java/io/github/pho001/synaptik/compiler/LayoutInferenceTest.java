package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

final class LayoutInferenceTest {
    @Test
    void acceptsLayoutAndWindowFamilies() {
        Tensor x = tensor(Shape.of(2, 3));
        Tensor expanded = x.expandDims(1);
        Tensor windows = x.unfold(1, 2, 1);
        Tensor update = tensor(Shape.of(2, 1));
        List<Tensor> outputs = List.of(
                x.contiguous(), x.reshape(Shape.of(6)), x.expand(Shape.of(4, 2, 3)),
                expanded.squeeze(1), expanded, x.permute(1, 0), x.sliceAxis(1, 0, 2),
                x.sliceUpdate(update, new long[] {1}, new int[] {1}, new long[] {1}),
                x.cropToShape(Shape.of(1, 2), Shape.of(1, 1)),
                x.pad(new long[] {0, 1}, new long[] {0, 1}, ScalarValue.float32(0)),
                x.tile(1, 2), Tensor.concat(0, x, x), Tensor.stack(0, x, x),
                windows, windows.foldAxis(1, 3, 1));
        for (Tensor output : outputs) {
            assertDoesNotThrow(() -> CapturedGraphInference.inferAndValidate(
                    GraphCapture.capture(List.of(output))));
        }
        Tensor image = tensor(Shape.of(1, 2, 4, 4));
        Window2dAttrs window = new Window2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false);
        Tensor columns = image.unfold2d(window);
        for (Tensor output : List.of(
                columns,
                image.unfold2d(window, ScalarValue.float32(0)),
                columns.fold2d(image.descriptor().shape(), window))) {
            assertDoesNotThrow(() -> CapturedGraphInference.inferAndValidate(
                    GraphCapture.capture(List.of(output))));
        }
    }

    @Test
    void rejectsManuallyConstructedInvalidLayoutRuleCategories() {
        TensorDescriptor vector = descriptor(Shape.of(2));

        assertInvalid(
                new Operation(ShapeTransformKind.RESHAPE,
                        new TargetShapeAttrs(Shape.of(3))),
                List.of(vector), descriptor(Shape.of(3)),
                "constraint reshape element count failed");
        assertInvalid(
                new Operation(ShapeTransformKind.EXPAND,
                        new TargetShapeAttrs(Shape.of(3))),
                List.of(vector), descriptor(Shape.of(3)),
                "incompatible expand dimension");
        assertInvalid(
                new Operation(AxisTransformKind.SQUEEZE,
                        new AxisTransformAttrs(0)),
                List.of(vector), descriptor(Shape.scalar()),
                "squeeze axis must be singleton");
        assertInvalid(
                new Operation(SliceKind.SLICE,
                        new SliceAttrs(List.of(1L), List.of(2L),
                                List.of(0), List.of(1L))),
                List.of(vector), descriptor(Shape.of(2)),
                "constraint slice axis 0 failed");
        assertInvalid(
                new Operation(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(0L), List.of(1L),
                                List.of(0), List.of(1L))),
                List.of(vector, descriptor(Shape.of(2))), vector,
                "slice update descriptor mismatch");
        assertInvalid(
                new Operation(PadKind.PAD,
                        new PadAttrs(List.of(1L), List.of(1L),
                                ScalarValue.float64(0))),
                List.of(vector), descriptor(Shape.of(4)),
                "pad attributes mismatch");
        assertInvalid(
                new Operation(TileKind.TILE, new TileAttrs(List.of(2L, 2L))),
                List.of(vector), descriptor(Shape.of(4)),
                "tile repeat rank mismatch");
        assertInvalid(
                new Operation(TensorCompositionKind.CONCAT,
                        new CompositionAxisAttrs(0)),
                List.of(descriptor(Shape.of(2, 3)), descriptor(Shape.of(2, 4))),
                descriptor(Shape.of(4, 3)), "concat dimension mismatch");
        assertInvalid(
                new Operation(WindowTransformKind.UNFOLD_AXIS,
                        new UnfoldAxisAttrs(0, 3, 1)),
                List.of(vector), descriptor(Shape.of(1, 3)),
                "window does not fit");
        Window2dAttrs window = new Window2dAttrs(
                2, 2, 1, 1, 0, 0, 1, 1, false);
        assertInvalid(
                new Operation(WindowTransformKind.FOLD2D,
                        new Fold2dAttrs(Shape.of(1, 1, 3, 3), window)),
                List.of(descriptor(Shape.of(1, 4, 3))),
                descriptor(Shape.of(1, 1, 3, 3)), "fold2d columns mismatch");
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

    private static TensorDescriptor descriptor(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape, Optional.empty(), true);
    }

    private static Tensor tensor(Shape shape) {
        return TensorFactory.create(descriptor(shape));
    }
}
