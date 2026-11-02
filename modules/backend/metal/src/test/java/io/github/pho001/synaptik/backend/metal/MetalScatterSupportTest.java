package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Behavioral admission and raw-image preflight checks, independent of provider truth pins. */
class MetalScatterSupportTest {
    @Test
    void generalGatherCotangentLowersArbitraryIndexRankAndChecksGradientOr() {
        var op = new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(1));
        var inputs = List.of(tensor(DataType.FLOAT32, false, 2, 3, 4), tensor(DataType.INT64, false, 2, 2),
                tensor(DataType.FLOAT32, true, 2, 2, 2, 4));
        var outputs = List.of(tensor(DataType.FLOAT32, true, 2, 3, 4));
        assertTrue(MetalScatterSupport.supports(op, inputs, outputs));
        var node = MetalScatterSupport.lower(op, new int[] {0, 1, 2}, new int[] {3}, inputs, outputs);
        assertArrayEquals(new long[] {1}, node.attributeWords());
        var raw = values(inputs, outputs);
        assertDoesNotThrow(() -> MetalScatterSupport.validate(node, raw));
        raw.set(3, value(DataType.FLOAT32, false, 2, 3, 4));
        assertThrows(IllegalArgumentException.class, () -> MetalScatterSupport.validate(node, raw));
        assertFalse(MetalScatterSupport.supports(op, inputs, List.of(tensor(DataType.FLOAT32, false, 2, 3, 4))));
    }

    @Test
    void independentEncodedValidatorRejectsWrongIndexTypeAxisShapeAndPhysicalOutput() {
        var node = MetalMpsGraphProgram.Node.scatterAdd(0, 1, 2, 3, 1);
        var raw = new ArrayList<>(List.of(value(DataType.FLOAT32, false, 2, 3),
                value(DataType.INT64, false, 4), value(DataType.FLOAT32, false, 2, 4),
                value(DataType.FLOAT32, false, 2, 3)));
        assertDoesNotThrow(() -> MetalScatterSupport.validate(node, raw));
        raw.set(1, value(DataType.FLOAT32, false, 4));
        assertThrows(IllegalArgumentException.class, () -> MetalScatterSupport.validate(node, raw));
        raw.set(1, value(DataType.INT64, false, 4));
        raw.set(2, value(DataType.FLOAT32, false, 4, 2));
        assertThrows(IllegalArgumentException.class, () -> MetalScatterSupport.validate(node, raw));
        raw.set(2, value(DataType.FLOAT32, false, 2, 4));
        raw.set(3, new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {2, 3},
                Optional.of(LayoutDescriptor.of(Shape.of(2, 3), new long[] {4, 1}, 1, true)), false, false));
        assertThrows(IllegalArgumentException.class, () -> MetalScatterSupport.validate(node, raw));
        raw.set(3, value(DataType.FLOAT32, false, 2, 3));
        var invalidAxis = MetalMpsGraphProgram.Node.scatterAdd(0, 1, 2, 3, 2);
        assertThrows(IllegalArgumentException.class, () -> MetalScatterSupport.validate(invalidAxis, raw));
    }

    @Test
    void reductionWiresAndBatchedNdShapeRemainExact() {
        for (ScatterReduction reduction : ScatterReduction.values()) {
            var op = new Operation(ScatterNdKind.SCATTER_ND, new ScatterNdAttrs(1, reduction));
            var inputs = List.of(tensor(DataType.FLOAT32, false, 2, 3, 4),
                    tensor(DataType.INT32, false, 2, 5, 1), tensor(DataType.FLOAT32, false, 2, 5, 4));
            var outputs = List.of(tensor(DataType.FLOAT32, false, 2, 3, 4));
            assertTrue(MetalScatterSupport.supports(op, inputs, outputs));
            var node = MetalScatterSupport.lower(op, new int[] {0, 1, 2}, new int[] {3}, inputs, outputs);
            assertArrayEquals(new long[] {1, reduction.ordinal() + 1L}, node.attributeWords());
            var raw = values(inputs, outputs);
            raw.set(1, value(DataType.INT32, false, 1, 5, 1));
            assertThrows(IllegalArgumentException.class, () -> MetalScatterSupport.validate(node, raw));
        }
    }

    @Test
    void originalCarrierDispatchNeverUnlocksLowArithmeticOrFloat64Arithmetic() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.INT32, DataType.INT64,
                DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16)) {
            for (ScatterReduction reduction : ScatterReduction.values()) {
                var op = new Operation(AxisScatterKind.SCATTER_ELEMENTS, new ScatterElementsAttrs(0, reduction));
                var inputs = List.of(tensor(type, false, 3), tensor(DataType.INT32, false, 2), tensor(type, false, 2));
                var outputs = List.of(tensor(type, false, 3));
                boolean selected = type == DataType.FLOAT32
                        || (type == DataType.INT32 || type == DataType.INT64) && reduction != ScatterReduction.NONE
                        || type == DataType.FLOAT64 && (reduction == ScatterReduction.MIN || reduction == ScatterReduction.MAX);
                var node = MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.SCATTER_ELEMENTS,
                        new int[] {0, 1, 2}, new int[] {3}, MetalMpsGraphProgram.AttributeKind.SCATTER_ELEMENTS,
                        new long[] {0, reduction.ordinal() + 1L});
                var raw = values(inputs, outputs);
                if (selected) {
                    assertTrue(MetalScatterSupport.supports(op, inputs, outputs), type + "/" + reduction);
                    assertTrue(MetalScatterSupport.handles(node, raw));
                    assertDoesNotThrow(() -> MetalScatterSupport.validate(node, raw));
                } else {
                    assertFalse(MetalScatterSupport.supports(op, inputs, outputs), type + "/" + reduction);
                    assertFalse(MetalScatterSupport.handles(node, raw));
                    assertThrows(IllegalArgumentException.class, () -> MetalScatterSupport.validate(node, raw));
                }
                if ((type == DataType.BFLOAT16 || type == DataType.FLOAT16) && reduction == ScatterReduction.NONE) {
                    assertDoesNotThrow(() -> MetalScatterSupport.lower(op, new int[] {0, 1, 2}, new int[] {3}, inputs, outputs));
                }
            }
        }
    }

    @Test
    void broadcastAxisDoesNotHideCollisionsBetweenNonbroadcastInputAxes() {
        var operation = new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(0));
        Shape shape = Shape.of(2, 2, 2);
        var indices = tensor(DataType.INT32, false, 1);
        var updates = tensor(DataType.FLOAT32, false, 1, 2, 2);
        var outputs = List.of(tensor(DataType.FLOAT32, false, 2, 2, 2));
        var node = MetalMpsGraphProgram.Node.scatterAdd(0, 1, 2, 3, 0);
        for (boolean collision : List.of(false, true)) {
            var base = new TensorDescriptor(DataType.FLOAT32, shape,
                    Optional.of(LayoutDescriptor.of(shape,
                            collision ? new long[] {0, 1, 1} : new long[] {0, 2, 1}, 0, true)), false);
            var inputs = List.of(base, indices, updates);
            var raw = values(inputs, outputs);
            if (collision) {
                assertFalse(MetalScatterSupport.supports(operation, inputs, outputs));
                assertThrows(IllegalArgumentException.class, () -> MetalScatterSupport.validate(node, raw));
                assertThrows(IllegalArgumentException.class, () -> MetalScatterSupport.lower(
                        operation, new int[] {0, 1, 2}, new int[] {3}, inputs, outputs));
            } else {
                assertTrue(MetalScatterSupport.supports(operation, inputs, outputs));
                assertDoesNotThrow(() -> MetalScatterSupport.validate(node, raw));
            }
        }
    }

    private static TensorDescriptor tensor(DataType type, boolean gradient, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), gradient);
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(DataType type, boolean gradient, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions, gradient);
    }

    private static ArrayList<MetalMpsGraphProgram.ValueDescriptor> values(
            List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        for (var descriptor : inputs) values.add(MetalMpsGraphProgram.ValueDescriptor.from(descriptor));
        for (var descriptor : outputs) values.add(MetalMpsGraphProgram.ValueDescriptor.from(descriptor));
        return values;
    }
}
