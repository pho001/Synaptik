package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalExactSupportTest {
    @Test
    void originalFloat64UnarySupportsAffineReadsButRequiresExactShapeGradAndCanonicalOutput() {
        var operation = new Operation(UnaryElementwiseKind.FLOOR, NoOperationAttrs.INSTANCE);
        Shape shape = Shape.of(2, 3);
        var input = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {9, 2}, 3, true)), true);
        var output = descriptor(DataType.FLOAT64, true, 2, 3);
        assertTrue(MetalExactSupport.supports(operation, List.of(input), List.of(output)));
        assertFalse(MetalExactSupport.supports(operation, List.of(input),
                List.of(descriptor(DataType.FLOAT64, false, 2, 3))));
        assertFalse(MetalExactSupport.supports(operation, List.of(input),
                List.of(descriptor(DataType.FLOAT64, true, 3, 2))));
        assertFalse(MetalExactSupport.supports(operation, List.of(input), List.of(input)));
        var node = MetalExactSupport.lower(operation, new int[] {0}, new int[] {1}, List.of(input), List.of(output));
        assertDoesNotThrow(() -> MetalExactSupport.validate(node, List.of(
                MetalMpsGraphProgram.ValueDescriptor.from(input), MetalMpsGraphProgram.ValueDescriptor.from(output))));
    }

    @Test
    void broadcastViewsStillProveNonbroadcastInjectivityInActualAndEncodedAdmission() {
        Shape shape = Shape.of(2, 2, 2);
        for (DataType type : List.of(DataType.FLOAT64, DataType.INT32, DataType.INT64)) {
            Operation operation = type == DataType.FLOAT64
                    ? new Operation(UnaryElementwiseKind.FLOOR, NoOperationAttrs.INSTANCE)
                    : new Operation(BinaryArithmeticKind.MIN, NoOperationAttrs.INSTANCE);
            var output = descriptor(type, false, 2, 2, 2);
            var scalar = descriptor(type, false);
            for (boolean valid : List.of(false, true)) {
                var input = new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.of(
                        shape, new long[] {0, valid ? 2 : 1, 1}, 0, true)), false);
                List<TensorDescriptor> inputs = type == DataType.FLOAT64
                        ? List.of(input) : List.of(input, scalar);
                List<TensorDescriptor> outputs = List.of(output);
                if (valid) {
                    assertTrue(MetalExactSupport.supports(operation, inputs, outputs));
                    assertTrue(MetalCapabilityProvider.supportsOccurrence(operation, inputs, outputs));
                } else {
                    assertFalse(MetalExactSupport.supports(operation, inputs, outputs));
                    assertFalse(MetalCapabilityProvider.supportsOccurrence(operation, inputs, outputs));
                }
                var node = type == DataType.FLOAT64
                        ? MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.FLOOR,
                                new int[] {0}, new int[] {1}, MetalMpsGraphProgram.AttributeKind.NONE, new long[0])
                        : MetalMpsGraphProgram.Node.binary(MetalMpsGraphProgram.NodeKind.TENSOR_MIN, 0, 1, 2);
                List<MetalMpsGraphProgram.ValueDescriptor> values = type == DataType.FLOAT64
                        ? List.of(MetalMpsGraphProgram.ValueDescriptor.from(input),
                                MetalMpsGraphProgram.ValueDescriptor.from(output))
                        : List.of(MetalMpsGraphProgram.ValueDescriptor.from(input),
                                MetalMpsGraphProgram.ValueDescriptor.from(scalar),
                                MetalMpsGraphProgram.ValueDescriptor.from(output));
                if (valid) assertDoesNotThrow(() -> MetalExactSupport.validate(node, values));
                else assertThrows(IllegalArgumentException.class, () -> MetalExactSupport.validate(node, values));
            }
        }
    }

    @Test
    void exactOwnerDoesNotInterceptBaselineLowOrEnableFloat64Arithmetic() {
        var floor = new Operation(UnaryElementwiseKind.FLOOR, NoOperationAttrs.INSTANCE);
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16)) {
            var value = descriptor(type, false, 2);
            assertFalse(MetalExactSupport.handles(floor, List.of(value), List.of(value)));
        }
        var add = new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE);
        var value = descriptor(DataType.FLOAT64, false, 2);
        assertTrue(MetalExactSupport.handles(add, List.of(value, value), List.of(value)));
        assertFalse(MetalExactSupport.supports(add, List.of(value, value), List.of(value)));
        var node = MetalMpsGraphProgram.Node.binary(MetalMpsGraphProgram.NodeKind.ADD, 0, 1, 2);
        assertThrows(IllegalArgumentException.class, () -> MetalExactSupport.validate(node,
                List.of(encoded(DataType.FLOAT64, false, 2), encoded(DataType.FLOAT64, false, 2),
                        encoded(DataType.FLOAT64, false, 2))));
    }

    @Test
    void integralPromotionAndBroadcastAreRecheckedWithoutFloatingProxies() {
        var operation = new Operation(BinaryArithmeticKind.MUL, NoOperationAttrs.INSTANCE);
        var left = descriptor(DataType.INT32, false, 3, 1);
        var right = descriptor(DataType.INT64, false, 2);
        var output = descriptor(DataType.INT64, false, 3, 2);
        assertTrue(MetalExactSupport.supports(operation, List.of(left, right), List.of(output)));
        var node = MetalExactSupport.lower(operation, new int[] {0, 1}, new int[] {2},
                List.of(left, right), List.of(output));
        assertDoesNotThrow(() -> MetalExactSupport.validate(node, List.of(
                encoded(DataType.INT32, false, 3, 1), encoded(DataType.INT64, false, 2), encoded(DataType.INT64, false, 3, 2))));
        assertThrows(IllegalArgumentException.class, () -> MetalExactSupport.validate(node, List.of(
                encoded(DataType.INT32, false, 3, 1), encoded(DataType.INT64, false, 2), encoded(DataType.INT32, false, 3, 2))));
        assertThrows(IllegalArgumentException.class, () -> MetalExactSupport.validate(node, List.of(
                encoded(DataType.INT32, false, 3, 1), encoded(DataType.FLOAT64, false, 2), encoded(DataType.FLOAT64, false, 3, 2))));
        assertThrows(IllegalArgumentException.class, () -> MetalExactSupport.validate(node, List.of(
                encoded(DataType.INT32, false, 3, 1), encoded(DataType.INT64, false, 2), encoded(DataType.INT64, false, 2, 3))));
    }

    @Test
    void encodedScalarAndClampGrammarRetainsTypeAndExactBoundOrder() {
        var scalar = MetalMpsGraphProgram.Node.scalarValue(MetalMpsGraphProgram.NodeKind.SCALAR_MIN,
                0, 1, DataType.FLOAT64, 0);
        assertThrows(IllegalArgumentException.class, () -> MetalExactSupport.validate(scalar,
                List.of(encoded(DataType.INT64, false), encoded(DataType.INT64, false))));
        var badBounds = MetalMpsGraphProgram.Node.clamp(0, 1, DataType.INT64,
                9007199254740993L, 9007199254740992L);
        assertThrows(IllegalArgumentException.class, () -> MetalExactSupport.validate(badBounds,
                List.of(encoded(DataType.INT64, false), encoded(DataType.INT64, false))));
        for (long lower : new long[] {Long.MIN_VALUE, 0, 0x7ff0000000001234L}) {
            var clamp = MetalMpsGraphProgram.Node.clamp(0, 1, DataType.FLOAT64, lower, 0);
            assertDoesNotThrow(() -> MetalExactSupport.validate(clamp,
                    List.of(encoded(DataType.FLOAT64, false), encoded(DataType.FLOAT64, false))));
        }
    }

    @Test
    void encodedGradientRelationshipIsNotInheritedFromProviderAdmission() {
        var node = MetalMpsGraphProgram.Node.binary(MetalMpsGraphProgram.NodeKind.TENSOR_MAX, 0, 1, 2);
        assertDoesNotThrow(() -> MetalExactSupport.validate(node,
                List.of(encoded(DataType.FLOAT64, true, 2), encoded(DataType.FLOAT16, false, 2),
                        encoded(DataType.FLOAT64, true, 2))));
        assertThrows(IllegalArgumentException.class, () -> MetalExactSupport.validate(node,
                List.of(encoded(DataType.FLOAT64, true, 2), encoded(DataType.FLOAT16, false, 2),
                        encoded(DataType.FLOAT64, false, 2))));
    }

    @Test
    void exactComparisonsHaveNondifferentiableBoolResultsAndIndependentPromotionChecks() {
        var operation = new Operation(
                io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind.EQUAL,
                NoOperationAttrs.INSTANCE);
        var left = descriptor(DataType.FLOAT64, true, 2, 1);
        var right = descriptor(DataType.FLOAT16, true, 3);
        var output = descriptor(DataType.BOOL, false, 2, 3);
        assertTrue(MetalExactSupport.supports(operation, List.of(left, right), List.of(output)));
        var node = MetalExactSupport.lower(operation, new int[] {0, 1}, new int[] {2},
                List.of(left, right), List.of(output));
        assertDoesNotThrow(() -> MetalExactSupport.validate(node, List.of(
                encoded(DataType.FLOAT64, true, 2, 1), encoded(DataType.FLOAT16, true, 3),
                encoded(DataType.BOOL, false, 2, 3))));
        assertThrows(IllegalArgumentException.class, () -> MetalExactSupport.validate(node, List.of(
                encoded(DataType.FLOAT64, true, 2, 1), encoded(DataType.INT64, false, 3),
                encoded(DataType.BOOL, false, 2, 3))));
        assertThrows(IllegalArgumentException.class, () -> MetalExactSupport.validate(node, List.of(
                encoded(DataType.FLOAT64, true, 2, 1), encoded(DataType.FLOAT16, true, 3),
                encoded(DataType.FLOAT64, false, 2, 3))));
    }

    private static TensorDescriptor descriptor(DataType type, boolean grad, long... dims) {
        Shape shape = Shape.of(dims);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }

    private static MetalMpsGraphProgram.ValueDescriptor encoded(DataType type, boolean grad, long... dims) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dims, grad);
    }
}
