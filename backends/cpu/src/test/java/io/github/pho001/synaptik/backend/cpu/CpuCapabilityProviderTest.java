package io.github.pho001.synaptik.backend.cpu;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuCapabilityProviderTest {
    @Test void reportsOnlyExactResolvedFloat64ProvingOccurrences() {
        var provider = new CpuCapabilityProvider();
        Shape shape = Shape.of(2, 0, 3);
        var dense = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        assertAll(
                () -> assertSame(CpuCapabilityProvider.CPU_BACKEND_ID, provider.backendId()),
                () -> assertTrue(provider.supports(query(BinaryArithmeticKind.ADD,
                        List.of(dense, dense), dense))),
                () -> assertTrue(provider.supports(query(UnaryElementwiseKind.GELU,
                        List.of(dense), dense))),
                () -> assertTrue(provider.supports(query(BinaryArithmeticKind.MUL,
                        List.of(dense, dense), dense))),
                () -> assertFalse(provider.supports(query(UnaryElementwiseKind.GELU_TANH_APPROXIMATION,
                        List.of(dense), dense))));
        var unresolved = new TensorDescriptor(DataType.FLOAT64, shape, Optional.empty(), false);
        var float32 = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        assertAll(
                () -> assertFalse(provider.supports(query(BinaryArithmeticKind.ADD,
                        List.of(unresolved, unresolved), unresolved))),
                () -> assertFalse(provider.supports(query(BinaryArithmeticKind.ADD,
                        List.of(float32, float32), float32))),
                () -> assertEquals("query", assertThrows(NullPointerException.class,
                        () -> provider.supports(null)).getMessage()));
    }

    @Test void reportsStaticRightBroadcastAndResolvedStridedOccurrencesTruthfully() {
        var provider = new CpuCapabilityProvider();
        Shape leftShape = Shape.of(2, 1, 3), rightShape = Shape.of(4, 3);
        Shape outputShape = Shape.of(2, 4, 3);
        var left = descriptor(leftShape, LayoutDescriptor.of(leftShape,
                new long[]{7, 0, 2}, 3, true));
        var right = descriptor(rightShape, LayoutDescriptor.contiguous(rightShape));
        var output = descriptor(outputShape, LayoutDescriptor.contiguous(outputShape));
        var wrong = descriptor(Shape.of(2, 3), LayoutDescriptor.contiguous(Shape.of(2, 3)));
        assertAll(
                () -> assertTrue(provider.supports(query(BinaryArithmeticKind.ADD,
                        List.of(left, right), output))),
                () -> assertFalse(provider.supports(query(BinaryArithmeticKind.ADD,
                        List.of(left, right), wrong))),
                () -> assertFalse(provider.supports(query(UnaryElementwiseKind.GELU,
                        List.of(left), output))));
    }

    private static TensorDescriptor descriptor(Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(DataType.FLOAT64, shape, Optional.of(layout), false);
    }

    private static OperationCapabilityQuery query(Object kind, List<TensorDescriptor> inputs,
            TensorDescriptor output) {
        return new OperationCapabilityQuery(new Operation(
                (io.github.pho001.synaptik.model.operation.OperationKind) kind,
                NoOperationAttrs.INSTANCE), inputs, List.of(output));
    }
}
