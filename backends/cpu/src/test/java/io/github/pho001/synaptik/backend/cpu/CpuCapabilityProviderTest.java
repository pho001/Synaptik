package io.github.pho001.synaptik.backend.cpu;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuCapabilityProviderTest {
    private final CpuCapabilityProvider provider = new CpuCapabilityProvider();

    @Test void hasStableIdentityAndAdvertisesOnlyExactDenseAddMatrix() {
        assertAll(
                () -> assertEquals("cpu", CpuCapabilityProvider.CPU_BACKEND_ID.value()),
                () -> assertSame(CpuCapabilityProvider.CPU_BACKEND_ID, provider.backendId()),
                () -> assertSame(CpuCapabilityProvider.CPU_BACKEND_ID,
                        new CpuCapabilityProvider().backendId()),
                () -> assertEquals("query", assertThrows(NullPointerException.class,
                        () -> provider.supports(null)).getMessage()));

        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32,
                DataType.INT32, DataType.INT64)) {
            assertTrue(provider.supports(add(descriptor(type, Shape.scalar(), Optional.empty()),
                    descriptor(type, Shape.scalar(), Optional.empty()),
                    descriptor(type, Shape.scalar(), Optional.empty()))));
            Shape shape = Shape.of(2, 0, 3);
            var resolved = descriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)));
            assertTrue(provider.supports(add(resolved, resolved, resolved)));
        }

        var dense = descriptor(DataType.FLOAT32, Shape.of(4), Optional.empty());
        assertFalse(provider.supports(new OperationCapabilityQuery(
                new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE),
                List.of(dense), List.of(dense))));
        assertFalse(provider.supports(add(descriptor(DataType.BFLOAT16, Shape.of(4), Optional.empty()),
                descriptor(DataType.BFLOAT16, Shape.of(4), Optional.empty()),
                descriptor(DataType.BFLOAT16, Shape.of(4), Optional.empty()))));
        assertFalse(provider.supports(add(dense,
                descriptor(DataType.INT32, Shape.of(4), Optional.empty()), dense)));
        assertFalse(provider.supports(add(dense,
                descriptor(DataType.FLOAT32, Shape.of(1), Optional.empty()), dense)));
        var dynamic = descriptor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("n")), Optional.empty());
        assertFalse(provider.supports(add(dynamic, dynamic, dynamic)));

        Shape shape = Shape.of(4);
        var offset = descriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[]{1}, 1, true)));
        var strided = descriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[]{2}, 0, true)));
        var viewedDense = descriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[]{1}, 0, true)));
        assertAll(() -> assertFalse(provider.supports(add(offset, dense, dense))),
                () -> assertFalse(provider.supports(add(dense, strided, dense))),
                () -> assertFalse(provider.supports(add(dense, dense, viewedDense))));
    }

    private static OperationCapabilityQuery add(
            TensorDescriptor left, TensorDescriptor right, TensorDescriptor output) {
        return new OperationCapabilityQuery(
                new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                List.of(left, right), List.of(output));
    }

    private static TensorDescriptor descriptor(
            DataType type, Shape shape, Optional<LayoutDescriptor> layout) {
        return new TensorDescriptor(type, shape, layout, false);
    }
}
