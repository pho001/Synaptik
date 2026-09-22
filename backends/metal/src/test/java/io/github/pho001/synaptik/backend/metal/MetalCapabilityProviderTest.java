package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalCapabilityProviderTest {
    private final MetalCapabilityProvider provider = new MetalCapabilityProvider();

    @Test
    void supportsExactlyPositiveStaticContiguousFloat32NegOccurrences() {
        assertSame(MetalCapabilityProvider.METAL_BACKEND_ID, provider.backendId());
        assertEquals("metal", provider.backendId().value());
        assertTrue(provider.supports(query(descriptor(Shape.of(2, 3), false),
                descriptor(Shape.of(2, 3), false))));
        Shape rank16 = Shape.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assertTrue(provider.supports(query(descriptor(rank16, true), descriptor(rank16, true))));
    }

    @Test
    void rejectsEveryBoundaryOutsideTheExactDomain() {
        TensorDescriptor valid = descriptor(Shape.of(2, 3), false);
        Shape dynamic = Shape.ofDimensions(new DynamicDimension("N"));
        Shape rank17 = Shape.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        var noLayout = new TensorDescriptor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
        var offset = new TensorDescriptor(DataType.FLOAT32, Shape.of(2, 3),
                Optional.of(LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 1, true)), false);
        var view = new TensorDescriptor(DataType.FLOAT32, Shape.of(2, 3),
                Optional.of(LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 0, true)), false);

        assertFalse(provider.supports(new OperationCapabilityQuery(
                new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE),
                List.of(valid), List.of(valid))));
        assertFalse(provider.supports(query(typed(DataType.FLOAT64), typed(DataType.FLOAT64))));
        assertFalse(provider.supports(query(valid, descriptor(Shape.of(3, 2), false))));
        assertFalse(provider.supports(query(valid, descriptor(Shape.of(2, 3), true))));
        assertFalse(provider.supports(query(
                new TensorDescriptor(DataType.FLOAT32, dynamic, Optional.empty(), false),
                new TensorDescriptor(DataType.FLOAT32, dynamic, Optional.empty(), false))));
        assertFalse(provider.supports(query(descriptor(Shape.of(), false), descriptor(Shape.of(), false))));
        assertFalse(provider.supports(query(descriptor(Shape.of(2, 0), false),
                descriptor(Shape.of(2, 0), false))));
        assertFalse(provider.supports(query(descriptor(rank17, false), descriptor(rank17, false))));
        assertFalse(provider.supports(query(noLayout, noLayout)));
        assertFalse(provider.supports(query(offset, offset)));
        assertFalse(provider.supports(query(view, view)));
    }

    @Test
    void rejectsNullWithTheContractMessage() {
        var failure = assertThrows(NullPointerException.class, () -> provider.supports(null));
        assertEquals("query", failure.getMessage());
    }

    private static OperationCapabilityQuery query(TensorDescriptor input, TensorDescriptor output) {
        return new OperationCapabilityQuery(neg(), List.of(input), List.of(output));
    }

    private static Operation neg() {
        return new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE);
    }

    private static TensorDescriptor descriptor(Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
    }

    private static TensorDescriptor typed(DataType type) {
        Shape shape = Shape.of(2, 3);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)),
                type.isDifferentiable());
    }
}
