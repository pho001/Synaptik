package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalCapabilityProviderTest {
    @Test
    void remainsFailClosedForEveryCurrentDataType() {
        var provider = new MetalCapabilityProvider();

        assertSame(MetalCapabilityProvider.METAL_BACKEND_ID, provider.backendId());
        assertEquals("metal", provider.backendId().value());
        for (DataType type : DataType.values()) {
            Shape shape = Shape.of(2, 3);
            var descriptor = new TensorDescriptor(type, shape,
                    Optional.of(LayoutDescriptor.contiguous(shape)), type.isDifferentiable());
            var query = new OperationCapabilityQuery(
                    new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                    List.of(descriptor), List.of(descriptor));
            assertFalse(provider.supports(query), type.toString());
        }
    }

    @Test
    void rejectsNullWithTheContractMessage() {
        var failure = assertThrows(
                NullPointerException.class, () -> new MetalCapabilityProvider().supports(null));
        assertEquals("query", failure.getMessage());
    }
}
