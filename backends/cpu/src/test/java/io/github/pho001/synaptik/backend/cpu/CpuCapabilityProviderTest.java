package io.github.pho001.synaptik.backend.cpu;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuCapabilityProviderTest {
    @Test void hasStableIdentityAndFailsClosedForEveryQuery() {
        var provider = new CpuCapabilityProvider();
        var descriptor = new TensorDescriptor(DataType.FLOAT32, Shape.of(0), Optional.empty(), false);
        var query = new OperationCapabilityQuery(
                new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE),
                List.of(descriptor), List.of(descriptor));
        var failure = assertThrows(NullPointerException.class, () -> provider.supports(null));
        assertAll(
                () -> assertEquals("cpu", CpuCapabilityProvider.CPU_BACKEND_ID.value()),
                () -> assertSame(CpuCapabilityProvider.CPU_BACKEND_ID, provider.backendId()),
                () -> assertSame(CpuCapabilityProvider.CPU_BACKEND_ID,
                        new CpuCapabilityProvider().backendId()),
                () -> assertFalse(provider.supports(query)),
                () -> assertEquals("query", failure.getMessage()));
    }
}
