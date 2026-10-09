package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider;
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

/** Cross-module provider evidence for independently qualified FLOAT32 and low EXP routes. */
final class MetalExpCapabilityConformanceTest {
    private final MetalCapabilityProvider metal = new MetalCapabilityProvider();
    private static final Operation EXP = new Operation(
            UnaryElementwiseKind.EXP, NoOperationAttrs.INSTANCE);

    @Test
    void currentProviderAdmitsOnlyCanonicalNoGradientTypedExp() {
        TensorDescriptor float32 = descriptor(DataType.FLOAT32, Shape.of(2, 3), false);
        assertTrue(supports(EXP, float32, float32));
        for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            TensorDescriptor lowValue = descriptor(low, Shape.of(2, 3), false);
            assertTrue(supports(EXP, lowValue, lowValue), low.name());
            assertFalse(supports(EXP, lowValue, float32), low.name());
            assertFalse(supports(EXP, descriptor(low, Shape.of(2, 3), true),
                    lowValue), low.name());
            assertFalse(supports(EXP, lowValue,
                    descriptor(low, Shape.of(3, 2), false)), low.name());
            assertFalse(supports(EXP, descriptor(low, Shape.scalar(), false),
                    descriptor(low, Shape.scalar(), false)), low.name());
        }
        assertFalse(supports(EXP, descriptor(DataType.BFLOAT16, Shape.of(2, 3), false),
                descriptor(DataType.FLOAT16, Shape.of(2, 3), false)));
        TensorDescriptor float64 = descriptor(DataType.FLOAT64, Shape.of(2, 3), false);
        assertFalse(supports(EXP, float64, float64));
        assertFalse(supports(EXP, descriptor(DataType.FLOAT32, Shape.of(2, 3), true),
                descriptor(DataType.FLOAT32, Shape.of(2, 3), true)));
        assertFalse(supports(EXP, float32,
                descriptor(DataType.FLOAT32, Shape.of(3, 2), false)));
        assertFalse(supports(EXP, descriptor(DataType.FLOAT32, Shape.scalar(), false),
                descriptor(DataType.FLOAT32, Shape.scalar(), false)));
        assertFalse(supports(new Operation(UnaryElementwiseKind.SIGMOID,
                NoOperationAttrs.INSTANCE), float32, float32));
    }

    private boolean supports(Operation operation, TensorDescriptor input,
            TensorDescriptor output) {
        return metal.supports(new OperationCapabilityQuery(
                operation, List.of(input), List.of(output)));
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, boolean grad) {
        return new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }
}
