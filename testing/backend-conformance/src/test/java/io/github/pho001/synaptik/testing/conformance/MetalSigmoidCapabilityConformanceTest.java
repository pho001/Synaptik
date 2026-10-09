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

/** Cross-module evidence for independent current F32 and homogeneous low SIGMOID answers. */
final class MetalSigmoidCapabilityConformanceTest {
    private final MetalCapabilityProvider metal = new MetalCapabilityProvider();
    private static final Operation SIGMOID = new Operation(
            UnaryElementwiseKind.SIGMOID, NoOperationAttrs.INSTANCE);

    @Test
    void eachQualifiedTypeAdmitsOnlyItsOwnNoGradientOccurrence() {
        TensorDescriptor float32 = descriptor(DataType.FLOAT32, Shape.of(2, 3), false);
        assertTrue(supports(float32, float32));
        for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            TensorDescriptor value = descriptor(low, Shape.of(2, 3), false);
            assertTrue(supports(value, value));
            assertFalse(supports(value, float32));
            assertFalse(supports(float32, value));
            assertFalse(supports(descriptor(low, Shape.of(2, 3), true), value));
        }
        for (DataType unsupported : List.of(DataType.FLOAT64)) {
            TensorDescriptor value = descriptor(unsupported, Shape.of(2, 3), false);
            assertFalse(supports(value, value), unsupported.name());
        }
        assertFalse(supports(descriptor(DataType.FLOAT32, Shape.of(2, 3), true),
                descriptor(DataType.FLOAT32, Shape.of(2, 3), true)));
        assertFalse(supports(float32,
                descriptor(DataType.FLOAT32, Shape.of(3, 2), false)));
        assertFalse(supports(descriptor(DataType.FLOAT32, Shape.scalar(), false),
                descriptor(DataType.FLOAT32, Shape.scalar(), false)));
    }

    private boolean supports(TensorDescriptor input, TensorDescriptor output) {
        return metal.supports(new OperationCapabilityQuery(SIGMOID,
                List.of(input), List.of(output)));
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, boolean grad) {
        return new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }
}
