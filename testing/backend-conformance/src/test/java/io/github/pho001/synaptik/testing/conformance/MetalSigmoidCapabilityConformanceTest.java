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

/** Cross-module provider evidence for original-typed SIGMOID admission boundaries. */
final class MetalSigmoidCapabilityConformanceTest {
    private final MetalCapabilityProvider metal = new MetalCapabilityProvider();
    private static final Operation SIGMOID = new Operation(
            UnaryElementwiseKind.SIGMOID, NoOperationAttrs.INSTANCE);

    @Test
    void sigmoidRequiresMatchingOriginalTypeShapeAndGradientDescriptors() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            TensorDescriptor value = descriptor(type, Shape.of(2, 3), false);
            assertTrue(supports(value, value), type.name());
            for (DataType other : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
                if (other != type) {
                    assertFalse(supports(value,
                            descriptor(other, value.shape(), false)), type + "/" + other);
                }
            }
            assertFalse(supports(value,
                    descriptor(type, Shape.of(3, 2), false)), type.name());
            TensorDescriptor gradient = descriptor(type, value.shape(), true);
            assertTrue(supports(gradient, gradient), type.name());
            assertFalse(supports(gradient, value), type.name());
            assertFalse(supports(value, gradient), type.name());
            TensorDescriptor scalar = descriptor(type, Shape.scalar(), false);
            assertTrue(supports(scalar, scalar), type.name());
        }
        TensorDescriptor float64 = descriptor(DataType.FLOAT64, Shape.of(2, 3), false);
        assertFalse(supports(float64, float64));
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
