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

/** Cross-module provider evidence for original-typed EXP admission boundaries. */
final class MetalExpCapabilityConformanceTest {
    private final MetalCapabilityProvider metal = new MetalCapabilityProvider();
    private static final Operation EXP = new Operation(
            UnaryElementwiseKind.EXP, NoOperationAttrs.INSTANCE);

    @Test
    void expRequiresMatchingOriginalTypeShapeAndGradientDescriptors() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            TensorDescriptor value = descriptor(type, Shape.of(2, 3), false);
            assertTrue(supports(EXP, value, value), type.name());
            for (DataType other : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
                if (other != type) {
                    assertFalse(supports(EXP, value,
                            descriptor(other, value.shape(), false)), type + "/" + other);
                }
            }
            assertFalse(supports(EXP, value,
                    descriptor(type, Shape.of(3, 2), false)), type.name());
            TensorDescriptor gradient = descriptor(type, value.shape(), true);
            assertTrue(supports(EXP, gradient, gradient), type.name());
            assertFalse(supports(EXP, gradient, value), type.name());
            assertFalse(supports(EXP, value, gradient), type.name());
            TensorDescriptor scalar = descriptor(type, Shape.scalar(), false);
            assertTrue(supports(EXP, scalar, scalar), type.name());
        }
        TensorDescriptor float64 = descriptor(DataType.FLOAT64, Shape.of(2, 3), false);
        assertFalse(supports(EXP, float64, float64));
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
