package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Independent exact provider predicates for the bounded FLOAT32-only SIGMOID occurrence. */
final class MetalSigmoidCapabilityTest {
    private final MetalCapabilityProvider provider = new MetalCapabilityProvider();
    private static final Operation SIGMOID = new Operation(
            UnaryElementwiseKind.SIGMOID, NoOperationAttrs.INSTANCE);

    @Test
    void admitsOnlyCanonicalPositiveStaticFloat32WithoutGradients() {
        for (Shape shape : List.of(Shape.of(1), Shape.of(2, 3), rank(16))) {
            TensorDescriptor value = canonical(DataType.FLOAT32, shape, false);
            assertTrue(supports(value, value), shape.toString());
        }
        Shape shape = Shape.of(2, 3);
        TensorDescriptor value = canonical(DataType.FLOAT32, shape, false);
        for (DataType wrong : List.of(DataType.BFLOAT16, DataType.FLOAT16,
                DataType.FLOAT64, DataType.INT32, DataType.INT64, DataType.BOOL)) {
            TensorDescriptor rejected = canonical(wrong, shape, false);
            assertFalse(supports(rejected, rejected), wrong.name());
            assertFalse(supports(value, rejected), wrong.name());
        }
        assertFalse(supports(canonical(DataType.FLOAT32, shape, true), value));
        assertFalse(supports(value, canonical(DataType.FLOAT32, shape, true)));
        assertFalse(supports(value, canonical(DataType.FLOAT32, Shape.of(3, 2), false)));
        assertFalse(supports(value, canonical(DataType.FLOAT32, Shape.of(3), false)));
        assertFalse(supports(canonical(DataType.FLOAT32, Shape.scalar(), false),
                canonical(DataType.FLOAT32, Shape.scalar(), false)));
        assertFalse(supports(canonical(DataType.FLOAT32, rank(17), false),
                canonical(DataType.FLOAT32, rank(17), false)));
        assertFalse(supports(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.empty(), false), value));
        TensorDescriptor view = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 2}, 0L, true)), false);
        assertFalse(supports(view, value));
        assertFalse(supports(value, view));
        TensorDescriptor huge = canonical(DataType.FLOAT32, Shape.of(1_073_741_824L), false);
        assertFalse(supports(huge, huge));
    }

    private boolean supports(TensorDescriptor input, TensorDescriptor output) {
        return provider.supports(new OperationCapabilityQuery(SIGMOID,
                List.of(input), List.of(output)));
    }

    private static TensorDescriptor canonical(DataType type, Shape shape, boolean grad) {
        return new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }

    private static Shape rank(int rank) {
        long[] dimensions = new long[rank];
        Arrays.fill(dimensions, 1L);
        return Shape.of(dimensions);
    }
}
