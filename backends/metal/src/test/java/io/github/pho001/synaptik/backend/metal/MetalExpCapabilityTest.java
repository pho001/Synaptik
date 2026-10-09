package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Exact independent provider predicates for bounded FLOAT32 and low-precision EXP. */
final class MetalExpCapabilityTest {
    private final MetalCapabilityProvider provider = new MetalCapabilityProvider();
    private static final Operation EXP = new Operation(
            UnaryElementwiseKind.EXP, NoOperationAttrs.INSTANCE);

    @Test
    void admitsOnlyPositiveStaticCanonicalFloat32WithoutGradients() {
        for (Shape shape : List.of(Shape.of(1), Shape.of(2, 3), rank(16))) {
            TensorDescriptor value = canonical(DataType.FLOAT32, shape, false);
            assertTrue(supports(EXP, List.of(value), List.of(value)), shape.toString());
        }
        Shape shape = Shape.of(2, 3);
        TensorDescriptor value = canonical(DataType.FLOAT32, shape, false);
        for (DataType type : List.of(
                DataType.FLOAT64,
                DataType.INT32, DataType.INT64, DataType.BOOL)) {
            TensorDescriptor wrong = canonical(type, shape, false);
            assertFalse(supports(EXP, List.of(wrong), List.of(wrong)), type.name());
            assertFalse(supports(EXP, List.of(value), List.of(wrong)), type.name());
        }
        assertFalse(supports(EXP, List.of(canonical(DataType.FLOAT32, shape, true)),
                List.of(value)));
        assertFalse(supports(EXP, List.of(value),
                List.of(canonical(DataType.FLOAT32, shape, true))));
        assertFalse(supports(EXP, List.of(value),
                List.of(canonical(DataType.FLOAT32, Shape.of(3, 2), false))));
        assertFalse(supports(EXP, List.of(value),
                List.of(canonical(DataType.FLOAT32, Shape.of(3), false))));
        assertThrows(IllegalArgumentException.class,
                () -> supports(EXP, List.of(), List.of(value)));
        assertThrows(IllegalArgumentException.class,
                () -> supports(EXP, List.of(value, value), List.of(value)));
        assertThrows(IllegalArgumentException.class,
                () -> supports(EXP, List.of(value), List.of()));
        assertThrows(IllegalArgumentException.class,
                () -> supports(new Operation(UnaryElementwiseKind.EXP,
                                new ScalarValueAttrs(ScalarValue.float32(1.0f))),
                        List.of(value), List.of(value)));
        assertFalse(supports(new Operation(UnaryElementwiseKind.SIGMOID,
                        NoOperationAttrs.INSTANCE),
                List.of(value), List.of(value)));
        TensorDescriptor scalar = canonical(DataType.FLOAT32, Shape.scalar(), false);
        assertFalse(supports(EXP, List.of(scalar), List.of(scalar)));
        Shape rank17 = rank(17);
        assertFalse(supports(EXP,
                List.of(canonical(DataType.FLOAT32, rank17, false)),
                List.of(canonical(DataType.FLOAT32, rank17, false))));
        TensorDescriptor absentLayout = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), false);
        assertFalse(supports(EXP, List.of(absentLayout), List.of(value)));
        TensorDescriptor view = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 2}, 0L, true)), false);
        assertFalse(supports(EXP, List.of(view), List.of(value)));
        assertFalse(supports(EXP, List.of(value), List.of(view)));
        Shape tooLarge = Shape.of(1_073_741_824L);
        TensorDescriptor oversized = canonical(DataType.FLOAT32, tooLarge, false);
        assertFalse(supports(EXP, List.of(oversized), List.of(oversized)));
    }

    @Test
    void lowExpHasIndependentSameTypeCanonicalNoGradientPredicates() {
        for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            for (Shape shape : List.of(Shape.of(1), Shape.of(2, 3), rank(16))) {
                TensorDescriptor value = canonical(low, shape, false);
                assertTrue(supports(EXP, List.of(value), List.of(value)), low + " " + shape);
            }
            Shape shape = Shape.of(2, 3);
            TensorDescriptor value = canonical(low, shape, false);
            DataType other = low == DataType.BFLOAT16
                    ? DataType.FLOAT16 : DataType.BFLOAT16;
            assertFalse(supports(EXP, List.of(value),
                    List.of(canonical(other, shape, false))));
            assertFalse(supports(EXP, List.of(value),
                    List.of(canonical(DataType.FLOAT32, shape, false))));
            assertFalse(supports(EXP,
                    List.of(canonical(low, shape, true)), List.of(value)));
            assertFalse(supports(EXP,
                    List.of(value), List.of(canonical(low, shape, true))));
            assertFalse(supports(EXP, List.of(value),
                    List.of(canonical(low, Shape.of(3, 2), false))));
            assertFalse(supports(EXP, List.of(value),
                    List.of(canonical(low, Shape.of(3), false))));
            TensorDescriptor scalar = canonical(low, Shape.scalar(), false);
            assertFalse(supports(EXP, List.of(scalar), List.of(scalar)));
            TensorDescriptor rank17 = canonical(low, rank(17), false);
            assertFalse(supports(EXP, List.of(rank17), List.of(rank17)));
            TensorDescriptor noLayout = new TensorDescriptor(
                    low, shape, Optional.empty(), false);
            assertFalse(supports(EXP, List.of(noLayout), List.of(value)));
            TensorDescriptor view = new TensorDescriptor(low, shape,
                    Optional.of(LayoutDescriptor.of(shape, new long[] {1, 2}, 0L, true)), false);
            assertFalse(supports(EXP, List.of(view), List.of(value)));
            assertFalse(supports(EXP, List.of(value), List.of(view)));
            TensorDescriptor oversized = canonical(low, Shape.of(2_147_483_648L), false);
            assertFalse(supports(EXP, List.of(oversized), List.of(oversized)));
            assertThrows(IllegalArgumentException.class,
                    () -> supports(new Operation(UnaryElementwiseKind.EXP,
                            new ScalarValueAttrs(ScalarValue.float32(1.0f))),
                            List.of(value), List.of(value)));
        }
    }

    private boolean supports(Operation operation, List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        return provider.supports(new OperationCapabilityQuery(operation, inputs, outputs));
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
