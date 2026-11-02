package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalReductionSupportTest {
    @Test
    void exactAdmissionProvesCarrierShapeAndAffineSpanWithoutUnlockingFloat64Arithmetic() {
        var attrs = new MultiAxisReductionAttrs(List.of(1), false);
        for (DataType type : List.of(DataType.INT32, DataType.INT64, DataType.FLOAT64)) {
            Shape shape = Shape.of(2, 3);
            var input = new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.of(
                    shape, new long[] {8, 2}, 1, true)), false);
            for (var kind : List.of(AggregateReductionKind.MIN, AggregateReductionKind.MAX)) {
                assertTrue(MetalReductionSupport.supports(new Operation(kind, attrs), List.of(input),
                        List.of(descriptor(type, 2))));
                assertFalse(MetalReductionSupport.supports(new Operation(kind, attrs), List.of(input),
                        List.of(descriptor(type, 3))));
            }
            boolean integral = type != DataType.FLOAT64;
            assertTrue(integral == MetalReductionSupport.supports(
                    new Operation(AggregateReductionKind.SUM, attrs), List.of(input),
                    List.of(descriptor(type, 2))));
        }
    }

    @Test
    void rawValidationRejectsShapeAndGradientMismatchWithoutAProviderQuery() {
        var node = MetalMpsGraphProgram.Node.reduction(MetalMpsGraphProgram.NodeKind.SUM,
                0, 1, MetalMpsGraphProgram.ReductionForm.MULTI_AXIS, List.of(1), false);
        var input = new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {2, 3}, true);
        assertThrows(IllegalArgumentException.class, () -> MetalReductionSupport.validate(node,
                List.of(input, new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {2}, false))));
        assertThrows(IllegalArgumentException.class, () -> MetalReductionSupport.validate(node,
                List.of(input, new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {3}, true))));
        MetalReductionSupport.validate(node, List.of(input,
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {2}, true)));
    }

    @Test
    void numericalProjectionCannotAdmitLowOrFloat64Arithmetic() {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT16, DataType.BFLOAT16)) {
            for (var kind : List.of(MetalMpsGraphProgram.NodeKind.MEAN,
                    MetalMpsGraphProgram.NodeKind.PROD, MetalMpsGraphProgram.NodeKind.L2_NORM)) {
                var node = MetalMpsGraphProgram.Node.reduction(kind, 0, 1,
                        MetalMpsGraphProgram.ReductionForm.MULTI_AXIS, List.of(0), false);
                var values = List.of(new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {3}, false),
                        new MetalMpsGraphProgram.ValueDescriptor(type, new long[0], false));
                assertFalse(MetalReductionSupport.handles(node, values));
                assertThrows(IllegalArgumentException.class, () -> MetalReductionSupport.validate(node, values));
            }
        }
    }

    private static TensorDescriptor descriptor(DataType type, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }
}
