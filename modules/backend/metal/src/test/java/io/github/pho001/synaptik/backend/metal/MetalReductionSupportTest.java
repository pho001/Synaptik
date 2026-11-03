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
    void numericProjectionAdmitsAllThreeOriginalCarriersWithoutFloat64Arithmetic() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16)) {
            for (var kind : List.of(MetalMpsGraphProgram.NodeKind.SUM, MetalMpsGraphProgram.NodeKind.MEAN,
                    MetalMpsGraphProgram.NodeKind.PROD, MetalMpsGraphProgram.NodeKind.L1_NORM,
                    MetalMpsGraphProgram.NodeKind.L2_NORM, MetalMpsGraphProgram.NodeKind.LOG_SUM_EXP,
                    MetalMpsGraphProgram.NodeKind.REDUCTION_MIN, MetalMpsGraphProgram.NodeKind.REDUCTION_MAX)) {
                for (boolean grad : List.of(false, true)) {
                    var node = MetalMpsGraphProgram.Node.reduction(kind, 0, 1,
                            MetalMpsGraphProgram.ReductionForm.MULTI_AXIS, List.of(0), false);
                    var values = List.of(new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {3}, grad),
                            new MetalMpsGraphProgram.ValueDescriptor(type, new long[0], grad));
                    assertTrue(MetalReductionSupport.handles(node, values));
                    MetalReductionSupport.validate(node, values);
                }
            }
        }
        var mean = MetalMpsGraphProgram.Node.reduction(MetalMpsGraphProgram.NodeKind.MEAN, 0, 1,
                MetalMpsGraphProgram.ReductionForm.MULTI_AXIS, List.of(0), false);
        var doubles = List.of(new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT64, new long[] {3}, false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT64, new long[0], false));
        assertFalse(MetalReductionSupport.handles(mean, doubles));
        assertThrows(IllegalArgumentException.class, () -> MetalReductionSupport.validate(mean, doubles));
    }

    @Test
    void scanEmptyReductionProofRetainsRealAxesAndOrdinaryBoundary() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            var input = new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {0, 3}, true);
            var axisOutput = new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {3}, true);
            for (var kind : List.of(MetalMpsGraphProgram.NodeKind.SUM, MetalMpsGraphProgram.NodeKind.MEAN)) {
                var axis = MetalMpsGraphProgram.Node.reduction(kind, 0, 1,
                        MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS, List.of(0), false);
                var values = List.of(input, axisOutput);
                MetalReductionSupport.validateScan(axis, values);
                assertThrows(IllegalArgumentException.class, () -> MetalReductionSupport.validate(axis, values));
                var invalidAxis = MetalMpsGraphProgram.Node.reduction(kind, 0, 1,
                        MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS, List.of(2), false);
                assertThrows(IllegalArgumentException.class, () -> MetalReductionSupport.validateScan(invalidAxis, values));
                var full = MetalMpsGraphProgram.Node.reduction(kind, 0, 1,
                        MetalMpsGraphProgram.ReductionForm.FULL, List.of(), false);
                MetalReductionSupport.validateScan(full, List.of(input,
                        new MetalMpsGraphProgram.ValueDescriptor(type, new long[0], true)));
            }
            var target = new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {1, 3}, true);
            MetalReductionSupport.validateScan(MetalMpsGraphProgram.Node.sumToShape(0, 1, new long[] {1, 3}),
                    List.of(input, target));
            assertThrows(IllegalArgumentException.class, () -> MetalReductionSupport.validateScan(
                    MetalMpsGraphProgram.Node.sumToShape(0, 1, new long[] {1, 2}), List.of(input, target)));
        }
    }

    @Test
    void emptyStatisticalOutputStillRequiresPositiveActualCorrectedContributorCount() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            var input = new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {0, 3}, true);
            for (var kind : List.of(MetalMpsGraphProgram.NodeKind.VARIANCE,
                    MetalMpsGraphProgram.NodeKind.STANDARD_DEVIATION)) {
                var output = new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {0}, true);
                MetalReductionSupport.validateScan(MetalMpsGraphProgram.Node.statisticalReduction(
                        kind, 0, 1, List.of(1), false, 2), List.of(input, output));
                assertThrows(IllegalArgumentException.class, () -> MetalReductionSupport.validateScan(
                        MetalMpsGraphProgram.Node.statisticalReduction(kind, 0, 1, List.of(1), false, 3),
                        List.of(input, output)));
                var nonemptyOutput = new MetalMpsGraphProgram.ValueDescriptor(
                        type, new long[] {3}, true);
                assertThrows(IllegalArgumentException.class, () -> MetalReductionSupport.validateScan(
                        MetalMpsGraphProgram.Node.statisticalReduction(kind, 0, 1, List.of(0), false, 0),
                        List.of(input, nonemptyOutput)));
            }
        }
    }

    @Test
    void allOwnedFloatingReducersPreserveEmptyRowsAndAllowedSelectedEmptyDomains() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            var input = new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {0, 3}, true);
            for (var kind : List.of(MetalMpsGraphProgram.NodeKind.SUM, MetalMpsGraphProgram.NodeKind.MEAN,
                    MetalMpsGraphProgram.NodeKind.PROD, MetalMpsGraphProgram.NodeKind.LOG_SUM_EXP,
                    MetalMpsGraphProgram.NodeKind.L1_NORM, MetalMpsGraphProgram.NodeKind.L2_NORM,
                    MetalMpsGraphProgram.NodeKind.REDUCTION_MIN, MetalMpsGraphProgram.NodeKind.REDUCTION_MAX)) {
                for (int axis : new int[] {0, 1}) {
                    long[] resultShape = axis == 0 ? new long[] {3} : new long[] {0};
                    var output = new MetalMpsGraphProgram.ValueDescriptor(type, resultShape, true);
                    var node = MetalMpsGraphProgram.Node.reduction(kind, 0, 1,
                            MetalMpsGraphProgram.ReductionForm.MULTI_AXIS, List.of(axis), false);
                    var values = List.of(input, output);
                    MetalReductionSupport.validateScan(node, values);
                    assertThrows(IllegalArgumentException.class, () -> MetalReductionSupport.validate(node, values));
                    var badAxis = MetalMpsGraphProgram.Node.reduction(kind, 0, 1,
                            MetalMpsGraphProgram.ReductionForm.MULTI_AXIS, List.of(2), false);
                    assertThrows(IllegalArgumentException.class, () -> MetalReductionSupport.validateScan(badAxis, values));
                }
            }
        }
    }

    @Test
    void broadcastReadsRequireTruthfulViewAndInjectiveNonbroadcastAxes() {
        Shape shape = Shape.of(2, 2, 2);
        var operation = new Operation(AggregateReductionKind.SUM,
                new MultiAxisReductionAttrs(List.of(2), false));
        var raw = MetalMpsGraphProgram.Node.reduction(MetalMpsGraphProgram.NodeKind.SUM, 0, 1,
                MetalMpsGraphProgram.ReductionForm.MULTI_AXIS, List.of(2), false);
        var output = new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {2, 2}, false);
        for (int fixture = 0; fixture < 3; fixture++) {
            long[] strides = fixture == 0 ? new long[] {0, 1, 1} : new long[] {0, 2, 1};
            boolean view = fixture != 1;
            if (fixture == 1) {
                assertThrows(IllegalArgumentException.class,
                        () -> LayoutDescriptor.of(shape, strides, 0, view));
                continue;
            }
            LayoutDescriptor layout = LayoutDescriptor.of(shape, strides, 0, view);
            boolean valid = fixture == 2;
            var input = new TensorDescriptor(DataType.INT64, shape, Optional.of(layout), false);
            assertTrue(valid == MetalReductionSupport.supports(operation, List.of(input),
                    List.of(descriptor(DataType.INT64, 2, 2))));
            var encoded = new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32,
                    shape.toLongArray(), Optional.of(layout), false, false);
            if (valid) MetalReductionSupport.validate(raw, List.of(encoded, output));
            else assertThrows(IllegalArgumentException.class,
                    () -> MetalReductionSupport.validate(raw, List.of(encoded, output)));
        }
    }

    private static TensorDescriptor descriptor(DataType type, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }
}
