package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuAccessPlanTest {
    @Test void classifiesTheCompleteOrderedRegimeFamily() {
        assertEquals(CpuAccessPlan.Regime.DENSE_LINEAR,
                plan(Shape.of(2, 3), LayoutDescriptor.contiguous(Shape.of(2, 3)), Shape.of(2, 3)).regime());
        assertEquals(CpuAccessPlan.Regime.SCALAR_ALL_ZERO,
                plan(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar()), Shape.of(2, 3)).regime());
        assertEquals(CpuAccessPlan.Regime.SCALAR_ALL_ZERO,
                plan(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar()), Shape.scalar()).regime());
        assertEquals(CpuAccessPlan.Regime.LAST_AXIS_BIAS,
                plan(Shape.of(3), LayoutDescriptor.contiguous(Shape.of(3)), Shape.of(2, 3)).regime());
        Shape block = Shape.of(2, 1, 3);
        assertEquals(CpuAccessPlan.Regime.BLOCK_OUTER,
                plan(block, LayoutDescriptor.contiguous(block), Shape.of(2, 4, 3)).regime());
        Shape general = Shape.of(2, 3);
        assertEquals(CpuAccessPlan.Regime.GENERAL_ODOMETER,
                plan(general, LayoutDescriptor.of(general, new long[]{1, 2}, 0, true), general).regime());
    }

    @Test void retainsOffsetsStridesAndExactReferencedSpansButRejectsRepeatedWrites() {
        Shape shape = Shape.of(2, 3);
        var offset = LayoutDescriptor.of(shape, new long[]{3, 1}, 4, true);
        var analysis = analyze(descriptor(shape, offset), shape);
        assertAll(
                () -> assertEquals(4, analysis.plan().accessBindings().getFirst().baseElementOffset()),
                () -> assertEquals(offset.referencedElementSpan() * Double.BYTES,
                        analysis.plan().bufferDeclarations().getFirst().byteSize()));
        var repeated = descriptor(shape, LayoutDescriptor.of(shape, new long[]{1, 1}, 0, true));
        assertThrows(IllegalArgumentException.class, () -> CpuPartitionPreparerTest.analyze(
                descriptor(shape, LayoutDescriptor.contiguous(shape)),
                descriptor(shape, LayoutDescriptor.contiguous(shape)),
                descriptor(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar())), repeated,
                CpuPartitionAnalysisInputs.DEFAULT));
        var interleaved = descriptor(shape,
                LayoutDescriptor.of(shape, new long[]{3, 2}, 0, true));
        assertDoesNotThrow(() -> CpuPartitionPreparerTest.analyze(
                descriptor(shape, LayoutDescriptor.contiguous(shape)),
                descriptor(shape, LayoutDescriptor.contiguous(shape)),
                descriptor(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar())),
                interleaved, CpuPartitionAnalysisInputs.DEFAULT));
    }

    @Test void coldBindingStoresExactSubrangeSpanWithoutElementScanAtAliasTime() {
        var plan = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.GENERAL_ODOMETER, 2,
                java.util.List.of(CpuAccessPlan.AxisRole.STRIDED,
                        CpuAccessPlan.AxisRole.STRIDED), 0);
        var binding = CpuAccessPlan.Binding.create(plan, new long[]{2, 3}, 0,
                new long[]{3, 2}, 6, 1, 5, 8);
        assertAll(
                () -> assertEquals(2, binding.accessedElementStart()),
                () -> assertEquals(6, binding.accessedElementEnd()),
                () -> assertEquals(java.util.List.of(0L, 1L), binding.startCoordinates()),
                () -> assertEquals(2, binding.startAddress()));
    }

    @Test void storedSubrangeSpanMatchesExhaustiveSmallGeometryOracle() {
        long[] extents = {2, 3, 2};
        long[] strides = {3, 7, 2};
        var plan = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.GENERAL_ODOMETER, 3,
                java.util.List.of(CpuAccessPlan.AxisRole.STRIDED,
                        CpuAccessPlan.AxisRole.STRIDED, CpuAccessPlan.AxisRole.STRIDED), 0);
        for (long start = 0; start <= 12; start++) for (long end = start; end <= 12; end++) {
            var binding = CpuAccessPlan.Binding.create(plan, extents, 5, strides, 12,
                    start, end, 31);
            long minimum = 5, maximum = 5;
            if (start < end) {
                minimum = Long.MAX_VALUE; maximum = Long.MIN_VALUE;
                for (long index = start; index < end; index++) {
                    long remainder = index;
                    long address = 5;
                    for (int axis = extents.length - 1; axis >= 0; axis--) {
                        address += (remainder % extents[axis]) * strides[axis];
                        remainder /= extents[axis];
                    }
                    minimum = Math.min(minimum, address);
                    maximum = Math.max(maximum, address + 1);
                }
            }
            assertEquals(minimum, binding.accessedElementStart());
            assertEquals(maximum, binding.accessedElementEnd());
        }
    }

    @Test void outputInjectivityDecisionMatchesExhaustiveSmallStrideOracle() {
        Shape shape = Shape.of(2, 3);
        var dense = descriptor(shape, LayoutDescriptor.contiguous(shape));
        var scalar = descriptor(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar()));
        for (long outer = 0; outer <= 5; outer++) for (long inner = 0; inner <= 5; inner++) {
            long outerStride = outer, innerStride = inner;
            var addresses = new java.util.HashSet<Long>();
            for (long row = 0; row < 2; row++) for (long column = 0; column < 3; column++) {
                addresses.add(row * outer + column * inner);
            }
            boolean injective = addresses.size() == 6;
            var output = descriptor(shape,
                    LayoutDescriptor.of(shape, new long[]{outer, inner}, 0, true));
            if (injective) assertDoesNotThrow(() -> CpuPartitionPreparerTest.analyze(
                    dense, dense, scalar, output, CpuPartitionAnalysisInputs.DEFAULT),
                    () -> "expected injective strides " + outerStride + "," + innerStride);
            else assertThrows(IllegalArgumentException.class,
                    () -> CpuPartitionPreparerTest.analyze(dense, dense, scalar, output,
                            CpuPartitionAnalysisInputs.DEFAULT),
                    () -> "expected colliding strides " + outerStride + "," + innerStride);
        }
    }

    private static CpuAccessPlan plan(Shape source, LayoutDescriptor layout, Shape target) {
        return analyze(descriptor(source, layout), target).plan().units().getFirst().portablePlan()
                .kernelIr().values().getFirst().accessPlan();
    }

    private static io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis<
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan>
            analyze(TensorDescriptor first, Shape target) {
        var targetDescriptor = descriptor(target, LayoutDescriptor.contiguous(target));
        var scalar = descriptor(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar()));
        return CpuPartitionPreparerTest.analyze(first, targetDescriptor, scalar, targetDescriptor,
                CpuPartitionAnalysisInputs.DEFAULT);
    }

    private static TensorDescriptor descriptor(Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(DataType.FLOAT64, shape, Optional.of(layout), false);
    }
}
