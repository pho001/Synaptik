package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;
import org.junit.jupiter.api.Test;

public class CpuNonAffineMovementLoweringTest {
    @Test void lowersSignedAndTargetRelativeSliceUpdateWithDeduplicatedBoundaries() {
        var signed = lower(context(new Operation(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(4L), List.of(2L), List.of(0), List.of(-2L))),
                List.of(0, 1), List.of(descriptor(DataType.INT32, Shape.of(5)),
                        descriptor(DataType.INT32, Shape.of(2))),
                descriptor(DataType.INT32, Shape.of(5))));
        var crop = lower(context(new Operation(SliceKind.SLICE_UPDATE,
                        new CropToShapeAttrs(Shape.of(2, 2), Shape.of(0, 1))),
                List.of(0, 1), List.of(descriptor(DataType.FLOAT32, Shape.of(2, 4)),
                        descriptor(DataType.FLOAT32, Shape.of(2, 2))),
                descriptor(DataType.FLOAT32, Shape.of(2, 4))));
        var same = lower(context(new Operation(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(), List.of(), List.of(), List.of())),
                List.of(0, 0), List.of(descriptor(DataType.BOOL, Shape.of(2))),
                descriptor(DataType.BOOL, Shape.of(2))));
        var signedGeometry = assertInstanceOf(CpuNonAffineMovementLowering.Geometry.SliceUpdate.class,
                signed.movementGeometry().orElseThrow().variant());
        var cropGeometry = assertInstanceOf(CpuNonAffineMovementLowering.Geometry.SliceUpdate.class,
                crop.movementGeometry().orElseThrow().variant());
        assertAll(
                () -> assertArrayEquals(new long[]{4}, signedGeometry.starts()),
                () -> assertArrayEquals(new long[]{-2}, signedGeometry.steps()),
                () -> assertArrayEquals(new long[]{0, 1}, cropGeometry.starts()),
                () -> assertEquals(List.of(0, 1), ((CpuDataMovementIr) signed.portableKernelIr())
                        .plan().occurrenceToBoundary()),
                () -> assertEquals(List.of(0, 0), ((CpuDataMovementIr) same.portableKernelIr())
                        .plan().occurrenceToBoundary()),
                () -> assertEquals(2, same.boundaryValues().size()));
    }

    @Test void sliceUpdateLoweringAcceptsExactBoundsAndRejectsInvalidBoundsAndOutputLayout() {
        var base = descriptor(DataType.INT32, Shape.of(5));
        var update = descriptor(DataType.INT32, Shape.of(2));
        var output = descriptor(DataType.INT32, Shape.of(5));
        var positiveBoundary = context(new Operation(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(0L), List.of(2L), List.of(0), List.of(4L))),
                List.of(0, 1), List.of(base, update), output);
        var negativeBoundary = context(new Operation(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(4L), List.of(2L), List.of(0), List.of(-4L))),
                List.of(0, 1), List.of(base, update), output);
        var invalidPositive = context(new Operation(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(1L), List.of(2L), List.of(0), List.of(4L))),
                List.of(0, 1), List.of(base, update), output);
        var invalidCrop = context(new Operation(SliceKind.SLICE_UPDATE,
                        new CropToShapeAttrs(Shape.of(2), Shape.of(4))),
                List.of(0, 1), List.of(base, update), output);
        var repeatedOutput = context(new Operation(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(0L), List.of(2L), List.of(0), List.of(1L))),
                List.of(0, 1), List.of(base, update), descriptor(DataType.INT32, Shape.of(5),
                        LayoutDescriptor.of(Shape.of(5), new long[]{0}, 0, true)));
        assertAll(
                () -> assertDoesNotThrow(() -> lower(positiveBoundary)),
                () -> assertDoesNotThrow(() -> lower(negativeBoundary)),
                () -> assertThrows(IllegalArgumentException.class, () -> lower(invalidPositive)),
                () -> assertThrows(IllegalArgumentException.class, () -> lower(invalidCrop)),
                () -> assertThrows(IllegalArgumentException.class, () -> lower(repeatedOutput)));
    }

    @Test void lowersWindowFamiliesWithUnequalRanksAndExactPaddingBits() {
        var axis = lower(context(new Operation(WindowTransformKind.UNFOLD_AXIS,
                        new UnfoldAxisAttrs(1, 2, 1)), List.of(0),
                List.of(descriptor(DataType.BOOL, Shape.of(2, 3))),
                descriptor(DataType.BOOL, Shape.of(2, 2, 2))));
        var window = new Window2dAttrs(2, 2, 1, 1, 1, 1, 1, 1, false);
        var image = lower(context(new Operation(WindowTransformKind.UNFOLD2D,
                        new Unfold2dAttrs(window,
                                ScalarValue.float64(Double.longBitsToDouble(0xfff8_0000_0000_0042L)))),
                List.of(0), List.of(descriptor(DataType.FLOAT64, Shape.of(1, 1, 3, 3))),
                descriptor(DataType.FLOAT64, Shape.of(1, 4, 16))));
        var imagePlan = assertInstanceOf(CpuDataMovementIr.Unfold2dPlan.class,
                ((CpuDataMovementIr) image.portableKernelIr()).plan());
        assertAll(
                () -> assertInstanceOf(CpuDataMovementIr.UnfoldAxisPlan.class,
                        ((CpuDataMovementIr) axis.portableKernelIr()).plan()),
                () -> assertEquals(0xfff8_0000_0000_0042L, imagePlan.immediateBits()),
                () -> assertEquals(4, image.movementGeometry().orElseThrow()
                        .inputs().getFirst().extents().length),
                () -> assertEquals(3, image.movementGeometry().orElseThrow().outputExtents().length));
    }

    @Test void lowersEveryFamilyAndDeduplicatesRepeatedCompositionInputs() {
        var pad = lower(context(new Operation(PadKind.PAD,
                new PadAttrs(List.of(1L), List.of(2L), ScalarValue.int32(-7))),
                List.of(0), List.of(descriptor(DataType.INT32, Shape.of(2))),
                descriptor(DataType.INT32, Shape.of(5))));
        var tile = lower(context(new Operation(TileKind.TILE, new TileAttrs(List.of(3L))),
                List.of(0), List.of(descriptor(DataType.INT32, Shape.of(2))),
                descriptor(DataType.INT32, Shape.of(6))));
        var concat = lower(context(new Operation(TensorCompositionKind.CONCAT,
                        new CompositionAxisAttrs(0)), List.of(0, 1, 0),
                List.of(descriptor(DataType.INT32, Shape.of(2)),
                        descriptor(DataType.INT32, Shape.of(1))),
                descriptor(DataType.INT32, Shape.of(5))));
        var stack = lower(context(new Operation(TensorCompositionKind.STACK,
                        new CompositionAxisAttrs(1)), List.of(0, 1),
                List.of(descriptor(DataType.INT32, Shape.of(2)),
                        descriptor(DataType.INT32, Shape.of(2))),
                descriptor(DataType.INT32, Shape.of(2, 2))));
        assertAll(
                () -> assertInstanceOf(CpuDataMovementIr.PadPlan.class,
                        ((CpuDataMovementIr) pad.portableKernelIr()).plan()),
                () -> assertInstanceOf(CpuDataMovementIr.TilePlan.class,
                        ((CpuDataMovementIr) tile.portableKernelIr()).plan()),
                () -> assertEquals(3, concat.boundaryValues().size()),
                () -> assertEquals(List.of(0, 1, 0),
                        ((CpuDataMovementIr) concat.portableKernelIr()).plan()
                                .occurrenceToBoundary()),
                () -> assertInstanceOf(CpuDataMovementIr.StackPlan.class,
                        ((CpuDataMovementIr) stack.portableKernelIr()).plan()),
                () -> assertTrue(concat.movementGeometry().isPresent()),
                () -> assertEquals(3, new CpuPartitionPreparer().analyze(context(
                        new Operation(TensorCompositionKind.CONCAT,
                                new CompositionAxisAttrs(0)), List.of(0, 1, 0),
                        List.of(descriptor(DataType.INT32, Shape.of(2)),
                                descriptor(DataType.INT32, Shape.of(1))),
                        descriptor(DataType.INT32, Shape.of(5)))).requirements().size()));
    }

    @Test void rejectsWrongShapeCountAndNonInjectiveOutput() {
        var wrong = context(new Operation(TileKind.TILE, new TileAttrs(List.of(2L))),
                List.of(0), List.of(descriptor(DataType.INT64, Shape.of(2))),
                descriptor(DataType.INT64, Shape.of(5)));
        var repeatedOutput = context(new Operation(TileKind.TILE, new TileAttrs(List.of(2L))),
                List.of(0), List.of(descriptor(DataType.INT64, Shape.of(2))),
                descriptor(DataType.INT64, Shape.of(4),
                        LayoutDescriptor.of(Shape.of(4), new long[]{0}, 0, true)));
        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuPartitionLowering().lower(wrong)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuPartitionLowering().lower(repeatedOutput)));
    }

    @Test void rejectsExcludedWindowSignaturesTypesLayoutsAndOverflow() {
        var window = new Window2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false);
        var input = descriptor(DataType.FLOAT32, Shape.of(1, 1, 3, 3));
        var output = descriptor(DataType.FLOAT32, Shape.of(1, 4, 4));
        var unresolved = new TensorDescriptor(DataType.FLOAT32, Shape.of(1, 4, 4),
                Optional.empty(), false);
        var overflow = new Window2dAttrs(Long.MAX_VALUE, 1, 1, 1,
                0, 0, Long.MAX_VALUE, 1, false);
        var oneNode = context(new Operation(WindowTransformKind.UNFOLD2D, window), List.of(0),
                List.of(input), output);
        var secondNode = new CompiledNode(new NodeId(1), oneNode.nodes().getFirst().operation(),
                oneNode.nodes().getFirst().inputs(), oneNode.nodes().getFirst().outputs());
        var twoNodePartition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                List.of(new NodeId(0), new NodeId(1)));
        var twoNodes = new PrepareContext<>(twoNodePartition,
                List.of(oneNode.nodes().getFirst(), secondNode), oneNode.values(),
                oneNode.memoryRequirements(), oneNode.constants(), oneNode.backendInputs());
        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () -> lower(twoNodes)),
                () -> assertThrows(IllegalArgumentException.class, () -> lower(context(
                        new Operation(WindowTransformKind.UNFOLD2D, window), List.of(0, 0),
                        List.of(input), output))),
                () -> assertThrows(IllegalArgumentException.class, () -> lower(context(
                        new Operation(WindowTransformKind.UNFOLD2D,
                                new Unfold2dAttrs(window, ScalarValue.float64(0))),
                        List.of(0), List.of(input), output))),
                () -> assertThrows(IllegalArgumentException.class, () -> lower(context(
                        new Operation(WindowTransformKind.UNFOLD2D, window), List.of(0),
                        List.of(descriptor(DataType.INT32, Shape.of(1, 1, 3, 3))),
                        descriptor(DataType.INT32, Shape.of(1, 4, 4))))),
                () -> assertThrows(IllegalArgumentException.class, () -> lower(context(
                        new Operation(WindowTransformKind.UNFOLD2D, window), List.of(0),
                        List.of(input), unresolved))),
                () -> assertThrows(ArithmeticException.class, () -> lower(context(
                        new Operation(WindowTransformKind.UNFOLD2D, overflow), List.of(0),
                        List.of(input), output))),
                () -> assertThrows(IllegalArgumentException.class, () -> lower(context(
                        new Operation(WindowTransformKind.FOLD2D,
                                new Fold2dAttrs(Shape.of(1, 1, 3, 3), window)),
                        List.of(0), List.of(output), input))));
    }

    @Test void acceptsOneAndSixteenCompositionOccurrencesAndRejectsSeventeen() {
        var input = descriptor(DataType.INT32, Shape.of(1));
        var one = context(new Operation(TensorCompositionKind.CONCAT,
                        new CompositionAxisAttrs(0)), List.of(0), List.of(input),
                descriptor(DataType.INT32, Shape.of(1)));
        var sixteenOccurrences = java.util.Collections.nCopies(16, 0);
        var sixteen = context(new Operation(TensorCompositionKind.CONCAT,
                        new CompositionAxisAttrs(0)), sixteenOccurrences, List.of(input),
                descriptor(DataType.INT32, Shape.of(16)));
        var seventeenOccurrences = java.util.Collections.nCopies(17, 0);
        var seventeen = context(new Operation(TensorCompositionKind.CONCAT,
                        new CompositionAxisAttrs(0)), seventeenOccurrences, List.of(input),
                descriptor(DataType.INT32, Shape.of(17)));
        assertAll(
                () -> assertEquals(List.of(0), ((CpuDataMovementIr) lower(one)
                        .portableKernelIr()).plan().occurrenceToBoundary()),
                () -> assertEquals(sixteenOccurrences, ((CpuDataMovementIr) lower(sixteen)
                        .portableKernelIr()).plan().occurrenceToBoundary()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuPartitionLowering().lower(seventeen)));
    }

    private static CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        return new CpuPartitionLowering().lower(context);
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(Operation operation,
            List<Integer> occurrenceValues, List<TensorDescriptor> uniqueInputs,
            TensorDescriptor output) {
        var inputIds = occurrenceValues.stream().map(index -> new ValueId(index)).toList();
        ValueId outputId = new ValueId(uniqueInputs.size());
        var node = new CompiledNode(new NodeId(0), operation, inputIds, List.of(outputId));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                List.of(node.id()));
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < uniqueInputs.size(); index++) {
            ValueId id = new ValueId(index);
            values.add(new GraphValue(id, uniqueInputs.get(index)));
            memory.add(new LogicalMemoryRequirement(id, uniqueInputs.get(index), Optional.empty(),
                    List.of(partition), false));
        }
        values.add(new GraphValue(outputId, output));
        memory.add(new LogicalMemoryRequirement(outputId, output, Optional.of(partition),
                List.of(), true));
        return new PrepareContext<>(partition, List.of(node), values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
    }

    public static TensorDescriptor descriptor(DataType type, Shape shape) {
        return descriptor(type, shape, LayoutDescriptor.contiguous(shape));
    }

    public static TensorDescriptor descriptor(DataType type, Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(type, shape, Optional.of(layout), false);
    }
}
