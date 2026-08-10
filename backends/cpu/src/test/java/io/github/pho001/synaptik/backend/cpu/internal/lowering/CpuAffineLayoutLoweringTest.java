package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;
import org.junit.jupiter.api.Test;

public class CpuAffineLayoutLoweringTest {
    @Test void lowersSelectToExactlyTwoBoundariesAndOneInstructionFreeCopy() {
        var lowered = new CpuPartitionLowering().lower(select(DataType.INT32, List.of()));
        assertAll(
                () -> assertEquals(2, lowered.boundaryValues().size()),
                () -> assertTrue(lowered.virtualValues().isEmpty()),
                () -> assertTrue(lowered.kernelIr().instructions().isEmpty()),
                () -> assertArrayEquals(new long[]{3}, lowered.extents()),
                () -> assertArrayEquals(new long[]{1, 1, 4, 4, 7, 7},
                        lowered.affineAddressPairs()));
    }

    @Test void affinePreparationAlwaysUsesScalarComputeAndNoWorkspace() {
        var base = select(DataType.FLOAT32, List.of());
        var configured = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, List.of(),
                new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference
                                .VECTOR_IF_ELIGIBLE, 2, 2, 1)));
        var plan = new CpuPartitionPreparer().analyze(configured).plan();
        assertAll(
                () -> assertEquals("parallel-scalar", plan.executionStrategy().toString()),
                () -> assertEquals(2, plan.bufferDeclarations().size()),
                () -> assertTrue(plan.workspaceDeclaration().isEmpty()),
                () -> assertEquals(0, plan.vectorSpeciesBitSize()));
    }

    @Test void bfloat16UsesOnlyShortArrayOrSegmentCarrierForms() {
        var plan = new CpuPartitionPreparer().analyze(select(DataType.BFLOAT16,
                List.of(CarrierAccess.SHORT_ARRAY, CarrierAccess.MEMORY_SEGMENT))).plan();
        assertEquals(List.of(CarrierAccess.SHORT_ARRAY, CarrierAccess.MEMORY_SEGMENT),
                plan.carrierPattern());
    }

    @Test void composesEveryAdmittedAffineSemantic() {
        Shape twoThree = Shape.of(2, 3);
        TensorDescriptor dense23 = descriptor(DataType.INT64, twoThree,
                LayoutDescriptor.contiguous(twoThree));
        assertPairs(context(List.of(new Operation(ContiguousKind.CONTIGUOUS,
                        NoOperationAttrs.INSTANCE)), List.of(
                        descriptor(DataType.INT64, twoThree,
                                LayoutDescriptor.of(twoThree, new long[]{1, 2}, 0, true)), dense23)),
                0,0, 2,1, 4,2, 1,3, 3,4, 5,5);
        Shape threeTwo = Shape.of(3, 2);
        assertPairs(context(List.of(new Operation(ShapeTransformKind.RESHAPE,
                        new TargetShapeAttrs(threeTwo))), List.of(dense23,
                        descriptor(DataType.INT64, threeTwo,
                                LayoutDescriptor.of(threeTwo, new long[]{2,1}, 0, true)))),
                0,0, 1,1, 2,2, 3,3, 4,4, 5,5);
        Shape twoOne = Shape.of(2, 1), twoThreeExpanded = Shape.of(2, 3);
        assertPairs(context(List.of(new Operation(ShapeTransformKind.EXPAND,
                        new TargetShapeAttrs(twoThreeExpanded))), List.of(
                        descriptor(DataType.INT64, twoOne, LayoutDescriptor.contiguous(twoOne)),
                        descriptor(DataType.INT64, twoThreeExpanded,
                                LayoutDescriptor.of(twoThreeExpanded, new long[]{1,0}, 0, true)))),
                0,0, 1,1);
        assertPairs(context(List.of(new Operation(AxisTransformKind.PERMUTE,
                        new PermutationAttrs(List.of(1,0)))), List.of(dense23,
                        descriptor(DataType.INT64, threeTwo,
                                LayoutDescriptor.of(threeTwo, new long[]{1,3}, 0, true)))),
                0,0, 3,3, 1,1, 4,4, 2,2, 5,5);
        Shape expandedDims = Shape.of(2,1,3);
        var expandedDescriptor = descriptor(DataType.INT64, expandedDims,
                LayoutDescriptor.of(expandedDims, new long[]{3,3,1}, 0, true));
        assertPairs(context(List.of(new Operation(AxisTransformKind.EXPAND_DIMS,
                        new AxisTransformAttrs(1))), List.of(dense23, expandedDescriptor)),
                0,0, 1,1, 2,2, 3,3, 4,4, 5,5);
        assertPairs(context(List.of(new Operation(AxisTransformKind.SQUEEZE,
                        new AxisTransformAttrs(1))), List.of(expandedDescriptor,
                        descriptor(DataType.INT64, twoThree,
                                LayoutDescriptor.of(twoThree, new long[]{3,1}, 0, true)))),
                0,0, 1,1, 2,2, 3,3, 4,4, 5,5);
        Shape threeFour = Shape.of(3,4), twoFour = Shape.of(2,4);
        assertPairs(context(List.of(new Operation(SliceKind.SLICE,
                        new SliceAttrs(List.of(1L), List.of(2L), List.of(0), List.of(1L)))),
                        List.of(descriptor(DataType.INT64, threeFour,
                                        LayoutDescriptor.contiguous(threeFour)),
                                descriptor(DataType.INT64, twoFour,
                                        LayoutDescriptor.of(twoFour, new long[]{4,1}, 4, true)))),
                4,4, 5,5, 6,6, 7,7, 8,8, 9,9, 10,10, 11,11);
        Shape fourFour = Shape.of(4,4), twoTwo = Shape.of(2,2);
        assertPairs(context(List.of(new Operation(SliceKind.SLICE,
                        new CropToShapeAttrs(twoTwo, Shape.of(1,1)))), List.of(
                        descriptor(DataType.INT64, fourFour, LayoutDescriptor.contiguous(fourFour)),
                        descriptor(DataType.INT64, twoTwo,
                                LayoutDescriptor.of(twoTwo, new long[]{4,1}, 5, true)))),
                5,5, 6,6, 9,9, 10,10);
    }

    @Test void composesAcrossContiguousReshapeAndPermutationWithoutAnIntermediateSlot() {
        Shape sourceShape = Shape.of(2,3), reshapeShape = Shape.of(3,2), resultShape = Shape.of(2,3);
        var context = context(List.of(
                new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE),
                new Operation(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(reshapeShape)),
                new Operation(AxisTransformKind.PERMUTE, new PermutationAttrs(List.of(1,0)))),
                List.of(
                        descriptor(DataType.INT32, sourceShape,
                                LayoutDescriptor.of(sourceShape, new long[]{1,2}, 0, true)),
                        descriptor(DataType.INT32, sourceShape, LayoutDescriptor.contiguous(sourceShape)),
                        descriptor(DataType.INT32, reshapeShape,
                                LayoutDescriptor.of(reshapeShape, new long[]{2,1}, 0, true)),
                        descriptor(DataType.INT32, resultShape,
                                LayoutDescriptor.of(resultShape, new long[]{1,2}, 0, true))));
        var lowered = new CpuPartitionLowering().lower(context);
        assertAll(
                () -> assertArrayEquals(new long[]{0,0, 4,2, 3,4, 2,1, 1,3, 5,5},
                        lowered.affineAddressPairs()),
                () -> assertEquals(2, lowered.boundaryValues().size()),
                () -> assertEquals(2, lowered.virtualValues().size()),
                () -> assertEquals(2, new CpuPartitionPreparer().analyze(context).requirements().size()));
    }

    @Test void lowersEightNodesAndRejectsPublishedOrBranchedIntermediates() {
        Shape shape = Shape.of(3);
        TensorDescriptor descriptor = descriptor(DataType.BOOL, shape,
                LayoutDescriptor.contiguous(shape));
        List<Operation> operations = java.util.Collections.nCopies(8,
                new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE));
        List<TensorDescriptor> descriptors = java.util.Collections.nCopies(9, descriptor);
        var eight = context(operations, descriptors);
        var lowered = new CpuPartitionLowering().lower(eight);
        assertAll(
                () -> assertEquals(7, lowered.virtualValues().size()),
                () -> assertEquals(2, new CpuPartitionPreparer().analyze(eight).requirements().size()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuPartitionLowering().lower(context(
                                java.util.Collections.nCopies(9, operations.getFirst()),
                                java.util.Collections.nCopies(10, descriptor)))));

        var two = context(operations.subList(0, 2), descriptors.subList(0, 3));
        var publishedMemory = new ArrayList<>(two.memoryRequirements());
        var old = publishedMemory.get(1);
        publishedMemory.set(1, new LogicalMemoryRequirement(old.valueId(), old.descriptor(),
                old.producerPartition(), old.consumerPartitions(), true));
        var published = new PrepareContext<>(two.partition(), two.nodes(), two.values(),
                publishedMemory, Map.of(), two.backendInputs());
        assertThrows(IllegalArgumentException.class,
                () -> new CpuPartitionLowering().lower(published));

        CompiledNode second = two.nodes().get(1);
        assertThrows(IllegalArgumentException.class, () -> new CompiledNode(second.id(),
                second.operation(), List.of(new ValueId(1), new ValueId(0)), second.outputs()));
    }

    @Test void handlesScalarZeroElementAndDistinctAddressDomains() {
        Shape scalar = Shape.scalar();
        var scalarContext = context(List.of(new Operation(ContiguousKind.CONTIGUOUS,
                        NoOperationAttrs.INSTANCE)), List.of(
                        descriptor(DataType.FLOAT64, scalar,
                                LayoutDescriptor.of(scalar, new long[0], 2, true)),
                        descriptor(DataType.FLOAT64, scalar, LayoutDescriptor.contiguous(scalar))));
        Shape zero = Shape.of(0,3);
        var zeroContext = context(List.of(new Operation(ContiguousKind.CONTIGUOUS,
                        NoOperationAttrs.INSTANCE)), List.of(
                        descriptor(DataType.FLOAT64, zero, LayoutDescriptor.contiguous(zero)),
                        descriptor(DataType.FLOAT64, zero, LayoutDescriptor.contiguous(zero))));
        var scalarLowered = new CpuPartitionLowering().lower(scalarContext);
        var zeroLowered = new CpuPartitionLowering().lower(zeroContext);
        var expanded = new CpuPartitionLowering().lower(context(List.of(new Operation(
                ShapeTransformKind.EXPAND, new TargetShapeAttrs(Shape.of(2,3)))), List.of(
                descriptor(DataType.INT32, Shape.of(2,1), LayoutDescriptor.contiguous(Shape.of(2,1))),
                descriptor(DataType.INT32, Shape.of(2,3),
                        LayoutDescriptor.of(Shape.of(2,3), new long[]{1,0}, 0, true)))));
        assertAll(
                () -> assertArrayEquals(new long[]{2,0}, scalarLowered.affineAddressPairs()),
                () -> assertEquals(1, scalarLowered.elementCount()),
                () -> assertEquals(0, zeroLowered.elementCount()),
                () -> assertEquals(0, zeroLowered.affineAddressPairs().length),
                () -> assertEquals(2, expanded.elementCount()),
                () -> assertArrayEquals(new long[]{0,0,1,1}, expanded.affineAddressPairs()));
    }

    private static void assertPairs(PrepareContext<CpuPartitionAnalysisInputs> context,
            long... expected) {
        assertArrayEquals(expected, new CpuPartitionLowering().lower(context).affineAddressPairs());
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(type, shape, Optional.of(layout), false);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(List<Operation> operations,
            List<TensorDescriptor> descriptors) {
        var nodes = new ArrayList<CompiledNode>();
        for (int i = 0; i < operations.size(); i++) nodes.add(new CompiledNode(new NodeId(i),
                operations.get(i), List.of(new ValueId(i)), List.of(new ValueId(i + 1L))));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < descriptors.size(); i++) {
            ValueId id = new ValueId(i);
            values.add(new GraphValue(id, descriptors.get(i)));
            memory.add(new LogicalMemoryRequirement(id, descriptors.get(i),
                    i == 0 ? Optional.empty() : Optional.of(partition),
                    i == descriptors.size() - 1 ? List.of() : List.of(partition),
                    i == descriptors.size() - 1));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> select(
            DataType type, List<CarrierAccess> carriers) {
        Shape inputShape = Shape.of(3, 3), outputShape = Shape.of(3);
        var input = new TensorDescriptor(type, inputShape,
                Optional.of(LayoutDescriptor.contiguous(inputShape)), false);
        var output = new TensorDescriptor(type, outputShape,
                Optional.of(LayoutDescriptor.of(outputShape, new long[]{3}, 1, true)), false);
        var base = context(List.of(new Operation(SelectKind.SELECT, new SelectAttrs(1, 1))),
                List.of(input, output));
        return new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, carriers,
                CpuPartitionAnalysisInputs.DEFAULT.portableExecution()));
    }
}
