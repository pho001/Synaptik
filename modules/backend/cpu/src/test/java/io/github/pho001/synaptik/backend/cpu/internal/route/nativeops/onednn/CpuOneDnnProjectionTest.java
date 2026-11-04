package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Cold common-unit boundary tests; representability is not native implementation qualification. */
final class CpuOneDnnProjectionTest {
    @Test void selectedReductionDeclaresNativeStorageWithoutUnusedOrdinaryAccumulator() {
        String requested = System.getenv("SYNAPTIK_ONEDNN_LIBRARY");
        org.junit.jupiter.api.Assumptions.assumeTrue(requested != null && !requested.isBlank(),
                "SYNAPTIK_ONEDNN_LIBRARY is not supplied");
        var context = CpuAggregateLoweringTest.context(AggregateReductionKind.SUM,
                DataType.FLOAT32, Shape.of(65_536), NoOperationAttrs.INSTANCE, Shape.scalar());
        var ordinary = new CpuPartitionPreparer().analyze(context);
        var ordinaryState = ordinary.plan().units().getFirst().runtimeFacts()
                .workspaceDeclaration().orElseThrow();
        try (var library = CpuOneDnnLibrary.open(java.nio.file.Path.of(requested))) {
            var selected = new CpuOneDnnSelector(library, CpuOneDnnRoutePlan.Api.PRIMITIVE)
                    .select(context, ordinary, ordinary);
            assertEquals(CpuPartitionPreparationPlan.Route.ONEDNN, selected.plan().route());
            assertFalse(selected.requirements().contains(ordinaryState));
            var nativeUnit = selected.plan().oneDnnPlan().orElseThrow().units().getFirst();
            assertEquals(nativeUnit.scratchpadBytes(), nativeUnit.invocationWorkspace().byteSize());
            assertTrue(selected.requirements().containsAll(nativeUnit.workspaceRequirements()));
            assertEquals(1, nativeUnit.inputCopies().size());
            assertEquals(65_536L * Float.BYTES,
                    nativeUnit.inputCopies().getFirst().workspace().byteSize());
        }
    }

    @Test void scalarSitesUseFixedDeduplicatedConstantsAndKeepEverySsaSite() {
        var add = new Operation(ScalarElementwiseKind.ADD, new ScalarValueAttrs(ScalarValue.float32(-0.0f)));
        var unit = chain(add, add, false).units().getFirst();
        var projection = CpuOneDnnProjection.from(unit).orElseThrow();
        var pipeline = assertInstanceOf(CpuOneDnnKernelSpec.Pipeline.class, projection.spec());
        assertEquals(1, pipeline.inputs().size());
        assertEquals(1, pipeline.constants().size());
        assertEquals(List.of(0x80000000), pipeline.constants().getFirst().rawWords());
        assertEquals(2, pipeline.steps().size());
        assertEquals(1, pipeline.intermediates().size());
        assertEquals(List.of(0, 3), pipeline.steps().getFirst().inputSlots());
        assertEquals(List.of(2, 3), pipeline.steps().getLast().inputSlots());
        assertEquals(1, pipeline.steps().getLast().outputSlot());
        assertEquals(List.of(0), projection.inputBoundaryPositions());
        assertEquals(unit.boundaryValues().size() - 1, projection.outputBoundaryPosition());
    }

    @Test void unsupportedSiteRejectsWholeUnitAndSavedSiblingIsNeverDiscarded() {
        var add = new Operation(ScalarElementwiseKind.ADD, new ScalarValueAttrs(ScalarValue.float32(1)));
        var log = new Operation(UnaryElementwiseKind.LOG, NoOperationAttrs.INSTANCE);
        assertTrue(CpuOneDnnProjection.from(chain(add, log, false).units().getFirst()).isEmpty());
        var saved = chain(add, add, true);
        assertEquals(2, saved.units().stream().mapToInt(CpuPartitionPreparationPlan.ExecutionUnitPlan::outputCount).sum());
        for (var unit : saved.units()) {
            if (unit.outputCount() > 1) assertTrue(CpuOneDnnProjection.from(unit).isEmpty());
        }
    }

    @Test void greaterRelationsReverseOperandsAndKeepOneByteBooleanOrigins() {
        Shape shape = Shape.of(4);
        var input = descriptor(DataType.FLOAT32, shape, new long[] {1}, 2);
        var output = descriptor(DataType.BOOL, shape, new long[] {1}, 3);
        for (var kind : List.of(BinaryComparisonKind.GREATER_THAN, BinaryComparisonKind.GREATER_OR_EQUAL)) {
            var context = CpuScatterLoweringTest.context(new Operation(kind, NoOperationAttrs.INSTANCE),
                    List.of(0, 1), List.of(input, input), output);
            var unit = new CpuPartitionPreparer().analyze(context).plan().units().getFirst();
            var projection = CpuOneDnnProjection.from(unit).orElseThrow();
            var pipeline = assertInstanceOf(CpuOneDnnKernelSpec.Pipeline.class, projection.spec());
            var step = pipeline.steps().getFirst();
            var binary = assertInstanceOf(CpuOneDnnKernelSpec.Binary.class, step.spec());
            assertEquals(kind == BinaryComparisonKind.GREATER_THAN
                    ? CpuOneDnnKernelSpec.BinaryAlgorithm.LT : CpuOneDnnKernelSpec.BinaryAlgorithm.LE,
                    binary.algorithm());
            assertEquals(List.of(1, 0), step.inputSlots());
            assertEquals(DataType.BOOL, pipeline.output().dataType());
            assertEquals(4, pipeline.output().byteSize());
            assertEquals(List.of(8L, 8L, 3L), projection.byteOffsets());
        }
    }

    @Test void meanRetainsReducedAxesAndMandatoryDivisionWhileEmptyAndF64StayOrdinary() {
        var context = CpuAggregateLoweringTest.context(AggregateReductionKind.MEAN,
                DataType.FLOAT32, Shape.of(2, 3), new AxisReductionAttrs(1, false), Shape.of(2));
        var unit = new CpuPartitionPreparer().analyze(context).plan().units().getFirst();
        var pipeline = assertInstanceOf(CpuOneDnnKernelSpec.Pipeline.class,
                CpuOneDnnProjection.from(unit).orElseThrow().spec());
        assertEquals(List.of(2L, 1L), pipeline.output().dimensions());
        assertEquals(8, pipeline.output().byteSize());
        assertEquals(CpuOneDnnKernelSpec.ReductionAlgorithm.SUM,
                assertInstanceOf(CpuOneDnnKernelSpec.Reduction.class, pipeline.steps().getFirst().spec()).algorithm());
        assertEquals(CpuOneDnnKernelSpec.BinaryAlgorithm.DIV,
                assertInstanceOf(CpuOneDnnKernelSpec.Binary.class, pipeline.steps().getLast().spec()).algorithm());
        assertEquals(List.of(Float.floatToRawIntBits(3f)), pipeline.constants().getFirst().rawWords());
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            var unsupported = CpuAggregateLoweringTest.context(AggregateReductionKind.SUM,
                    type, Shape.of(2, type == DataType.FLOAT32 ? 0 : 3),
                    new AxisReductionAttrs(1, false), Shape.of(2));
            assertTrue(CpuOneDnnProjection.from(new CpuPartitionPreparer().analyze(unsupported)
                    .plan().units().getFirst()).isEmpty());
        }
    }

    @Test void contiguousProjectsPhysicalTransposeWithoutConversionOrOriginLoss() {
        Shape shape = Shape.of(2, 3);
        var source = descriptor(DataType.FLOAT32, shape, new long[] {1, 2}, 5);
        var target = descriptor(DataType.FLOAT32, shape, new long[] {3, 1}, 1);
        var context = CpuScatterLoweringTest.context(new Operation(ContiguousKind.CONTIGUOUS,
                NoOperationAttrs.INSTANCE), List.of(0), List.of(source), target);
        var projection = CpuOneDnnProjection.from(new CpuPartitionPreparer().analyze(context)
                .plan().units().getFirst()).orElseThrow();
        var reorder = assertInstanceOf(CpuOneDnnKernelSpec.Reorder.class, projection.spec());
        assertEquals(List.of(1L, 2L), reorder.input().strides());
        assertEquals(List.of(3L, 1L), reorder.output().strides());
        assertEquals(List.of(20L, 4L), projection.byteOffsets());
        assertEquals(24, reorder.output().byteSize());
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, long[] strides, long origin) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.of(shape, strides, origin, true)), false);
    }

    private static CpuPartitionPreparationPlan chain(Operation first, Operation second, boolean saveIntermediate) {
        Shape shape = Shape.of(8);
        var descriptor = CpuScatterLoweringTest.desc(DataType.FLOAT32, shape);
        var nodes = List.of(new CompiledNode(new NodeId(0), first, List.of(new ValueId(0)), List.of(new ValueId(1))),
                new CompiledNode(new NodeId(1), second, List.of(new ValueId(1)), List.of(new ValueId(2))));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < 3; index++) {
            var id = new ValueId(index);
            values.add(new GraphValue(id, descriptor));
            memory.add(new LogicalMemoryRequirement(id, descriptor, index == 0 ? Optional.empty() : Optional.of(partition),
                    index < 2 ? List.of(partition) : List.of(), index == 2 || index == 1 && saveIntermediate));
        }
        return new CpuPartitionPreparer().analyze(new PrepareContext<>(partition, nodes, values, memory,
                Map.of(), CpuPartitionAnalysisInputs.DEFAULT)).plan();
    }
}
