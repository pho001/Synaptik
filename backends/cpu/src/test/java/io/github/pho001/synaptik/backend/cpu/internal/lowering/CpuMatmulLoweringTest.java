package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Optional;
import java.util.List;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.ArrayList;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

public final class CpuMatmulLoweringTest {
    private final CpuMatmulLowering lowering = new CpuMatmulLowering();

    @Test void normalizesVectorPromotionAndRightAlignedBatchBroadcast() {
        var vector = lowering.lower(descriptor(DataType.FLOAT64, Shape.of(5)),
                descriptor(DataType.FLOAT64, Shape.of(5)), descriptor(DataType.FLOAT64, Shape.scalar()));
        var batch = lowering.lower(descriptor(DataType.FLOAT32, Shape.of(3, 1, 2, 5)),
                descriptor(DataType.FLOAT32, Shape.of(4, 5, 7)),
                descriptor(DataType.FLOAT32, Shape.of(3, 4, 2, 7)));
        assertAll(() -> assertTrue(vector.removedM()), () -> assertTrue(vector.removedN()),
                () -> assertEquals(1, vector.m()), () -> assertEquals(5, vector.k()),
                () -> assertEquals(1, vector.n()), () -> assertEquals(1, vector.outputCount()),
                () -> assertArrayEquals(new long[] {3, 4}, batch.batchExtents()),
                () -> assertArrayEquals(new long[] {10, 0}, batch.leftBatchStrides()),
                () -> assertArrayEquals(new long[] {0, 35}, batch.rightBatchStrides()),
                () -> assertEquals(12, batch.batchCount()));
    }

    @Test void acceptsZeroKAndRejectsMismatchOrOverlappingOutput() {
        var empty = lowering.lower(descriptor(DataType.FLOAT32, Shape.of(2, 0)),
                descriptor(DataType.FLOAT32, Shape.of(0, 3)),
                descriptor(DataType.FLOAT32, Shape.of(2, 3)));
        assertEquals(0, empty.k()); assertEquals(6, empty.outputCount());
        assertThrows(IllegalArgumentException.class, () -> lowering.lower(
                descriptor(DataType.INT32, Shape.of(2, 3)), descriptor(DataType.INT32, Shape.of(4, 2)),
                descriptor(DataType.INT32, Shape.of(2, 2))));
        var shape = Shape.of(2, 2);
        var overlap = new TensorDescriptor(DataType.INT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 1}, 0, true)), false);
        assertThrows(IllegalArgumentException.class, () -> lowering.lower(
                descriptor(DataType.INT32, Shape.of(2, 3)), descriptor(DataType.INT32, Shape.of(3, 2)), overlap));
    }

    @Test void wholePartitionLoweringCarriesTypedIrGeometryAndExecutableAssociation() {
        var lowered = new CpuPartitionLowering().lower(context(DataType.FLOAT32,
                Shape.of(2, 3), Shape.of(3, 4), Shape.of(2, 4)));
        assertInstanceOf(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.class,
                lowered.portableKernelIr());
        assertTrue(lowered.matmulIr().isPresent());
        assertEquals(8, lowered.elementCount());
        assertEquals(8, lowered.matmulGeometry().orElseThrow().outputCount());
        assertEquals(List.of(new io.github.pho001.synaptik.model.graph.ValueId(0),
                new io.github.pho001.synaptik.model.graph.ValueId(1),
                new io.github.pho001.synaptik.model.graph.ValueId(2)), lowered.boundaryValues());
    }

    @Test void preparationRetainsSchema54TypedScalarFallbackAndWorkDomain() {
        var plan = new CpuPartitionPreparer().analyze(context(DataType.FLOAT32,
                Shape.of(2, 3), Shape.of(3, 4), Shape.of(2, 4))).plan();
        var unit = plan.units().getFirst();
        assertAll(() -> assertEquals(54, unit.portablePlan().specialization()
                        .classIdentitySchema()),
                () -> assertTrue(unit.portablePlan().specialization().matmulIr().isPresent()),
                () -> assertEquals(8, unit.elementCount()),
                () -> assertEquals(8, plan.elementCount()),
                () -> assertEquals(io.github.pho001.synaptik.backend.cpu.internal.ir
                        .CpuSpecializedSubgraph.ExecutionDisposition.EXECUTABLE_ALTERNATIVES,
                        plan.specializedSubgraphs().getFirst().disposition()));
    }

    @Test void preparationThreadsExactRowAndTileWorkDomains() {
        var directVector=new CpuPartitionPreparer().analyze(context(DataType.FLOAT32,
                Shape.of(2,63),Shape.of(63,128),Shape.of(2,128))).plan().units().getFirst();
        var scalarTile=new CpuPartitionPreparer().analyze(context(DataType.BFLOAT16,
                Shape.of(32,63),Shape.of(63,48),Shape.of(32,48))).plan().units().getFirst();
        var vectorTile=new CpuPartitionPreparer().analyze(context(DataType.FLOAT32,
                Shape.of(32,127),Shape.of(127,256),Shape.of(32,256))).plan().units().getFirst();
        int lanes=jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length();
        assertAll(()->assertEquals(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr
                        .Realization.DIRECT_N_VECTOR,directVector.portablePlan().specialization()
                        .matmulIr().orElseThrow().realization()),
                ()->assertEquals(2,directVector.elementCount()),
                ()->assertEquals(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr
                        .Realization.TILED_SCALAR_2X2,scalarTile.portablePlan().specialization()
                        .matmulIr().orElseThrow().realization()),
                ()->assertEquals(384,scalarTile.elementCount()),
                ()->assertEquals(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr
                        .Realization.TILED_N_VECTOR_2X2,vectorTile.portablePlan().specialization()
                        .matmulIr().orElseThrow().realization()),
                ()->assertEquals(16L*((256L+2L*lanes-1)/(2L*lanes)),vectorTile.elementCount()));
    }

    @Test void recognizedLinearSuffixRetainsCanonicalSplitAndOneFactDerivedFusedAlternative() {
        var plan=new CpuPartitionPreparer().analyze(linear()).plan();
        var legal=plan.fusionDecisions().stream().filter(io.github.pho001.synaptik.backend.cpu.internal.ir
                .CpuFusionDecision.LegalCandidate.class::isInstance).map(io.github.pho001.synaptik.backend.cpu.internal.ir
                .CpuFusionDecision.LegalCandidate.class::cast).toList();
        var selection=plan.fusionDecisions().stream().filter(io.github.pho001.synaptik.backend.cpu.internal.ir
                .CpuFusionDecision.Selection.class::isInstance).map(io.github.pho001.synaptik.backend.cpu.internal.ir
                .CpuFusionDecision.Selection.class::cast).findFirst().orElseThrow();
        assertAll(()->assertEquals(2,legal.size()),
                ()->assertTrue(legal.stream().anyMatch(candidate->candidate.identity().units().size()==2
                        &&candidate.identity().units().stream().map(unit->unit.memberNodePositions()).toList()
                            .equals(List.of(List.of(0),List.of(1,2))))),
                ()->assertTrue(legal.stream().anyMatch(candidate->candidate.identity().units().size()==1
                        &&candidate.identity().units().getFirst().memberNodePositions().equals(List.of(0,1,2)))),
                ()->assertTrue(legal.stream().anyMatch(candidate->candidate.identity().equals(selection.selected()))),
                ()->assertEquals(List.of(0,1),plan.specializedSubgraphs().getFirst().baselineUnitIndices()));
    }

    @Test void matmulRetainsDirectAndBothSingleCopiesWhileRejectingCoConsumedPair() {
        var policy=new CpuPartitionAnalysisInputs.MaterializationPolicy(true,0,1,20,1,3,1_000_000,1,1);
        var inputs=new CpuPartitionAnalysisInputs(false,CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT,policy);
        var plan=new CpuPartitionPreparer().analyze(materialized(inputs)).plan();
        var decisions=plan.representationDecisions();
        var pair=decisions.stream().filter(io.github.pho001.synaptik.backend.cpu.internal.ir
                .CpuRepresentationDecision.Rejection.class::isInstance).map(io.github.pho001.synaptik.backend.cpu.internal.ir
                .CpuRepresentationDecision.Rejection.class::cast).filter(value->value.reason()
                ==io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision.RejectionReason.CO_CONSUMED_PAIR)
                .findFirst().orElseThrow();
        var singles=decisions.stream().filter(io.github.pho001.synaptik.backend.cpu.internal.ir
                .CpuRepresentationDecision.Variant.class::isInstance).map(io.github.pho001.synaptik.backend.cpu.internal.ir
                .CpuRepresentationDecision.Variant.class::cast).filter(value->value.identity().topology()
                .equals(pair.identity().topology())&&value.identity().materializations().size()==1).toList();
        var selected=(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision.Selection)
                decisions.getLast();
        assertAll(()->assertEquals(List.of(List.of(0),List.of(1)),singles.stream().map(value->value
                        .identity().materializations().stream().map(copy->copy.sourceBoundaryPosition()).toList()).toList()),
                ()->assertTrue(selected.selected().materializations().isEmpty()),
                ()->assertEquals(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision
                        .SelectionReason.DIRECT_MATERIALIZATION_UNPROVED,selected.reason()),
                ()->assertTrue(plan.materializations().isEmpty()));
    }

    @Test void retainsExactNamedRightMaterializationCompanionPlans() throws Exception {
        var policy=new CpuPartitionAnalysisInputs.MaterializationPolicy(true,0,1,20,1,3,
                1_000_000_000L,1,1);
        var inputs=new CpuPartitionAnalysisInputs(false,
                CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT,policy);
        var f32=new CpuPartitionPreparer().analyze(materializedRight(DataType.FLOAT32,false,inputs)).plan();
        var f64=new CpuPartitionPreparer().analyze(materializedRight(DataType.FLOAT64,true,inputs)).plan();
        StringBuilder evidence=new StringBuilder();
        retainRightVariant("MATERIALIZE-RIGHT-F32",f32,evidence);
        retainRightVariant("MATERIALIZE-RIGHT-F64-BATCH",f64,evidence);
        var root=java.nio.file.Path.of(
                "/private/tmp/synaptik-cpu-0008f-retained-evidence-20260828/materialization");
        java.nio.file.Files.createDirectories(root);
        java.nio.file.Files.writeString(root.resolve("retained-candidates.txt"),evidence);
    }

    private static void retainRightVariant(String name,
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan,
            StringBuilder evidence) {
        var right=plan.representationDecisions().stream().filter(
                io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision.Variant.class::isInstance)
                .map(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision.Variant.class::cast)
                .filter(value->value.identity().materializations().stream().map(copy->copy.sourceBoundaryPosition())
                        .toList().equals(List.of(1))).findFirst().orElseThrow();
        var copy=right.identity().materializations().getFirst();
        assertAll(()->assertTrue(copy.byteCount()>0),()->assertTrue(right.copiedBytes()>0),
                ()->assertTrue(right.representationCost()>0),
                ()->assertTrue(plan.materializations().isEmpty()));
        evidence.append(name).append('\n').append("variant=").append(right).append('\n')
                .append("selection=").append(plan.representationDecisions().getLast()).append('\n');
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type, Shape left,
            Shape right, Shape result) {
        return CpuScatterLoweringTest.context(new Operation(MatmulKind.MATMUL,
                        io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE),
                List.of(0, 1), List.of(descriptor(type, left), descriptor(type, right)),
                descriptor(type, result));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> linear() {
        var nodes=List.of(new CompiledNode(new NodeId(0),new Operation(MatmulKind.MATMUL,
                        NoOperationAttrs.INSTANCE),List.of(new ValueId(0),new ValueId(1)),List.of(new ValueId(2))),
                new CompiledNode(new NodeId(1),new Operation(BinaryArithmeticKind.ADD,
                        NoOperationAttrs.INSTANCE),List.of(new ValueId(2),new ValueId(3)),List.of(new ValueId(4))),
                new CompiledNode(new NodeId(2),new Operation(UnaryElementwiseKind.RELU,
                        NoOperationAttrs.INSTANCE),List.of(new ValueId(4)),List.of(new ValueId(5))));
        var descriptors=List.of(descriptor(DataType.FLOAT32,Shape.of(2,3)),
                descriptor(DataType.FLOAT32,Shape.of(3,4)),descriptor(DataType.FLOAT32,Shape.of(2,4)),
                descriptor(DataType.FLOAT32,Shape.of(4)),descriptor(DataType.FLOAT32,Shape.of(2,4)),
                descriptor(DataType.FLOAT32,Shape.of(2,4)));
        var partition=new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());var values=new ArrayList<GraphValue>();
        var memory=new ArrayList<LogicalMemoryRequirement>();Set<ValueId> outputs=Set.of(new ValueId(5));
        for(int index=0;index<descriptors.size();index++){ValueId id=new ValueId(index);
            var descriptor=descriptors.get(index);boolean produced=nodes.stream().anyMatch(n->n.outputs().contains(id));
            boolean consumed=nodes.stream().anyMatch(n->n.inputs().contains(id));boolean output=outputs.contains(id);
            values.add(new GraphValue(id,descriptor));memory.add(new LogicalMemoryRequirement(id,descriptor,
                    produced?Optional.of(partition):Optional.empty(),consumed&&!output?List.of(partition):List.of(),output));}
        return new PrepareContext<>(partition,nodes,values,memory,Map.of(),CpuPartitionAnalysisInputs.DEFAULT);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> materialized(
            CpuPartitionAnalysisInputs inputs) {
        var node=new CompiledNode(new NodeId(0),new Operation(MatmulKind.MATMUL,NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0),new ValueId(1)),List.of(new ValueId(2)));
        Shape leftShape=Shape.of(2,3),rightShape=Shape.of(3,4),resultShape=Shape.of(2,4);
        var descriptors=List.of(new TensorDescriptor(DataType.FLOAT32,leftShape,Optional.of(
                        LayoutDescriptor.of(leftShape,new long[]{1,2},0,true)),false),
                new TensorDescriptor(DataType.FLOAT32,rightShape,Optional.of(
                        LayoutDescriptor.of(rightShape,new long[]{1,3},0,true)),false),
                descriptor(DataType.FLOAT32,resultShape));
        var partition=new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,List.of(node.id()));
        var values=new ArrayList<GraphValue>();var memory=new ArrayList<LogicalMemoryRequirement>();
        for(int index=0;index<3;index++){ValueId id=new ValueId(index);var descriptor=descriptors.get(index);
            values.add(new GraphValue(id,descriptor));memory.add(new LogicalMemoryRequirement(id,descriptor,
                    index==2?Optional.of(partition):Optional.empty(),index<2?List.of(partition):List.of(),index==2));}
        return new PrepareContext<>(partition,List.of(node),values,memory,Map.of(),inputs);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> materializedRight(DataType type,
            boolean batched,CpuPartitionAnalysisInputs inputs) {
        var node=new CompiledNode(new NodeId(0),new Operation(MatmulKind.MATMUL,NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0),new ValueId(1)),List.of(new ValueId(2)));
        Shape leftShape=batched?Shape.of(3,16,127):Shape.of(32,127);
        Shape rightShape=Shape.of(127,256);
        Shape resultShape=batched?Shape.of(3,16,256):Shape.of(32,256);
        var descriptors=List.of(descriptor(type,leftShape),new TensorDescriptor(type,rightShape,
                Optional.of(LayoutDescriptor.of(rightShape,new long[]{1,127},0,true)),false),
                descriptor(type,resultShape));
        var partition=new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,List.of(node.id()));
        var values=new ArrayList<GraphValue>();var memory=new ArrayList<LogicalMemoryRequirement>();
        for(int index=0;index<3;index++){ValueId id=new ValueId(index);var descriptor=descriptors.get(index);
            values.add(new GraphValue(id,descriptor));memory.add(new LogicalMemoryRequirement(id,descriptor,
                    index==2?Optional.of(partition):Optional.empty(),index<2?List.of(partition):List.of(),index==2));}
        return new PrepareContext<>(partition,List.of(node),values,memory,Map.of(),inputs);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }
}
