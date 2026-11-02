package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.*;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Declaration, prepared unit, carrier and original-source retention regressions. */
class CpuStaticTypedOneCountPreparationTest {
    /** Verifies actual Compiler count formulas and shared original-source initialization recipes. */
    @Test void actualCompilerLossAndNormalizationCountsKeepSharedSourceInitialization() {
        try(var cpu=io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.open()) {
            for(boolean normalization:List.of(false,true)) {
                var input=tensor(Shape.of(4,2),true);
                io.github.pho001.synaptik.model.tensor.Tensor objective;
                if(normalization) objective=input.batchNormTraining(1,tensor(Shape.of(2),false),tensor(Shape.of(2),false),
                        tensor(Shape.of(2),false),tensor(Shape.of(2),false),
                        io.github.pho001.synaptik.model.datatype.ScalarValue.float32(.5f),
                        io.github.pho001.synaptik.model.datatype.ScalarValue.float32(.01f)).nextRunningMean().sum();
                else objective=input.meanSquaredError(tensor(Shape.of(4,2),false),io.github.pho001.synaptik.model.operation.loss.LossReduction.MEAN);
                var request=new io.github.pho001.synaptik.compiler.FunctionalGradientRequest(List.of(
                        new io.github.pho001.synaptik.compiler.FunctionalGradientRequest.Stage(
                            List.of(new io.github.pho001.synaptik.compiler.FunctionalGradientRequest.ForwardTensorReference(objective)),
                            List.of(Optional.empty()),List.of(input),false,
                            io.github.pho001.synaptik.compiler.FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
                var artifacts=io.github.pho001.synaptik.compiler.GraphCompilationPort.compile(
                        io.github.pho001.synaptik.config.compile.CompileMode.FORWARD_AND_BACKWARD,List.of(objective),Optional.of(request),
                        io.github.pho001.synaptik.config.compile.GraphOptimizationConfig.disabled(),
                        io.github.pho001.synaptik.config.compile.BackendIntent.unconstrained(),
                        io.github.pho001.synaptik.config.compile.PartitionScoringConfig.neutral(),
                        List.of(cpu.capabilityProvider()),List.of(cpu.availabilitySnapshot()));
                var retainedSources=new HashSet<ValueId>();var removed=new HashSet<ValueId>();
                for(var partition:artifacts.partitions()) {
                    var context=io.github.pho001.synaptik.prepare.GraphPreparation.project(artifacts,partition,CpuPartitionAnalysisInputs.DEFAULT);
                    var plan=new CpuPartitionPreparer().analyze(context).plan();
                    for(var unit:plan.units()) if(unit.portablePlan()!=null&&unit.portablePlan().portableKernelIr() instanceof CpuConstantWriteIr constant) {
                        assertEquals(normalization?Float.floatToRawIntBits(4):Float.floatToRawIntBits(8),constant.immediate().bits());
                        assertEquals(1,unit.boundaryValues().size());assertTrue(unit.runtimeFacts().workspaceDeclaration().isEmpty());
                        var first=context.nodes().get(unit.memberNodeOrdinals().getFirst());
                        var source=first.inputs().getFirst();retainedSources.add(source);
                        assertEquals(io.github.pho001.synaptik.model.datatype.ScalarValue.float32(1),context.constants().get(source));
                        assertTrue(plan.boundaryValues().contains(source));
                        for(int ordinal:unit.memberNodeOrdinals()) if(ordinal!=unit.memberNodeOrdinals().getLast())
                            for(var id:context.nodes().get(ordinal).outputs()) { assertFalse(plan.boundaryValues().contains(id));removed.add(id); }
                    }
                }
                assertFalse(retainedSources.isEmpty(),"actual Compiler count must reach constant-write lowering");
                assertFalse(removed.isEmpty(),"current oneLike and mean-loss formulas both have private expansion work");
                var preparations=artifacts.partitions().stream().map(p->cpu.partitionPreparation()).toList();
                try(var prepared=io.github.pho001.synaptik.prepare.GraphPreparation.prepare(artifacts,preparations,cpu.scheduleAssembler())) {
                    var creations=prepared.schedule().steps().stream()
                            .filter(io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.RepresentationCreationStep.class::isInstance)
                            .map(io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.RepresentationCreationStep.class::cast).toList();
                    assertEquals(1,creations.size());
                    long initialized=creations.getFirst().representationPlan().bufferPreparations().stream().flatMap(List::stream)
                            .filter(io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.InitializedBuffer.class::isInstance).count();
                    assertTrue(initialized>=retainedSources.size());
                }
            }
        }
    }

    /**
     * Creates a storage-free FLOAT32 graph leaf for actual Compiler gradient construction.
     * @param shape non-null fully static logical shape
     * @param gradient whether the leaf is eligible as a differentiation target
     * @return non-null fresh Tensor identity with canonical layout and no host storage
     */
    private static io.github.pho001.synaptik.model.tensor.Tensor tensor(Shape shape,boolean gradient) {
        return io.github.pho001.synaptik.model.tensor.TensorFactory.create(
                new io.github.pho001.synaptik.model.tensor.TensorDescriptor(DataType.FLOAT32,shape,
                    Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(shape)),gradient));
    }

    /** Checks intermediate-plan admission, final source retention and mixed-carrier finalization. */
    @Test void oneFoldedUnitDeclaresOnlyOutputAndMandatoryOriginalSourceForEveryTypeAndCarrier() {
        for(var type:CpuStaticTypedOneCountFoldingTest.types()) for(int chain=0;chain<=2;chain++)
            for(boolean heap:List.of(false,true)) {
                var c=CpuStaticTypedOneCountFoldingTest.context(type,Shape.of(2,3),new AxisReductionAttrs(1,false),Shape.of(2),chain,true);
                var carrier=heap?switch(type) {
                    case FLOAT32->CarrierAccess.FLOAT_ARRAY;case FLOAT64->CarrierAccess.DOUBLE_ARRAY;
                    case BFLOAT16,FLOAT16->CarrierAccess.SHORT_ARRAY;case INT32->CarrierAccess.INT_ARRAY;
                    case INT64->CarrierAccess.LONG_ARRAY;default->throw new AssertionError();
                }:CarrierAccess.MEMORY_SEGMENT;
                c=new PrepareContext<>(c.partition(),c.nodes(),c.values(),c.memoryRequirements(),c.constants(),
                        new CpuPartitionAnalysisInputs(false,List.of(carrier,heap?CarrierAccess.MEMORY_SEGMENT:arrayCarrier(type))));
                if (io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertLowUnsupported(c)) continue;
                var analysis=new CpuPartitionPreparer().analyze(c);var plan=analysis.plan();
                assertEquals(2,analysis.requirements().size());
                assertEquals(1,plan.units().size());
                var unit=plan.units().getFirst();
                assertInstanceOf(CpuConstantWriteIr.class,unit.portablePlan().portableKernelIr());
                assertEquals(List.of(new ValueId(chain+1)),unit.boundaryValues());
                assertEquals(java.util.stream.IntStream.rangeClosed(0,chain).boxed().toList(),unit.memberNodeOrdinals());
                assertEquals(List.of(carrier),unit.carrierPattern());
                assertFalse(unit.portablePlan().specialization().scratchParameter());
                assertTrue(plan.workspaceDeclaration().isEmpty());
                long sourceBytes=plan.bufferDeclarations().stream().filter(b->b.valueId().equals(new ValueId(0))).findFirst().orElseThrow().byteSize();
                assertEquals((chain==0?6:1)*type.byteWidth(),sourceBytes);
                finalizeOutputOnly(analysis);
            }
    }

    /**
     * Selects the primitive heap representation matching a numeric result type.
     * @param type non-null supported numeric count type
     * @return non-null matching array carrier, including short arrays for either low type
     */
    private static CarrierAccess arrayCarrier(DataType type) { return switch(type) {
        case FLOAT32->CarrierAccess.FLOAT_ARRAY;case FLOAT64->CarrierAccess.DOUBLE_ARRAY;case INT32->CarrierAccess.INT_ARRAY;
        case INT64->CarrierAccess.LONG_ARRAY;default->CarrierAccess.SHORT_ARRAY;
    }; }

    /**
     * Assigns exact fixed slots and finalizes one output-only unit with its retained source.
     * No source is added to the generated entry, and no physical run buffer is allocated.
     * @param analysis non-null immutable constant-write analysis with buffer-only requirements
     */
    private static void finalizeOutputOnly(BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis) {
        var buffers=new ArrayList<io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.BufferEntry>();
        var assignments=new ArrayList<io.github.pho001.synaptik.prepare.PreparationResourceAssignment>();
        for(var declaration:analysis.plan().bufferDeclarations()) {
            var slot=new io.github.pho001.synaptik.runtime.memory.BufferSlot(buffers.size());
            buffers.add(new io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.BufferEntry(slot,declaration.byteSize(),declaration.byteAlignment()));
            assignments.add(new io.github.pho001.synaptik.prepare.PreparationResourceAssignment.Buffer(declaration,slot,buffers.size()-1,0));
        }
        var memory=new io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan(buffers,List.of());
        var prepared=new CpuPartitionFinalizer().finalizePartition(new io.github.pho001.synaptik.prepare.BackendPartitionFinalization<>(analysis,memory,assignments));
        assertNotNull(prepared.executable());assertTrue(memory.workspaces().isEmpty());
    }

    /** Retains interleaved node membership and executes the count's dependent consumer in order. */
    @Test void nonAdjacentPrivateChainHasExactMembershipAndDownstreamDependencies() {
        var c=CpuStaticTypedOneCountFoldingTest.context(DataType.FLOAT32,Shape.of(2,3),NoOperationAttrs.INSTANCE,Shape.scalar(),2,true);
        var descriptors=new ArrayList<>(c.values().stream().map(GraphValue::descriptor).toList());
        descriptors.add(CpuScatterLoweringTest.desc(DataType.FLOAT32,Shape.scalar()));
        descriptors.add(CpuScatterLoweringTest.desc(DataType.FLOAT32,Shape.scalar()));
        var unrelated=new CompiledNode(new NodeId(10),new Operation(UnaryElementwiseKind.NEG,NoOperationAttrs.INSTANCE),List.of(new ValueId(0)),List.of(new ValueId(4)));
        var downstream=new CompiledNode(new NodeId(11),new Operation(UnaryElementwiseKind.NEG,NoOperationAttrs.INSTANCE),List.of(new ValueId(3)),List.of(new ValueId(5)));
        var nodes=List.of(c.nodes().get(0),unrelated,c.nodes().get(1),c.nodes().get(2),downstream);
        var base=CpuRecurrentScanLoweringTest.project(descriptors,nodes);
        var memory=base.memoryRequirements().stream().map(r->new LogicalMemoryRequirement(r.valueId(),r.descriptor(),r.producerPartition(),r.consumerPartitions(),
                r.valueId().equals(new ValueId(4))||r.valueId().equals(new ValueId(5)))).toList();
        var context=new PrepareContext<>(base.partition(),nodes,base.values(),memory,c.constants(),CpuPartitionAnalysisInputs.DEFAULT);
        var analysis=new CpuPartitionPreparer().analyze(context);
        var units=analysis.plan().units();
        var count=units.stream().filter(u->u.portablePlan().portableKernelIr() instanceof CpuConstantWriteIr).findFirst().orElseThrow();
        assertEquals(List.of(0,2,3),count.memberNodeOrdinals());
        int index=units.indexOf(count);
        var consumer=units.stream().filter(u->u.memberNodeOrdinals().contains(4)).findFirst().orElseThrow();
        assertEquals(List.of(index),consumer.dependencies());
        assertEquals(Set.of(new ValueId(0),new ValueId(3),new ValueId(4),new ValueId(5)),new HashSet<>(analysis.plan().boundaryValues()));
        try(var f=new Fixture(analysis,1,true,false,Map.of())) {
            f.executable.bind(f.state).execute();assertArrayEquals(new float[]{-6},f.floats(5));
        }
    }

    /** Protects each fan-out stage and proves absent facts read changed storage on later runs. */
    @Test void fanoutAtEachAffineStageRetainsMaterializationAndCurrentMutableInput() {
        for(int stage=1;stage<=2;stage++) {
            var c=CpuStaticTypedOneCountFoldingTest.context(DataType.FLOAT32,Shape.of(2,3),NoOperationAttrs.INSTANCE,Shape.scalar(),2,true);
            var descriptors=new ArrayList<>(c.values().stream().map(GraphValue::descriptor).toList());
            descriptors.add(CpuScatterLoweringTest.desc(DataType.FLOAT32,Shape.of(2,3)));
            var nodes=new ArrayList<>(c.nodes());
            nodes.add(new CompiledNode(new NodeId(10),new Operation(UnaryElementwiseKind.NEG,NoOperationAttrs.INSTANCE),List.of(new ValueId(stage)),List.of(new ValueId(4))));
            var base=CpuRecurrentScanLoweringTest.project(descriptors,nodes);
            var memory=base.memoryRequirements().stream().map(r->new LogicalMemoryRequirement(r.valueId(),r.descriptor(),r.producerPartition(),r.consumerPartitions(),r.valueId().equals(new ValueId(3))||r.valueId().equals(new ValueId(4)))).toList();
            var context=new PrepareContext<>(base.partition(),nodes,base.values(),memory,c.constants(),CpuPartitionAnalysisInputs.DEFAULT);
            var plan=new CpuPartitionPreparer().analyze(context).plan();
            assertTrue(plan.units().stream().noneMatch(u->u.portablePlan().portableKernelIr() instanceof CpuConstantWriteIr));
            assertTrue(plan.boundaryValues().contains(new ValueId(stage)));
        }
        var mutable=CpuStaticTypedOneCountFoldingTest.context(DataType.FLOAT32,Shape.of(3),NoOperationAttrs.INSTANCE,Shape.scalar(),0,false);
        var analysis=new CpuPartitionPreparer().analyze(mutable);
        try(var f=new Fixture(analysis,1,false,false,Map.of())) {
            for(int run=1;run<=2;run++) {
                f.fill(0,run);f.executable.bind(f.state).execute();assertArrayEquals(new float[]{3*run},f.floats(1));
            }
        }
    }
}
