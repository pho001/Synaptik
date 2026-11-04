package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.*;
import java.lang.classfile.instruction.*;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Compact-input semantic, topology, cache and generated-body proofs; no low numerical oracle. */
public class CpuBroadcastFoldGeneratedKernelTest {
    /**
     * Builds the actual Tensor raw repetition recipe without allocating physical tensor storage.
     * Descriptors retain every logical column; only the final fold result is public. Neither
     * caller-owned shape nor window is mutated or retained as mutable invocation state.
     *
     * @param type non-null represented F32/F64 type, or a low type for topology-only negative controls
     * @param output non-null fully static rank-four restored [N,C,H,W] shape with an admitted window
     * @param window non-null positive-kernel/stride/dilation, nonnegative-padding fold window
     * @param sourceOffset nonnegative compact-source storage origin, in represented elements
     * @param stridedOutput whether to use twice-canonical output strides and element offset three
     * @return new immutable CPU context with logical raw mappings and resolved physical layouts;
     *     never null and owning no runtime resources
     * @throws NullPointerException if type, shape or window is null
     * @throws IllegalArgumentException if the Tensor recipe or resolved layout is invalid
     * @throws ArithmeticException if checked Tensor shape/layout construction overflows
     */
    public static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type, Shape output,
            Window2dAttrs window, long sourceOffset, boolean stridedOutput) {
        long h = output.toLongArray()[2], w = output.toLongArray()[3];
        long oh = count(h, window.paddingHeight(), window.dilationHeight() * (window.kernelHeight()-1)+1, window.strideHeight(), window.ceilMode());
        long ow = count(w, window.paddingWidth(), window.dilationWidth() * (window.kernelWidth()-1)+1, window.strideWidth(), window.ceilMode());
        long n = output.toLongArray()[0], c = output.toLongArray()[1], p = oh*ow, k = window.kernelHeight()*window.kernelWidth();
        Shape source = Shape.of(n,c,p);
        var input = TensorFactory.create(new TensorDescriptor(type,source,Optional.of(LayoutDescriptor.of(source,
                LayoutDescriptor.contiguous(source).strides(),sourceOffset,true)),false));
        var tensors = new ArrayList<Tensor>(); tensors.add(input);
        tensors.add(input.expandDims(2));
        tensors.add(tensors.getLast().expand(Shape.of(n,c,k,p)));
        tensors.add(tensors.getLast().contiguous());
        tensors.add(tensors.getLast().reshape(Shape.of(n,c*k,p)));
        tensors.add(tensors.getLast().contiguous());
        tensors.add(tensors.getLast().fold2d(output,window));
        var descriptors = new ArrayList<>(tensors.stream().map(Tensor::descriptor).toList());
        descriptors.set(descriptors.size()-1,new TensorDescriptor(type,output,Optional.of(LayoutDescriptor.contiguous(output)),false));
        if (stridedOutput) {
            long[] strides = LayoutDescriptor.contiguous(output).strides();
            for (int i=0;i<strides.length;i++) strides[i]*=2;
            descriptors.set(descriptors.size()-1,new TensorDescriptor(type,output,
                    Optional.of(LayoutDescriptor.of(output,strides,3,true)),false));
        }
        var nodes = new ArrayList<CompiledNode>();
        for (int i=1;i<tensors.size();i++) nodes.add(new CompiledNode(new NodeId(i-1),
                tensors.get(i).provenance().orElseThrow().operation(),List.of(new ValueId(i-1)),List.of(new ValueId(i))));
        var base = CpuRecurrentScanLoweringTest.project(descriptors,nodes);
        var memory = base.memoryRequirements().stream().map(r -> new LogicalMemoryRequirement(r.valueId(),r.descriptor(),
                r.producerPartition(),r.consumerPartitions(),r.valueId().equals(new ValueId(6)))).toList();
        return new PrepareContext<>(base.partition(),nodes,base.values(),memory,Map.of(),CpuPartitionAnalysisInputs.DEFAULT);
    }

    private static long count(long size,long pad,long effective,long stride,boolean ceil) {
        long numerator=size+2*pad-effective;
        return (numerator+(ceil?stride-1:0))/stride+1;
    }

    @Test void largeInverseNumeratorsUseLongFallbackEvenForTinyHeapArrays() throws Throwable {
        for (var type:List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var c=context(type,Shape.of(2,3,1,1),new Window2dAttrs(2,2,1,1,1L<<31,1L<<31,1L<<32,1L<<32,false),0,false);
            var lowered=new CpuPartitionLowering().lower(c);
            var g=lowered.foldGeometry().orElseThrow();
            var heap=type==DataType.FLOAT32?CarrierAccess.FLOAT_ARRAY:CarrierAccess.DOUBLE_ARRAY;
            var plan=new CpuPartitionPreparer().analyze(new PrepareContext<>(c.partition(),c.nodes(),c.values(),c.memoryRequirements(),Map.of(),
                    new CpuPartitionAnalysisInputs(false,List.of(heap,heap)))).plan().units().getFirst().portablePlan();
            var generator=new CpuClassFileKernelGenerator();
            var a=generator.defineClassBytes(plan.specialization(), plan.kernelIr(), generator.generateClassBytes(plan.specialization(), plan.kernelIr()));
            Object input=type==DataType.FLOAT32?new float[6]:new double[6],output=type==DataType.FLOAT32?new float[6]:new double[6];
            fill(input,Double.NaN);fill(output,-7);
            a.entryPoint().invokeWithArguments(input,output,g.pack(new long[2],0,6),0L,6L);
            for(int i=0;i<6;i++) same(0,get(output,i),type);
        }
    }

    @Test void completeGraphPublicationOutsideAndFanoutBarriersPreserveSplitStorage() {
        var c=context(DataType.FLOAT32,Shape.of(2,3,5,7),new Window2dAttrs(2,3,1,1,0,1,1,1,false),0,false);
        for(int stage=1;stage<=5;stage++) {
            final int protectedStage=stage;
            for(boolean outside:List.of(false,true)) {
                var other=new io.github.pho001.synaptik.planning.partition.PlannedPartition(
                        io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider.CPU_BACKEND_ID,List.of(new NodeId(100)));
                var memory=c.memoryRequirements().stream().map(r->new LogicalMemoryRequirement(r.valueId(),r.descriptor(),r.producerPartition(),
                        r.valueId().value()==protectedStage&&outside?List.of(c.partition(),other):r.consumerPartitions(),
                        r.graphOutput()||r.valueId().value()==protectedStage&&!outside)).toList();
                var protectedContext=new PrepareContext<>(c.partition(),c.nodes(),c.values(),memory,Map.of(),c.backendInputs());
                assertTrue(new CpuFoldLowering().kernelBroadcast(protectedContext).isEmpty());
                var plan=new CpuPartitionPreparer().analyze(protectedContext).plan();
                assertTrue(plan.units().stream().noneMatch(u->u.portablePlan().portableKernelIr() instanceof CpuFoldIr f&&f.kernelBroadcastInput()));
                assertTrue(plan.boundaryValues().contains(new ValueId(stage)));
            }
            var descriptors=new ArrayList<>(c.values().stream().map(GraphValue::descriptor).toList());
            var d=descriptors.get(stage);
            descriptors.add(new TensorDescriptor(d.dataType(),d.shape(),Optional.of(LayoutDescriptor.contiguous(d.shape())),false)); var nodes=new ArrayList<>(c.nodes());
            nodes.add(new CompiledNode(new NodeId(50),new io.github.pho001.synaptik.model.operation.Operation(
                    io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind.NEG,
                    io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE),List.of(new ValueId(stage)),List.of(new ValueId(7))));
            var fanout=CpuRecurrentScanLoweringTest.project(descriptors,nodes);
            var plan=new CpuPartitionPreparer().analyze(fanout).plan();
            assertTrue(plan.units().stream().noneMatch(u->u.portablePlan().portableKernelIr() instanceof CpuFoldIr f&&f.kernelBroadcastInput()));
        }
    }


    @Test void interleavedUnrelatedConsumerPreservesCompactSourceBoundaryAndExactMembers() {
        var c=context(DataType.FLOAT32,Shape.of(2,3,5,7),new Window2dAttrs(2,3,1,1,0,1,1,1,false),0,false);
        var descriptors=new ArrayList<>(c.values().stream().map(GraphValue::descriptor).toList());
        descriptors.add(new TensorDescriptor(DataType.FLOAT32,descriptors.getFirst().shape(),
                Optional.of(LayoutDescriptor.contiguous(descriptors.getFirst().shape())),false));
        var nodes=new ArrayList<>(c.nodes());
        nodes.add(1,new CompiledNode(new NodeId(50),new io.github.pho001.synaptik.model.operation.Operation(
                io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind.NEG,
                io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE),List.of(new ValueId(0)),List.of(new ValueId(7))));
        var base=CpuRecurrentScanLoweringTest.project(descriptors,nodes);
        var memory=base.memoryRequirements().stream().map(r->new LogicalMemoryRequirement(r.valueId(),r.descriptor(),r.producerPartition(),r.consumerPartitions(),
                r.valueId().value()==6||r.valueId().value()==7)).toList();
        var plan=new CpuPartitionPreparer().analyze(new PrepareContext<>(base.partition(),nodes,base.values(),memory,Map.of(),CpuPartitionAnalysisInputs.DEFAULT)).plan();
        var fold=plan.units().stream().filter(u->u.portablePlan().portableKernelIr() instanceof CpuFoldIr f&&f.kernelBroadcastInput()).findFirst().orElseThrow();
        assertEquals(List.of(0,2,3,4,5,6),fold.memberNodeOrdinals());
        assertEquals(List.of(new ValueId(0),new ValueId(6)),fold.boundaryValues());
        assertEquals(Set.of(new ValueId(0),new ValueId(6),new ValueId(7)),new HashSet<>(plan.boundaryValues()));
    }

    @Test void deceptiveMappingsArithmeticPartialChainsAndUnsupportedStrategiesFailClosed() {
        var c=context(DataType.FLOAT32,Shape.of(2,3,5,7),new Window2dAttrs(2,3,1,1,0,1,1,1,false),0,false);
        assertTrue(new CpuFoldLowering().kernelBroadcast(new CpuPartitionDagDecomposer().unitContext(c,c.nodes().subList(1,6),c.backendInputs())).isEmpty());
        for(int mutation=0;mutation<6;mutation++) {
            var nodes=new ArrayList<>(c.nodes());int index=mutation==0?0:mutation==1?3:mutation==2||mutation==4?2:mutation==5?1:5;
            var old=nodes.get(index);
            var op=mutation==0?new io.github.pho001.synaptik.model.operation.Operation(AxisTransformKind.EXPAND_DIMS,new AxisTransformAttrs(1)):
                    mutation==1?new io.github.pho001.synaptik.model.operation.Operation(AxisTransformKind.PERMUTE,new PermutationAttrs(List.of(0,2,1,3))):
                    mutation==2?new io.github.pho001.synaptik.model.operation.Operation(io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind.NEG,io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE):
                    mutation==3?new io.github.pho001.synaptik.model.operation.Operation(WindowTransformKind.FOLD2D,new Fold2dAttrs(Shape.of(2,3,5,7),new Window2dAttrs(2,2,1,1,0,1,1,1,false))):
                    mutation==4?new io.github.pho001.synaptik.model.operation.Operation(io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind.CAST,
                            new io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs(DataType.FLOAT32)):
                    new io.github.pho001.synaptik.model.operation.Operation(ShapeTransformKind.EXPAND,new TargetShapeAttrs(Shape.of(3,3,6,28)));
            nodes.set(index,new CompiledNode(old.id(),op,old.inputs(),old.outputs()));
            assertTrue(new CpuFoldLowering().kernelBroadcast(new PrepareContext<>(c.partition(),nodes,c.values(),c.memoryRequirements(),Map.of(),c.backendInputs())).isEmpty());
        }
        var config=new CpuPartitionAnalysisInputs.PortableExecutionConfig(CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE,4,4,1);
        assertTrue(new CpuFoldLowering().kernelBroadcast(new PrepareContext<>(c.partition(),c.nodes(),c.values(),c.memoryRequirements(),Map.of(),new CpuPartitionAnalysisInputs(false,List.of(),config))).isEmpty());
        var singleton=context(DataType.FLOAT32,Shape.of(2,3,5,7),new Window2dAttrs(1,1,1,1,0,0,1,1,false),0,false);
        assertTrue(new CpuFoldLowering().kernelBroadcast(singleton).isEmpty());
        var values=new ArrayList<>(c.values());var source=values.getFirst();var shape=source.descriptor().shape();
        var strides=LayoutDescriptor.contiguous(shape).strides();for(int i=0;i<3;i++)strides[i]*=2;
        var noncanonical=new TensorDescriptor(DataType.FLOAT32,shape,Optional.of(LayoutDescriptor.of(shape,strides,0,true)),false);
        values.set(0,new GraphValue(source.id(),noncanonical));
        var memory=c.memoryRequirements().stream().map(r->r.valueId().equals(source.id())?new LogicalMemoryRequirement(r.valueId(),noncanonical,r.producerPartition(),r.consumerPartitions(),r.graphOutput()):r).toList();
        assertTrue(new CpuFoldLowering().kernelBroadcast(new PrepareContext<>(c.partition(),c.nodes(),values,memory,Map.of(),c.backendInputs())).isEmpty());
    }

    @Test void storedDivisionCannotBeReplacedByReciprocalOrMovedAfterFolding() throws Throwable {
        assertNotEquals(Float.floatToRawIntBits(.3f/6f),Float.floatToRawIntBits(.3f*(1f/6f)));
        assertNotEquals(Double.doubleToRawLongBits(2.1d/6d),Double.doubleToRawLongBits(2.1d*(1d/6d)));
        assertNotEquals(Float.floatToRawIntBits(.1f/6f+2.1f/6f),Float.floatToRawIntBits((.1f+2.1f)/6f));
        for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var c=context(type,Shape.of(2,3,3,4),new Window2dAttrs(2,3,1,1,0,0,1,1,false),0,false);
            var analysis=new CpuPartitionPreparer().analyze(c);
            try(var fixture=new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(analysis,1,false,false,Map.of())) {
                double represented=type==DataType.FLOAT32?.3f/6f:2.1d/6d;
                var input=fixture.segment(0);
                for(int i=0;i<24;i++) if(type==DataType.FLOAT32)input.setAtIndex(ValueLayout.JAVA_FLOAT,i,(float)represented);else input.setAtIndex(ValueLayout.JAVA_DOUBLE,i,represented);
                fixture.executable.bind(fixture.state).execute();
                var actual=fixture.segment(6);
                same(represented,type==DataType.FLOAT32?actual.getAtIndex(ValueLayout.JAVA_FLOAT,0):actual.getAtIndex(ValueLayout.JAVA_DOUBLE,0),type);
            }
        }
    }

    @Test void selectedIdentityIsGeometryFreeAndBindsBothStreamsWithoutChangingOrdinaryMode() {
        var window=new Window2dAttrs(2,3,1,1,0,1,1,1,false);
        var first=new CpuPartitionPreparer().analyze(context(DataType.FLOAT32,Shape.of(2,3,5,7),window,0,false)).plan().units().getFirst().portablePlan();
        var second=new CpuPartitionPreparer().analyze(context(DataType.FLOAT32,Shape.of(3,2,6,8),window,2,false)).plan().units().getFirst().portablePlan();
        assertArrayEquals(first.specialization().compatibilityBytes(),second.specialization().compatibilityBytes());
        assertArrayEquals(first.specialization().classIdentityBytes(),second.specialization().classIdentityBytes());
        var ir=(CpuFoldIr)first.portableKernelIr();
        assertTrue(first.kernelIr().familyIdentity().endsWith(":kernel-broadcast-input-v1"));
        assertNotEquals(ir.structuralKey(),new CpuFoldIr(ir.family(),ir.dataType(),ir.inputAccess(),ir.outputAccess(),ir.additionPolicy()).structuralKey());
    }

    @Test void coldReferenceUsesLogicalExpandedCoordinatesAndRejectsModeMismatchBeforeWrites() {
        for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var c=context(type,Shape.of(2,3,3,3),new Window2dAttrs(2,2,1,1,0,0,1,1,false),0,false);
            var lowered=new CpuPartitionLowering().lower(c);
            var ir=(CpuFoldIr)lowered.portableKernelIr();
            var geometry=lowered.foldGeometry().orElseThrow();
            Object source=type==DataType.FLOAT32?new float[24]:new double[24],actual=type==DataType.FLOAT32?new float[54]:new double[54],expected=type==DataType.FLOAT32?new float[54]:new double[54];
            for(int i=0;i<24;i++)put(source,i,(i%7-3)/10d);
            oracle(source,expected,Shape.of(2,3,3,3),(CpuFoldLowering.TwoDimensionalGeometry)geometry.mapping());
            var args=List.<io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument>of(
                    new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument.Segment(type,segment(source),24L*type.byteWidth(),true),
                    new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument.Segment(type,segment(actual),54L*type.byteWidth(),false));
            io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel.execute(ir,geometry,args,0,54);
            for(int i=0;i<54;i++)same(get(expected,i),get(actual,i),type);
            fill(actual,-7);
            var mismatch=new CpuFoldIr(ir.family(),type,ir.inputAccess(),ir.outputAccess(),ir.additionPolicy());
            assertThrows(IllegalArgumentException.class,()->io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel.execute(mismatch,geometry,args,0,54));
            for(int i=0;i<54;i++)assertEquals(-7,get(actual,i));
        }
    }

    @Test void preparedHeapNativeAndMixedCarriersUseCompactPhysicalStorage() {
        for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64))for(int mask=0;mask<4;mask++) {
            var c=context(type,Shape.of(2,3,3,3),new Window2dAttrs(2,2,1,1,0,0,1,1,false),0,false);
            var heap=type==DataType.FLOAT32?CarrierAccess.FLOAT_ARRAY:CarrierAccess.DOUBLE_ARRAY;
            var analysis=new CpuPartitionPreparer().analyze(new PrepareContext<>(c.partition(),c.nodes(),c.values(),c.memoryRequirements(),Map.of(),
                    new CpuPartitionAnalysisInputs(false,List.of((mask&1)==0?heap:CarrierAccess.MEMORY_SEGMENT,(mask&2)==0?heap:CarrierAccess.MEMORY_SEGMENT))));
            try(var arena=Arena.ofShared()) {
                MemorySegment in=(mask&1)==0?segment(type==DataType.FLOAT32?new float[24]:new double[24]):arena.allocate(24L*type.byteWidth(),type.byteWidth());
                MemorySegment out=(mask&2)==0?segment(type==DataType.FLOAT32?new float[54]:new double[54]):arena.allocate(54L*type.byteWidth(),type.byteWidth());
                try(var f=new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(analysis,0,false,false,Map.of(new ValueId(0),in,new ValueId(6),out))) {
                    for(int run=1;run<=2;run++) {
                        for(int i=0;i<24;i++)if(type==DataType.FLOAT32)in.setAtIndex(ValueLayout.JAVA_FLOAT,i,run*(i%7-3));else in.setAtIndex(ValueLayout.JAVA_DOUBLE,i,run*(i%7-3));
                        f.executable.bind(f.state).execute();
                        Object expected=type==DataType.FLOAT32?new float[54]:new double[54];
                        oracle(type==DataType.FLOAT32?in.toArray(ValueLayout.JAVA_FLOAT):in.toArray(ValueLayout.JAVA_DOUBLE),expected,Shape.of(2,3,3,3),
                                (CpuFoldLowering.TwoDimensionalGeometry)new CpuPartitionLowering().lower(c).foldGeometry().orElseThrow().mapping());
                        for(int i=0;i<54;i++)same(get(expected,i),type==DataType.FLOAT32?out.getAtIndex(ValueLayout.JAVA_FLOAT,i):out.getAtIndex(ValueLayout.JAVA_DOUBLE,i),type);
                    }
                }
            }
        }
    }

    @Test void actualCompilerAveragePoolBackwardKeepsStoredDivisionAndTypedOneCount() {
        try(var cpu=io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.open()) {
            for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64)) {
                var input=TensorFactory.create(new TensorDescriptor(type,Shape.of(2,3,5,7),
                        Optional.of(LayoutDescriptor.contiguous(Shape.of(2,3,5,7))),true));
                var output=input.averagePool2d(new io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs(2,3,2,2,1,1,1,1,true));
                var seed=TensorFactory.create(new TensorDescriptor(type,output.descriptor().shape(),Optional.of(LayoutDescriptor.contiguous(output.descriptor().shape())),false));
                var request=new io.github.pho001.synaptik.compiler.FunctionalGradientRequest(List.of(new io.github.pho001.synaptik.compiler.FunctionalGradientRequest.Stage(
                        List.of(new io.github.pho001.synaptik.compiler.FunctionalGradientRequest.ForwardTensorReference(output)),List.of(Optional.of(seed)),List.of(input),false,
                        io.github.pho001.synaptik.compiler.FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
                for(boolean optimized:List.of(false,true)) {
                    var artifacts=io.github.pho001.synaptik.compiler.GraphCompilationPort.compile(
                            io.github.pho001.synaptik.config.compile.CompileMode.FORWARD_AND_BACKWARD,List.of(output),Optional.of(request),
                            optimized?io.github.pho001.synaptik.config.compile.GraphOptimizationConfig.standard():io.github.pho001.synaptik.config.compile.GraphOptimizationConfig.disabled(),
                            io.github.pho001.synaptik.config.compile.BackendIntent.unconstrained(),io.github.pho001.synaptik.config.compile.PartitionScoringConfig.neutral(),
                            List.of(cpu.capabilityProvider()),List.of(cpu.availabilitySnapshot()));
                    int folds=0,counts=0;
                    for(var partition:artifacts.partitions()) {
                        var c=io.github.pho001.synaptik.prepare.GraphPreparation.project(artifacts,partition,CpuPartitionAnalysisInputs.DEFAULT);
                        var plan=new CpuPartitionPreparer().analyze(c).plan();
                        for(var unit:plan.units()) {
                            if(unit.portablePlan().portableKernelIr() instanceof CpuConstantWriteIr) counts++;
                            if(unit.portablePlan().portableKernelIr() instanceof CpuFoldIr f&&f.kernelBroadcastInput()) {
                                folds++;var source=unit.boundaryValues().getFirst();
                                var producer=c.partitionDag().producer(source).orElseThrow().node();
                                assertEquals(io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.DIV,producer.operation().kind());
                                assertFalse(unit.memberNodeOrdinals().contains(c.nodes().indexOf(producer)));
                                assertTrue(plan.boundaryValues().contains(source));
                                assertTrue(unit.runtimeFacts().workspaceDeclaration().isEmpty());
                            }
                        }
                    }
                    assertEquals(1,folds);assertTrue(counts>0);
                }
            }
        }
    }

    @Test void compactResourcesAndAllCarriersMatchIndependentJavaAndOrdinarySplit() throws Throwable {
        var windows = List.of(new Window2dAttrs(2,3,1,1,0,1,1,1,false),
                new Window2dAttrs(2,3,2,2,2,2,2,1,true),
                new Window2dAttrs(2,2,4,3,1,1,1,1,false));
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (var window : windows)
            for (int carriers=0;carriers<4;carriers++) check(type,Shape.of(2,3,5,7),window,carriers,false);
    }

    @Test void offsetsStridedOutputPartialAndEmptyRangesPreserveUntouchedCells() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64))
            for (int carriers=0;carriers<4;carriers++) check(type,Shape.of(2,3,5,7),
                    new Window2dAttrs(2,3,2,2,2,2,2,1,true),carriers,true);
        for (var shape : List.of(Shape.of(0,3,3,3),Shape.of(2,0,3,3),Shape.of(2,3,0,3),Shape.of(2,3,3,0)))
            for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) check(type,shape,
                    new Window2dAttrs(2,2,1,1,1,1,1,1,false),0,false);
    }

    private static void check(DataType type,Shape output,Window2dAttrs window,int carriers,boolean strided) throws Throwable {
        var context=context(type,output,window,strided?2:0,strided);
        for (var node:context.nodes()) io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertCurrentSupport(new io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider(), new io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery(node.operation(),
            node.inputs().stream().map(id->context.values().get((int)id.value()).descriptor()).toList(),
            node.outputs().stream().map(id->context.values().get((int)id.value()).descriptor()).toList()), node+" values="+context.values());
        var lowered=new CpuPartitionLowering().lower(context);
        assertTrue(((CpuFoldIr)lowered.portableKernelIr()).kernelBroadcastInput());
        assertEquals(5,lowered.virtualValues().size());
        var read=type==DataType.FLOAT32?CarrierAccess.FLOAT_ARRAY:CarrierAccess.DOUBLE_ARRAY;
        var inputs=new CpuPartitionAnalysisInputs(false,List.of((carriers&1)==0?read:CarrierAccess.MEMORY_SEGMENT,
                (carriers&2)==0?read:CarrierAccess.MEMORY_SEGMENT));
        var prepared=new CpuPartitionPreparer().analyze(new PrepareContext<>(context.partition(),context.nodes(),context.values(),
                context.memoryRequirements(),Map.of(),inputs));
        assertEquals(1,prepared.plan().units().size());
        assertEquals(2,prepared.requirements().size());
        assertTrue(prepared.plan().workspaceDeclaration().isEmpty()); assertTrue(prepared.plan().materialization().isEmpty());
        assertEquals(List.of(new ValueId(0),new ValueId(6)),prepared.plan().units().getFirst().boundaryValues());
        var geometry=lowered.foldGeometry().orElseThrow(); var two=(CpuFoldLowering.TwoDimensionalGeometry)geometry.mapping();
        int inputSpan=Math.toIntExact(lowered.referencedElementSpans().getFirst());
        int outputSpan=Math.toIntExact(lowered.referencedElementSpans().getLast());
        int count=Math.toIntExact(lowered.elementCount());
        assertEquals(inputSpan*type.byteWidth(),prepared.plan().bufferDeclarations().getFirst().byteSize());
        int p=(int)(two.outputColumnsHeight()*two.outputColumnsWidth()); int nc=(int)(output.toLongArray()[0]*output.toLongArray()[1]);
        Object source=type==DataType.FLOAT32?new float[inputSpan]:new double[inputSpan];
        Object actual=type==DataType.FLOAT32?new float[outputSpan]:new double[outputSpan];
        Object expected=type==DataType.FLOAT32?new float[count]:new double[count];
        double[] samples={0.0,-0.0,1,-1,1e30,-1e30,3,Double.POSITIVE_INFINITY,Double.NEGATIVE_INFINITY,Double.NaN,
                type==DataType.FLOAT32?Float.MIN_VALUE:Double.MIN_VALUE,
                type==DataType.FLOAT32?Float.MAX_VALUE:Double.MAX_VALUE,0x1.000002p0,0x1p-24};
        int offset=(int)geometry.input().offset();
        for(int i=0;i<nc*p;i++) put(source,offset+i,samples[i%samples.length]);
        var dense=copy(source,offset,nc*p);
        oracle(dense,expected,output,two);
        var route=prepared.plan().units().getFirst().portablePlan(); var generator=new CpuClassFileKernelGenerator();
        byte[] bytes=generator.generateClassBytes(route.specialization(),route.kernelIr());
        inspect(bytes,type);
        var artifact=generator.defineClassBytes(route.specialization(), route.kernelIr(), bytes);
        Object in=(carriers&1)==0?source:segment(source),out=(carriers&2)==0?actual:segment(actual);
        long start=strided?1:0,end=strided?Math.max(1,count-1):count;
        fill(actual,-7); artifact.entryPoint().invokeWithArguments(in,out,geometry.pack(new long[2],start,end),start,end);
        long[] strides=geometry.output().strides(),extents=geometry.outputExtents();
        var touched=new HashSet<Integer>();
        for(int i=(int)start;i<end;i++) {
            long remainder=i,address=geometry.output().offset();
            for(int axis=3;axis>=0;axis--) {address+=(remainder%extents[axis])*strides[axis];remainder/=extents[axis];}
            touched.add((int)address); same(get(expected,i),get(actual,(int)address),type);
        }
        for(int i=0;i<outputSpan;i++) if(!touched.contains(i)) assertEquals(-7,get(actual,i));
        Object unchanged=copy(actual,0,outputSpan);
        artifact.entryPoint().invokeWithArguments(in,out,geometry.pack(new long[2],start,start),start,start);
        for(int i=0;i<outputSpan;i++) same(get(unchanged,i),get(actual,i),type);
        // Ordinary split consumes exactly the raw-repeated same-type source (not a numerical low oracle).
        var split=new CpuFoldLowering().lower(new CpuPartitionDagDecomposer().unitContext(context,List.of(context.nodes().getLast()),CpuPartitionAnalysisInputs.DEFAULT));
        var splitGeometry=split.foldGeometry().orElseThrow();
        Object expanded=type==DataType.FLOAT32?new float[nc*p*(int)(two.kernelHeight()*two.kernelWidth())]
                :new double[nc*p*(int)(two.kernelHeight()*two.kernelWidth())];
        int k=(int)(two.kernelHeight()*two.kernelWidth());
        for(int n=0;n<nc;n++) for(int q=0;q<k;q++) for(int pos=0;pos<p;pos++) put(expanded,(n*k+q)*p+pos,get(dense,n*p+pos));
        Object splitResult=type==DataType.FLOAT32?new float[outputSpan]:new double[outputSpan];
        var splitPlan=new CpuPartitionPreparer().analyze(new CpuPartitionDagDecomposer().unitContext(context,
                List.of(context.nodes().getLast()),CpuPartitionAnalysisInputs.DEFAULT)).plan().units().getFirst().portablePlan();
        var splitArtifact=generator.defineClassBytes(splitPlan.specialization(), splitPlan.kernelIr(), generator.generateClassBytes(splitPlan.specialization(), splitPlan.kernelIr()));
        splitArtifact.entryPoint().invokeWithArguments(segment(expanded),segment(splitResult),splitGeometry.pack(new long[2],start,end),start,end);
        for(int address:touched) same(get(actual,address),get(splitResult,address),type);
    }

    private static void inspect(byte[] bytes,DataType type) throws java.io.IOException {
        var code=ClassFile.of().parse(bytes).methods().getFirst().code().orElseThrow();
        var instructions=code.elementStream().filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        assertTrue(instructions.stream().anyMatch(i->i.opcode()==Opcode.IDIV));
        assertTrue(instructions.stream().anyMatch(i->i.opcode()==Opcode.LDIV));
        assertEquals(2,instructions.stream().filter(i->i.opcode()==(type==DataType.FLOAT32?Opcode.FADD:Opcode.DADD)).count());
        assertFalse(instructions.stream().anyMatch(i->i instanceof NewObjectInstruction||i instanceof NewPrimitiveArrayInstruction
                ||i instanceof NewReferenceArrayInstruction||i instanceof NewMultiArrayInstruction||i instanceof TypeCheckInstruction));
        assertTrue(instructions.stream().filter(InvokeInstruction.class::isInstance).map(InvokeInstruction.class::cast)
                .allMatch(i->i.owner().asInternalName().equals("java/lang/foreign/MemorySegment")
                    && (i.name().stringValue().equals("get")||i.name().stringValue().equals("set"))));
        String root=System.getenv("SYNAPTIK_CPU0050_STRUCTURE");
        if(root!=null) {
            var dir=java.nio.file.Path.of(root);java.nio.file.Files.createDirectories(dir);
            String name=ClassFile.of().parse(bytes).thisClass().asInternalName().replace('/','_');
            java.nio.file.Files.write(dir.resolve(name+".class"),bytes);
            try(var oracle=CpuBroadcastFoldCleanJavaOracle.class.getResourceAsStream("CpuBroadcastFoldCleanJavaOracle.class")) {
                java.nio.file.Files.write(dir.resolve("CpuBroadcastFoldCleanJavaOracle.class"),Objects.requireNonNull(oracle).readAllBytes());
            }
        }
    }

    /** Sole opt-in diagnostic; no timing participates in production decisions or test thresholds. */
    @Test @org.junit.jupiter.api.Timeout(60)
    void boundedGeneratedVersusDirectDiagnostic() throws Throwable {
        org.junit.jupiter.api.Assumptions.assumeTrue("true".equals(System.getenv("SYNAPTIK_CPU0050_DIAGNOSTIC")));
        long deadline=System.nanoTime()+60_000_000_000L;
        for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            Shape shape=Shape.of(2,3,16,17);var window=new Window2dAttrs(3,3,2,2,1,1,1,1,false);
            long planningStart=System.nanoTime();var c=context(type,shape,window,0,false);
            var heap=type==DataType.FLOAT32?CarrierAccess.FLOAT_ARRAY:CarrierAccess.DOUBLE_ARRAY;
            var plan=new CpuPartitionPreparer().analyze(new PrepareContext<>(c.partition(),c.nodes(),c.values(),c.memoryRequirements(),Map.of(),new CpuPartitionAnalysisInputs(false,List.of(heap,heap)))).plan();
            long planningNanos=System.nanoTime()-planningStart;var route=plan.units().getFirst().portablePlan();
            var geometry=plan.foldGeometry().orElseThrow();assertEquals(432,geometry.input().extents()[0]*geometry.input().extents()[1]*geometry.input().extents()[2]);
            var generator=new CpuClassFileKernelGenerator();long generationStart=System.nanoTime();
            var artifact=generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
            long generationNanos=System.nanoTime()-generationStart;
            Object source=type==DataType.FLOAT32?new float[432]:new double[432],generated=type==DataType.FLOAT32?new float[1632]:new double[1632],direct=type==DataType.FLOAT32?new float[1632]:new double[1632];
            for(int i=0;i<432;i++)put(source,i,(i%29-14)/7d);
            long bindingStart=System.nanoTime();
            var generatedCall=java.lang.invoke.MethodHandles.insertArguments(artifact.entryPoint().bindTo(source).bindTo(generated),0,geometry.pack(new long[2],0,1632),0L,1632L);
            long generatedBindingNanos=System.nanoTime()-bindingStart;
            Class<?>[] parameterTypes=new Class<?>[16];Arrays.fill(parameterTypes,int.class);parameterTypes[0]=source.getClass();parameterTypes[1]=direct.getClass();
            bindingStart=System.nanoTime();
            var directCall=java.lang.invoke.MethodHandles.lookup().findStatic(CpuBroadcastFoldCleanJavaOracle.class,"fold",java.lang.invoke.MethodType.methodType(void.class,parameterTypes)).bindTo(source).bindTo(direct);
            directCall=java.lang.invoke.MethodHandles.insertArguments(directCall,0,2,3,16,17,3,3,2,2,1,1,1,1,8,9);
            long directBindingNanos=System.nanoTime()-bindingStart;
            long entryStart=System.nanoTime();generatedCall.invokeExact();long generatedEntryNanos=System.nanoTime()-entryStart;
            entryStart=System.nanoTime();directCall.invokeExact();long directEntryNanos=System.nanoTime()-entryStart;
            for(int i=0;i<1632;i++)same(get(direct,i),get(generated,i),type);
            System.out.println("CPU0050 "+type+" planningNs="+planningNanos+" generationNs="+generationNanos+" generatedBindingNs="+generatedBindingNanos+" directBindingNs="+directBindingNanos+" coldEntryGeneratedNs="+generatedEntryNanos+" coldEntryDirectNs="+directEntryNanos);
            for(int warm=0;warm<6;warm++) {
                for(int call=0;call<25;call++)generatedCall.invokeExact();
                for(int call=0;call<25;call++)directCall.invokeExact();
                assertTrue(System.nanoTime()<deadline,"60-second body cap");
            }
            long[] generatedSamples=new long[9],directSamples=new long[9];
            for(int sample=0;sample<9;sample++) {
                for(int arm=0;arm<2;arm++) {
                    boolean generate=((sample+arm)&1)==0;long before=System.nanoTime();
                    if(generate)for(int call=0;call<25;call++)generatedCall.invokeExact();
                    else for(int call=0;call<25;call++)directCall.invokeExact();
                    (generate?generatedSamples:directSamples)[sample]=System.nanoTime()-before;
                }
                System.out.println("CPU0050 "+type+" pair="+sample+" generatedNs="+generatedSamples[sample]+" directNs="+directSamples[sample]+" order="+(sample%2==0?"GD":"DG"));
                for(int i=0;i<1632;i++)same(get(direct,i),get(generated,i),type);
                assertTrue(System.nanoTime()<deadline,"60-second body cap");
            }
            long[] gs=generatedSamples.clone(),ds=directSamples.clone();Arrays.sort(gs);Arrays.sort(ds);
            System.out.println("CPU0050 "+type+" generatedMinMedianMax="+gs[0]+","+gs[4]+","+gs[8]+" directMinMedianMax="+ds[0]+","+ds[4]+","+ds[8]+" nsPer25Calls; no subtraction, threshold, retry or steady-state JIT claim");
        }
    }

    static void oracle(Object input,Object output,Shape shape,CpuFoldLowering.TwoDimensionalGeometry m) {
        int[] s=Arrays.stream(shape.toLongArray()).mapToInt(Math::toIntExact).toArray();
        if(input instanceof float[] f) CpuBroadcastFoldCleanJavaOracle.fold(f,(float[])output,s[0],s[1],s[2],s[3],(int)m.kernelHeight(),(int)m.kernelWidth(),
                (int)m.strideHeight(),(int)m.strideWidth(),(int)m.paddingHeight(),(int)m.paddingWidth(),(int)m.dilationHeight(),(int)m.dilationWidth(),(int)m.outputColumnsHeight(),(int)m.outputColumnsWidth());
        else CpuBroadcastFoldCleanJavaOracle.fold((double[])input,(double[])output,s[0],s[1],s[2],s[3],(int)m.kernelHeight(),(int)m.kernelWidth(),
                (int)m.strideHeight(),(int)m.strideWidth(),(int)m.paddingHeight(),(int)m.paddingWidth(),(int)m.dilationHeight(),(int)m.dilationWidth(),(int)m.outputColumnsHeight(),(int)m.outputColumnsWidth());
    }
    static Object copy(Object a,int start,int count) {return a instanceof float[] f?Arrays.copyOfRange(f,start,start+count):Arrays.copyOfRange((double[])a,start,start+count);}
    static void fill(Object a,double value) {if(a instanceof float[] f) Arrays.fill(f,(float)value);else Arrays.fill((double[])a,value);}
    static void put(Object a,int i,double v) {if(a instanceof float[] f)f[i]=(float)v;else ((double[])a)[i]=v;}
    static double get(Object a,int i) {return a instanceof float[] f?f[i]:((double[])a)[i];}
    static MemorySegment segment(Object a) {return a instanceof float[] f?MemorySegment.ofArray(f):MemorySegment.ofArray((double[])a);}
    static void same(double expected,double actual,DataType type) {
        if(Double.isNaN(expected)) assertTrue(Double.isNaN(actual));
        else if(type==DataType.FLOAT32) assertEquals(Float.floatToRawIntBits((float)expected),Float.floatToRawIntBits((float)actual));
        else assertEquals(Double.doubleToRawLongBits(expected),Double.doubleToRawLongBits(actual));
    }
}
