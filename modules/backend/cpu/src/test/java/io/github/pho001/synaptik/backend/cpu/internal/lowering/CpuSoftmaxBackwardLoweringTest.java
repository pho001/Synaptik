package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Exact producer/use matching, private-store proof and selected scalar resources. */
public class CpuSoftmaxBackwardLoweringTest {
    @Test void actualCompilerAdWithExplicitNonuniformCotangentSelectsBothPatternsAndRetainsForwardPublication() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean log : List.of(false,true)) {
            var shape = Shape.of(2,3);
            var x = io.github.pho001.synaptik.model.tensor.TensorFactory.create(new TensorDescriptor(type,shape,Optional.of(LayoutDescriptor.contiguous(shape)),true));
            var seed = io.github.pho001.synaptik.model.tensor.TensorFactory.create(new TensorDescriptor(type,shape,Optional.of(LayoutDescriptor.contiguous(shape)),false));
            var y = log ? x.logSoftmax(1) : x.softmax(1);
            var request = new io.github.pho001.synaptik.compiler.FunctionalGradientRequest(List.of(new io.github.pho001.synaptik.compiler.FunctionalGradientRequest.Stage(
                    List.of(new io.github.pho001.synaptik.compiler.FunctionalGradientRequest.ForwardTensorReference(y)),List.of(Optional.of(seed)),List.of(x),false,
                    io.github.pho001.synaptik.compiler.FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
            try (var cpu = io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.open()) {
                var artifacts = io.github.pho001.synaptik.compiler.GraphCompilationPort.compile(
                        io.github.pho001.synaptik.config.compile.CompileMode.FORWARD_AND_BACKWARD,List.of(y),Optional.of(request),
                        io.github.pho001.synaptik.config.compile.GraphOptimizationConfig.disabled(),io.github.pho001.synaptik.config.compile.BackendIntent.unconstrained(),
                        io.github.pho001.synaptik.config.compile.PartitionScoringConfig.neutral(),List.of(cpu.capabilityProvider()),List.of(cpu.availabilitySnapshot()));
                int selected = 0;
                for (var p : artifacts.partitions()) {
                    var c = io.github.pho001.synaptik.prepare.GraphPreparation.project(artifacts,p,CpuPartitionAnalysisInputs.DEFAULT);
                    var plan = new CpuPartitionPreparer().analyze(c).plan();
                    for (var unit : plan.units()) if (unit.portablePlan().portableKernelIr() instanceof CpuSoftmaxBackwardIr backward) {
                        assertEquals(log,backward.logarithmic()); assertEquals(4,unit.memberNodeOrdinals().size());
                        var savedId = unit.boundaryValues().get(backward.savedInput());
                        assertTrue(c.memoryRequirements().stream().anyMatch(r -> r.valueId().equals(savedId) && r.graphOutput()));
                        selected++;
                    }
                }
                assertEquals(1,selected,"actual Compiler-generated derivative must select exactly once");
            }
        }
    }

    @Test void vectorAndWorkerPreferencesCannotExpandTheSelectedScalarSingleRange() {
        var c = context(DataType.FLOAT64,Shape.of(32,128),false,List.of());
        var inputs = new CpuPartitionAnalysisInputs(false,List.of(),new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE,8,8,1));
        var plan = new CpuPartitionPreparer().analyze(new PrepareContext<>(c.partition(),c.nodes(),c.values(),c.memoryRequirements(),Map.of(),inputs)).plan();
        assertEquals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,plan.executionStrategy());
        assertEquals(1,plan.selectedRangeCount()); assertEquals(0,plan.vectorSpeciesBitSize());
    }

    @Test void integerByteAndLogicalAddressLimitsRejectWithoutEnumeratingTheDomain() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean log : List.of(false,true)) {
            assertRejected(context(type,Shape.of(1,Integer.MAX_VALUE/type.byteWidth()+1L),log,List.of()));
            assertRejected(context(type,Shape.of(3,1_000_000_000),log,List.of()));
        }
    }
    @Test void bothPatternsRanksAndTypesHaveOnlyPhysicalReadsAndFullGradientWrite() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean log : List.of(false,true))
            for (var shape : List.of(Shape.of(7),Shape.of(3,7),Shape.of(0),Shape.of(0,7),Shape.of(3,0),Shape.of(3,1))) {
                var c = context(type,shape,log,List.of());
                var lowered = new CpuPartitionLowering().lower(c);
                var ir = assertInstanceOf(CpuSoftmaxBackwardIr.class,lowered.portableKernelIr());
                assertEquals(log,ir.logarithmic());
                assertEquals(List.of(new ValueId(0),new ValueId(1),new ValueId(5)),lowered.boundaryValues());
                assertEquals(List.of(new ValueId(2),new ValueId(3),new ValueId(4)),lowered.virtualValues());
                assertEquals(shape,Shape.of(lowered.accessBindings().getLast().extents().stream().mapToLong(Long::longValue).toArray()));
                var plan = new CpuPartitionPreparer().analyze(c).plan();
                assertEquals(1,plan.units().size()); assertEquals(3,plan.bufferDeclarations().size());
                assertEquals(1,plan.selectedRangeCount()); assertEquals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,plan.executionStrategy());
                assertEquals(0,plan.vectorSpeciesBitSize()); assertTrue(plan.materialization().isEmpty());
                assertEquals(shape.rank() == 1 ? 1 : shape.toLongArray()[0],plan.elementCount());
                assertEquals(lowered.aggregateGeometry().orElseThrow().workspaceBytes(1),plan.workspaceDeclaration().map(w -> w.byteSize()).orElse(0L));
                assertEquals((long)type.byteWidth()*shape.knownElementCount().orElseThrow(),plan.bufferDeclarations().getLast().byteSize());
            }
    }

    @Test void exactOperandOrderAndKeptTrailingAxisAreMandatory() {
        for (boolean log : List.of(false,true)) {
            var c = context(DataType.FLOAT32,Shape.of(2,3),log,List.of());
            for (int index = 0; index < 4; index++) {
                var nodes = new ArrayList<>(c.nodes()); var old = nodes.get(index);
                if (old.inputs().size() == 2) {
                    nodes.set(index,new CompiledNode(old.id(),old.operation(),old.inputs().reversed(),old.outputs()));
                    assertRejected(rebuild(c,nodes,c.values(),Set.of()));
                }
            }
            for (var attrs : List.<OperationAttrs>of(new AxisReductionAttrs(0,true),new AxisReductionAttrs(1,false),
                    new MultiAxisReductionAttrs(List.of(1),true),NoOperationAttrs.INSTANCE)) {
                var nodes = new ArrayList<>(c.nodes()); var sum = nodes.get(1);
                nodes.set(1,new CompiledNode(sum.id(),new Operation(AggregateReductionKind.SUM,attrs),sum.inputs(),sum.outputs()));
                assertRejected(rebuild(c,nodes,c.values(),Set.of()));
            }
            var nodes = new ArrayList<>(c.nodes()); var sum = nodes.get(1);
            nodes.set(1,new CompiledNode(sum.id(),sum.operation(),List.of(new ValueId(log ? 0 : 1)),sum.outputs()));
            assertRejected(rebuild(c,nodes,c.values(),Set.of()));
        }
        assertRejected(context(DataType.FLOAT32,Shape.of(2,3,4),false,List.of()));
        for (var type : List.of(DataType.BFLOAT16,DataType.FLOAT16,DataType.INT32))
            assertRejected(context(type,Shape.of(2,3),false,List.of()));
    }

    @Test void everyPublishedSavedFanOutAndOutsideIntermediateRemainsMaterialized() {
        for (boolean log : List.of(false,true)) for (int protectedId : List.of(2,3,4)) {
            var c = context(DataType.FLOAT64,Shape.of(2,3),log,List.of());
            var published = rebuild(c,c.nodes(),c.values(),Set.of(protectedId));
            assertRejected(published);
            var plan = new CpuPartitionPreparer().analyze(published).plan();
            assertTrue(plan.bufferDeclarations().stream().anyMatch(b -> b.valueId().equals(new ValueId(protectedId))));
            var memory = new ArrayList<>(c.memoryRequirements()); var old = memory.get(protectedId);
            memory.set(protectedId,new LogicalMemoryRequirement(old.valueId(),old.descriptor(),old.producerPartition(),
                    List.of(c.partition(),new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,List.of(new NodeId(99)))),false));
            assertRejected(new PrepareContext<>(c.partition(),c.nodes(),c.values(),memory,Map.of(),c.backendInputs()));
            var nodes = new ArrayList<>(c.nodes()); var values = new ArrayList<>(c.values());
            nodes.add(node(4,UnaryElementwiseKind.NEG,List.of(protectedId),6));
            values.add(new GraphValue(new ValueId(6),values.get(protectedId).descriptor()));
            var fanout = rebuild(c,nodes,values,Set.of(6));
            var split = new CpuPartitionPreparer().analyze(fanout).plan();
            assertTrue(split.units().stream().noneMatch(u -> u.portablePlan().portableKernelIr() instanceof CpuSoftmaxBackwardIr));
            assertTrue(split.bufferDeclarations().stream().anyMatch(b -> b.valueId().equals(new ValueId(protectedId))));
        }
    }

    @Test void unrelatedInterleavingIsAdmittedButDependentOutsidePathIsNot() {
        var c = context(DataType.FLOAT32,Shape.of(2,3),false,List.of());
        var nodes = new ArrayList<>(c.nodes()); var values = new ArrayList<>(c.values());
        nodes.add(1,node(10,UnaryElementwiseKind.NEG,List.of(0),6));
        values.add(new GraphValue(new ValueId(6),values.getFirst().descriptor()));
        var plan = new CpuPartitionPreparer().analyze(rebuild(c,nodes,values,Set.of(6))).plan();
        var selected = plan.units().stream().filter(u -> u.portablePlan().portableKernelIr() instanceof CpuSoftmaxBackwardIr).findFirst().orElseThrow();
        assertEquals(List.of(0,2,3,4),selected.memberNodeOrdinals());
        // An outside NEG after the first product feeding a supposedly external cotangent
        // makes contraction cyclic; both topology and the changed operands reject it.
        nodes.set(1,node(10,UnaryElementwiseKind.NEG,List.of(2),6));
        var last = nodes.getLast(); nodes.set(nodes.size()-1,new CompiledNode(last.id(),last.operation(),List.of(new ValueId(6),new ValueId(4)),last.outputs()));
        var nonconvex = new CpuPartitionPreparer().analyze(rebuild(c,nodes,values,Set.of(6))).plan();
        assertTrue(nonconvex.units().stream().noneMatch(u -> u.portablePlan().portableKernelIr() instanceof CpuSoftmaxBackwardIr));
    }

    @Test void onlyCanonicalSavedWriteAndCanonicalOrScalarColumnCotangentAreAdmitted() {
        for (boolean log : List.of(false,true)) {
            var c = context(DataType.FLOAT32,Shape.of(2,3),log,List.of());
            for (long[] strides : List.of(new long[]{1,0},new long[]{0,0})) {
                var values = new ArrayList<>(c.values());
                values.set(1,value(1,DataType.FLOAT32,Shape.of(2,3),strides,2));
                assertTrue(fuse(rebuild(c,c.nodes(),values,Set.of())).isPresent());
            }
            for (int id : List.of(0,1,5)) {
                var values = new ArrayList<>(c.values());
                values.set(id,value(id,DataType.FLOAT32,Shape.of(2,3),new long[]{7,2},0));
                assertRejected(rebuild(c,c.nodes(),values,Set.of()));
            }
            var values = new ArrayList<>(c.values()); values.set(0,value(0,DataType.FLOAT64,Shape.of(2,3),new long[]{3,1},0));
            assertRejected(rebuild(c,c.nodes(),values,Set.of()));
            values = new ArrayList<>(c.values()); values.set(0,value(0,DataType.FLOAT32,Shape.of(2,3),new long[]{3,1},Integer.MAX_VALUE));
            assertRejected(rebuild(c,c.nodes(),values,Set.of()));
        }
    }

    private static void assertRejected(PrepareContext<CpuPartitionAnalysisInputs> c) { assertTrue(fuse(c).isEmpty()); }
    private static Optional<CpuPartitionLowering.LoweredPartition> fuse(PrepareContext<CpuPartitionAnalysisInputs> c) {
        return new CpuAggregateLowering().fuseSoftmaxBackward(c,new CpuPartitionLowering());
    }

    /** Creates exact four-node test patterns with no protected intermediate.
     * @param type homogeneous represented type
     * @param shape static full gradient shape
     * @param log whether to use log-softmax's captured pattern
     * @param carriers physical read/read/write tuple, or empty for segments
     * @return complete CPU projection with original IDs and private-use obligations */
    public static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type,Shape shape,boolean log,List<CarrierAccess> carriers) {
        var nodes = List.of(node(0,log ? UnaryElementwiseKind.EXP : BinaryArithmeticKind.MUL,log ? List.of(0) : List.of(1,0),2),
                new CompiledNode(new NodeId(1),new Operation(AggregateReductionKind.SUM,new AxisReductionAttrs(shape.rank()-1,true)),
                    List.of(new ValueId(log ? 1 : 2)),List.of(new ValueId(3))),
                node(2,log ? BinaryArithmeticKind.MUL : BinaryArithmeticKind.SUB,log ? List.of(2,3) : List.of(1,3),4),
                node(3,log ? BinaryArithmeticKind.SUB : BinaryArithmeticKind.MUL,log ? List.of(1,4) : List.of(0,4),5));
        long[] reduced = shape.toLongArray(); reduced[reduced.length-1] = 1;
        var values = new ArrayList<GraphValue>();
        for (int i = 0; i < 6; i++) { var s = i == 3 ? Shape.of(reduced) : shape;
            values.add(value(i,type,s,LayoutDescriptor.contiguous(s).strides(),0)); }
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,nodes.stream().map(CompiledNode::id).toList());
        var c = new PrepareContext<>(partition,nodes,values,memory(partition,nodes,values,Set.of()),Map.of(),new CpuPartitionAnalysisInputs(false,carriers));
        return c;
    }

    /** Rebuilds exact memory/publication obligations after test-local node/descriptor changes.
     * @param c original backend analysis configuration
     * @param nodes new original-order members
     * @param values new complete descriptors
     * @param publications extra protected outputs, including saved/output obligations
     * @return complete immutable projection */
    public static PrepareContext<CpuPartitionAnalysisInputs> rebuild(PrepareContext<CpuPartitionAnalysisInputs> c,
            List<CompiledNode> nodes,List<GraphValue> values,Set<Integer> publications) {
        var p = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,nodes.stream().map(CompiledNode::id).toList());
        return new PrepareContext<>(p,nodes,values,memory(p,nodes,values,publications),Map.of(),c.backendInputs());
    }
    private static List<LogicalMemoryRequirement> memory(PlannedPartition p,List<CompiledNode> nodes,List<GraphValue> values,Set<Integer> publications) {
        return values.stream().map(v -> new LogicalMemoryRequirement(v.id(),v.descriptor(),nodes.stream().anyMatch(n -> n.outputs().contains(v.id())) ? Optional.of(p) : Optional.empty(),
                nodes.stream().anyMatch(n -> n.inputs().contains(v.id())) ? List.of(p) : List.of(),v.id().equals(new ValueId(5)) || publications.contains((int)v.id().value()))).toList();
    }
    private static GraphValue value(int id,DataType type,Shape shape,long[] strides,long offset) {
        return new GraphValue(new ValueId(id),new TensorDescriptor(type,shape,Optional.of(LayoutDescriptor.of(shape,strides,offset,
                java.util.Arrays.stream(strides).anyMatch(s -> s == 0))),false));
    }
    private static CompiledNode node(int id,OperationKind kind,List<Integer> in,int out) {
        return new CompiledNode(new NodeId(id),new Operation(kind,NoOperationAttrs.INSTANCE),in.stream().map(ValueId::new).toList(),List.of(new ValueId(out)));
    }
}
