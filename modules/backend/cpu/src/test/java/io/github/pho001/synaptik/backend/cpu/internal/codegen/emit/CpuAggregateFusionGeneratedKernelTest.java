package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.*;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
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
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Untimed semantic, selected-resource and generated-structure proof for ordinary fusion. */
public class CpuAggregateFusionGeneratedKernelTest {
    @Test void controlledProducerTerminalUsesRemovedValueProvenanceAndRepresentedSites()
            throws Throwable {
        var c = context(DataType.FLOAT32, Shape.of(3, 3), NoOperationAttrs.INSTANCE,
                Shape.scalar(), true, 0, false, Set.of(), List.of());
        var values = new ArrayList<>(c.values());
        values.set(1, new GraphValue(new ValueId(1), new TensorDescriptor(DataType.FLOAT32,
                Shape.scalar(), Optional.of(LayoutDescriptor.contiguous(Shape.scalar())), false)));
        var plan = new CpuPartitionPreparer().analyze(descriptors(c, values)).plan();
        var route = plan.units().getFirst().portablePlan();
        var fusion = assertInstanceOf(CpuAggregateFusionIr.class, route.portableKernelIr());
        var row = new CpuScanControl.Row(0, 3); var shared = CpuScanControl.Row.shared();
        var boundaries = List.of(row, shared, shared);
        var valueRows = new ArrayList<>(boundaries);
        var producer = fusion.producer().orElseThrow();
        for (var value : producer.values())
            valueRows.add(value.kind() == CpuKernelIr.Value.Kind.INPUT && value.ordinal() == 1
                    ? shared : row);
        var facts = new CpuScanControl(boundaries, valueRows);
        float[] input = {1e20f, 1, -1e20f, Float.NaN, Float.NaN, Float.NaN, 3, 4, 5};
        float[] output = {-19};
        CpuAggregateGeneratedKernelTest.invokeControlledAggregateBoundaries(route, facts,
                List.of(MemorySegment.ofArray(input), MemorySegment.ofArray(new float[]{1}),
                        MemorySegment.ofArray(output)),
                plan.aggregateGeometry().orElseThrow().packFusion(new long[3]),
                true, true, false, true);
        assertArrayEquals(new float[]{12}, output,
                "the removed ROW producer, not its SHARED broadcast input, owns contributor activity");
    }

    @Test void controlledRowOutputDominatesAggregateAndTailOperandLoads() throws Throwable {
        var c = context(DataType.FLOAT32, Shape.of(3, 3), new AxisReductionAttrs(1, false),
                Shape.of(3), false, 1, false, Set.of(), List.of());
        var values = new ArrayList<>(c.values());
        values.set(4, new GraphValue(new ValueId(4), new TensorDescriptor(DataType.FLOAT32,
                Shape.of(3), Optional.of(LayoutDescriptor.contiguous(Shape.of(3))), false)));
        var plan = new CpuPartitionPreparer().analyze(descriptors(c, values)).plan();
        var route = plan.units().getFirst().portablePlan();
        assertInstanceOf(CpuAggregateFusionIr.class, route.portableKernelIr());
        var rows = List.of(new CpuScanControl.Row(0, 3),
                new CpuScanControl.Row(0, 1), new CpuScanControl.Row(0, 1));
        float[] output = {-19, -19, -19};
        // The logical last row has no backing operand cells: its guard must precede either load.
        CpuAggregateGeneratedKernelTest.invokeControlledAggregateBoundaries(route,
                new CpuScanControl(rows, rows),
                List.of(MemorySegment.ofArray(new float[]{1, 2, 3, 4, 5, 6}),
                        MemorySegment.ofArray(new float[]{10, 20}), MemorySegment.ofArray(output)),
                plan.aggregateGeometry().orElseThrow().packFusion(new long[3]),
                true, true, true, false);
        assertArrayEquals(new float[]{16, 35, -19}, output);
    }

    @Test void sevenAndEightNodeUnitsDoNotEnlargeTheSixMemberRecognitionBudget() throws Throwable {
        for (int length : List.of(7,8)) {
            var c = context(DataType.FLOAT32,Shape.of(2,3),new AxisReductionAttrs(1,false),Shape.of(2),true,3,false,Set.of(),List.of());
            var nodes = new ArrayList<CompiledNode>(); var values = new ArrayList<>(c.values());
            int previous = 0;
            for (int i = 0; i < length-4; i++) {
                nodes.add(node(nodes.size(),new Operation(UnaryElementwiseKind.NEG,NoOperationAttrs.INSTANCE),List.of(previous),7+i));
                values.add(new GraphValue(new ValueId(7+i),values.getFirst().descriptor())); previous = 7+i;
            }
            nodes.add(node(nodes.size(),c.nodes().getFirst().operation(),List.of(previous,1),2));
            for (var old : c.nodes().subList(1,c.nodes().size())) nodes.add(new CompiledNode(new NodeId(nodes.size()),old.operation(),old.inputs(),old.outputs()));
            var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,nodes.stream().map(CompiledNode::id).toList());
            var memory = values.stream().map(v -> new LogicalMemoryRequirement(v.id(),v.descriptor(),nodes.stream().anyMatch(n -> n.outputs().contains(v.id()))
                    ? Optional.of(partition) : Optional.empty(),nodes.stream().anyMatch(n -> n.inputs().contains(v.id())) ? List.of(partition) : List.of(),v.id().equals(new ValueId(6)))).toList();
            var plan = new CpuPartitionPreparer().analyze(new PrepareContext<>(partition,nodes,values,memory,Map.of(),c.backendInputs())).plan();
            assertEquals(1,plan.units().size()); assertEquals(length,plan.units().getFirst().memberNodeOrdinals().size());
            assertTrue(plan.specializedSubgraphs().isEmpty(),"recognition budget remains six without changing executable selection");
        }
    }

    @Test void lateZeroSelectedExtentEmitsNoContributorTraversal() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean mean : List.of(false,true)) {
            var plan = new CpuPartitionPreparer().analyze(context(type,Shape.of(1_000_000_000,0),NoOperationAttrs.INSTANCE,
                    Shape.scalar(),true,0,mean,Set.of(),List.of())).plan();
            var route = plan.units().getFirst().portablePlan();
            assertEquals(0,plan.aggregateGeometry().orElseThrow().domainCount()); assertEquals(1,plan.elementCount());
            var code = ClassFile.of().parse(new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(),route.kernelIr()))
                    .methods().getFirst().code().orElseThrow();
            assertTrue(code.elementStream().filter(java.lang.classfile.Instruction.class::isInstance).map(java.lang.classfile.Instruction.class::cast)
                    .noneMatch(i -> i.opcode() == java.lang.classfile.Opcode.FMUL || i.opcode() == java.lang.classfile.Opcode.DMUL));
            // Structural only: never execute a billion-prefix test as a timing or loop budget.
        }
    }

    @Test void canonicalColumnBroadcastIsNotAnUnsupportedStridedRead() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var c = context(type,Shape.of(2,3),new AxisReductionAttrs(1,false),Shape.of(2),true,0,false,Set.of(),List.of());
            var values = new ArrayList<>(c.values()); values.set(1,new GraphValue(new ValueId(1),new TensorDescriptor(type,Shape.of(2,1),
                    Optional.of(LayoutDescriptor.contiguous(Shape.of(2,1))),false)));
            var plan = new CpuPartitionPreparer().analyze(descriptors(c,values)).plan();
            assertEquals(1,plan.units().size()); Object result = array(type,new double[2]);
            invoke(plan,List.of(segment(array(type,new double[]{1,2,3,4,5,6})),segment(array(type,new double[]{2,3})),segment(result)),0,2);
            assertEquals(12,get(result,0)); assertEquals(45,get(result,1));
        }
    }
    /** Shared test-only route for specialization and artifact-retention regressions.
     * @param type FLOAT32 or FLOAT64
     * @param tail zero for producer only, two for RELU
     * @return complete selected ordinary fusion route with segment boundaries */
    public static io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan ordinaryRoute(DataType type,int tail) {
        return new CpuPartitionPreparer().analyze(context(type,Shape.of(64,256),new AxisReductionAttrs(1,false),Shape.of(64),true,tail,false,Set.of(),List.of()))
                .plan().units().getFirst().portablePlan();
    }

    /** Creates the same canonical F32 typed topology with a different retained output extent.
     * @param rows positive static output extent
     * @return complete CPU projection for a geometry-only identity regression */
    public static PrepareContext<CpuPartitionAnalysisInputs> fusionContextForGeometryTest(int rows) {
        return context(DataType.FLOAT32,Shape.of(rows,256),new AxisReductionAttrs(1,false),Shape.of(rows),true,0,false,Set.of(),List.of());
    }
    /** Proves actual normalization scale AD reaches ordinary SUM, not only SUM-to-Shape. */
    @Test void actualCompilerLayerNormScaleGradientSelectsOrdinaryFusionAndRemovesItsProducer() throws Throwable {
        var input = leaf(Shape.of(2,3),false); var scale = leaf(Shape.of(3),true); var bias = leaf(Shape.of(3),false);
        var upstream = leaf(Shape.of(2,3),false);
        var objective = input.layerNorm(Shape.of(3),scale,bias,io.github.pho001.synaptik.model.datatype.ScalarValue.float32(.01f)).mul(upstream).sum();
        var request = new io.github.pho001.synaptik.compiler.FunctionalGradientRequest(List.of(
                new io.github.pho001.synaptik.compiler.FunctionalGradientRequest.Stage(List.of(
                    new io.github.pho001.synaptik.compiler.FunctionalGradientRequest.ForwardTensorReference(objective)),
                    List.of(Optional.empty()),List.of(scale),false,
                    io.github.pho001.synaptik.compiler.FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
        try (var cpu = io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.open()) {
            var artifacts = io.github.pho001.synaptik.compiler.GraphCompilationPort.compile(
                    io.github.pho001.synaptik.config.compile.CompileMode.FORWARD_AND_BACKWARD,List.of(objective),Optional.of(request),
                    io.github.pho001.synaptik.config.compile.GraphOptimizationConfig.disabled(),io.github.pho001.synaptik.config.compile.BackendIntent.unconstrained(),
                    io.github.pho001.synaptik.config.compile.PartitionScoringConfig.neutral(),List.of(cpu.capabilityProvider()),List.of(cpu.availabilitySnapshot()));
            int selected = 0;
            for (var partition : artifacts.partitions()) {
                var context = io.github.pho001.synaptik.prepare.GraphPreparation.project(artifacts,partition,CpuPartitionAnalysisInputs.DEFAULT);
                var plan = new CpuPartitionPreparer().analyze(context).plan();
                for (var unit : plan.units()) if (unit.portablePlan().portableKernelIr() instanceof CpuAggregateFusionIr fusion
                        && fusion.aggregate().kind() == CpuAggregateIr.Kind.SUM
                        && Arrays.equals(new int[]{0},fusion.aggregate().selectedAxes()) && fusion.producer().isPresent()) {
                    var reduction = unit.memberNodeOrdinals().stream().map(context.nodes()::get)
                            .filter(n -> n.operation().kind() == AggregateReductionKind.SUM).findFirst().orElseThrow();
                    assertFalse(plan.boundaryValues().contains(reduction.inputs().getFirst()));
                    assertTrue(fusion.producer().orElseThrow().instructions().stream().anyMatch(i -> i.opcode() == CpuPointwiseOpcode.MUL));
                    assertTrue(unit.runtimeFacts().workspaceDeclaration().isPresent()); selected++;
                }
            }
            assertTrue(selected > 0,"actual scale gradient must select ordinary producer/SUM fusion");
        }
    }

    private static io.github.pho001.synaptik.model.tensor.Tensor leaf(Shape shape, boolean gradient) {
        return io.github.pho001.synaptik.model.tensor.TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),gradient));
    }
    @Test void expandingAddStaysSplitAndInvalidDirectInputIsRejected() throws Throwable {
        var c = context(DataType.FLOAT32,Shape.of(2,3),NoOperationAttrs.INSTANCE,Shape.scalar(),true,1,false,Set.of(),List.of());
        var values = new ArrayList<>(c.values());
        for (int id : List.of(4,5)) values.set(id,new GraphValue(new ValueId(id),new TensorDescriptor(DataType.FLOAT32,
                Shape.of(2),Optional.of(LayoutDescriptor.contiguous(Shape.of(2))),false)));
        c = descriptors(c,values);
        var plan = new CpuPartitionPreparer().analyze(c).plan();
        assertEquals(2,plan.units().size());
        var producerOnly = assertInstanceOf(CpuAggregateFusionIr.class,plan.units().getFirst().portablePlan().portableKernelIr());
        assertTrue(producerOnly.epilogue().isEmpty());
        assertTrue(plan.bufferDeclarations().stream().anyMatch(b -> b.valueId().equals(new ValueId(3))));
        var tailOnly = (CpuAggregateFusionIr)new CpuPartitionLowering().lower(context(DataType.FLOAT32,Shape.of(2),
                NoOperationAttrs.INSTANCE,Shape.scalar(),false,1,false,Set.of(),List.of())).portableKernelIr();
        assertThrows(IllegalArgumentException.class,() -> new CpuAggregateFusionIr(tailOnly.aggregate(),Optional.empty(),
                List.of(),List.of(),-2,tailOnly.epilogue(),tailOnly.epilogueInputs(),tailOnly.epilogueAccesses(),
                tailOnly.boundaries(),tailOnly.inputExtents()));
    }

    @Test void fanOutOutsidePartitionAndUnsupportedTypesKeepTheirStores() throws Throwable {
        for (boolean outside : List.of(false,true)) {
            var c = context(DataType.FLOAT32,Shape.of(2,3),new AxisReductionAttrs(1,false),Shape.of(2),true,3,false,Set.of(),List.of());
            var nodes = new ArrayList<>(c.nodes()); var values = new ArrayList<>(c.values()); var memory = new ArrayList<>(c.memoryRequirements());
            if (outside) {
                var old = memory.get(2);
                memory.set(2,new LogicalMemoryRequirement(old.valueId(),old.descriptor(),old.producerPartition(),
                        List.of(c.partition(),new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,List.of(new NodeId(99)))),false));
            } else {
                nodes.add(node(nodes.size(),new Operation(UnaryElementwiseKind.NEG,NoOperationAttrs.INSTANCE),List.of(2),7));
                var id = new ValueId(7); var descriptor = values.get(2).descriptor(); values.add(new GraphValue(id,descriptor));
                var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,nodes.stream().map(CompiledNode::id).toList());
                memory.replaceAll(r -> new LogicalMemoryRequirement(r.valueId(),r.descriptor(),r.producerPartition().isPresent()
                        ? Optional.of(partition) : Optional.empty(),r.consumerPartitions().isEmpty() ? List.of() : List.of(partition),r.graphOutput()));
                memory.add(new LogicalMemoryRequirement(id,descriptor,Optional.of(partition),List.of(),true));
                c = new PrepareContext<>(partition,nodes,values,memory,Map.of(),c.backendInputs());
            }
            if (outside) c = new PrepareContext<>(c.partition(),nodes,values,memory,Map.of(),c.backendInputs());
            
            var plan = new CpuPartitionPreparer().analyze(c).plan();
            assertTrue(plan.bufferDeclarations().stream().anyMatch(b -> b.valueId().equals(new ValueId(2))));
            assertTrue(plan.units().stream().map(u -> u.portablePlan().portableKernelIr()).filter(CpuAggregateFusionIr.class::isInstance)
                    .map(CpuAggregateFusionIr.class::cast).allMatch(f -> f.producer().isEmpty()));
        }
        for (var type : List.of(DataType.INT32,DataType.INT64,DataType.BFLOAT16,DataType.FLOAT16)) {
            var c = context(type,Shape.of(2,3),new AxisReductionAttrs(1,false),Shape.of(2),true,0,false,Set.of(),List.of());
            
            var plan = new CpuPartitionPreparer().analyze(c).plan();
            assertEquals(2,plan.units().size());
            assertTrue(plan.bufferDeclarations().stream().anyMatch(b -> b.valueId().equals(new ValueId(2))));
        }
    }
    @Test void roundedProductIsNotADotProductAndTailCannotSeeAccumulatorPrecision() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            double epsilon = Math.scalb(1d,type == DataType.FLOAT32 ? -23 : -52);
            check(type,Shape.of(2),NoOperationAttrs.INSTANCE,Shape.scalar(),true,0,false,
                    new double[]{1+epsilon,-(1+2*epsilon)},new double[]{1+epsilon,1},0);
            check(type,Shape.of(2),NoOperationAttrs.INSTANCE,Shape.scalar(),false,1,false,
                    new double[]{1,epsilon/2},new double[]{1,1},-1);
            check(type,Shape.of(2),NoOperationAttrs.INSTANCE,Shape.scalar(),false,1,true,
                    new double[]{2,epsilon},new double[]{1,1},-1);
            var actual = execute(type,Shape.of(2),NoOperationAttrs.INSTANCE,Shape.scalar(),true,0,false,
                    new double[]{1+epsilon,-(1+2*epsilon)},new double[]{1+epsilon,1},0,0);
            assertEquals(0,actual[0]);
        }
    }

    @Test void exactAxesEmptyDomainsSpecialsAndBothAddOrdersMatchSplitAndIndependentOracle() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            for (boolean mean : List.of(false,true)) for (int tail : List.of(0,1,2,3,4)) {
                double[] values = {1,-1,Math.scalb(1,type == DataType.FLOAT32 ? -24 : -53),-0d,
                        Double.POSITIVE_INFINITY,Double.NEGATIVE_INFINITY,Double.NaN,Double.MIN_VALUE,
                        type == DataType.FLOAT32 ? Float.MAX_VALUE : Double.MAX_VALUE,2,3,-4,
                        5,-6,7,-8,9,-10,11,-12,13,-14,15,-16};
                double[] factors = new double[24]; Arrays.fill(factors,2);
                check(type,Shape.of(2,3,4),new MultiAxisReductionAttrs(List.of(2,0),false),Shape.of(3),true,tail,mean,values,factors,-1);
                check(type,Shape.of(2,3,4),new MultiAxisReductionAttrs(List.of(0,2),true),Shape.of(1,3,1),true,tail,mean,values,factors,-1);
                check(type,Shape.of(2,3),new AxisReductionAttrs(1,false),Shape.of(2),true,tail,mean,
                        new double[]{1,2,3,4,5,6},new double[]{2,2,2,2,2,2},-1);
            }
            for (boolean mean : List.of(false,true)) {
                check(type,Shape.scalar(),NoOperationAttrs.INSTANCE,Shape.scalar(),true,3,mean,new double[]{-0d},new double[]{1},-0d);
                check(type,Shape.of(2),new MultiAxisReductionAttrs(List.of(),false),Shape.of(2),true,3,mean,
                        new double[]{-0d,type == DataType.FLOAT32 ? Float.MIN_VALUE : Double.MIN_VALUE},new double[]{1,1},-0d);
                check(type,Shape.of(2,0),new AxisReductionAttrs(1,false),Shape.of(2),true,3,mean,new double[0],new double[0],0);
                check(type,Shape.of(0,3),new AxisReductionAttrs(1,false),Shape.of(0),true,3,mean,new double[0],new double[0],0);
                check(type,Shape.of(2,1),new AxisReductionAttrs(1,true),Shape.of(2,1),true,0,mean,new double[]{-0d,1},new double[]{1,2},0);
            }
        }
    }

    @Test void selectedPrivateResourcesAndProtectedPartialFusionAreExact() throws Throwable {
        var context = context(DataType.FLOAT64,Shape.of(2,3),new AxisReductionAttrs(1,false),Shape.of(2),true,3,false,Set.of(),List.of());
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        assertEquals(1,plan.units().size());
        var fusion = assertInstanceOf(CpuAggregateFusionIr.class,plan.units().getFirst().portablePlan().portableKernelIr());
        assertEquals(List.of(new ValueId(0),new ValueId(1),new ValueId(4),new ValueId(6)),plan.boundaryValues());
        assertEquals(272,plan.aggregateGeometry().orElseThrow().workspaceBytes(1));
        assertEquals(1,plan.selectedRangeCount());
        assertEquals(3,new CpuPartitionLowering().lower(context).virtualValues().size());
        assertEquals(CpuSpecializedSubgraph.ExecutionDisposition.EXISTING_SPECIALIZED,plan.specializedSubgraphs().getFirst().disposition());
        for (int protectedId : List.of(2,3,5)) {
            var protectedContext = context(DataType.FLOAT64,Shape.of(2,3),new AxisReductionAttrs(1,false),Shape.of(2),true,3,false,Set.of(protectedId),List.of());
            var selected = new CpuPartitionPreparer().analyze(protectedContext).plan();
            assertTrue(selected.bufferDeclarations().stream().anyMatch(b -> b.valueId().equals(new ValueId(protectedId))));
            assertTrue(selected.units().size() >= 2);
        }
        assertTrue(fusion.producer().isPresent()); assertTrue(fusion.epilogue().isPresent());
    }

    @Test void allHeapSegmentCarrierCombinationsRetainOffsetsAndSubranges() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (int mask = 0; mask < 16; mask++) {
            var patterns = new ArrayList<CarrierAccess>();
            for (int i = 0; i < 4; i++) patterns.add((mask&(1<<i)) == 0 ? CarrierAccess.MEMORY_SEGMENT
                    : type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY);
            var c = context(type,Shape.of(2,3),new AxisReductionAttrs(1,false),Shape.of(2),true,3,false,Set.of(),patterns);
            var values = c.values().stream().map(v -> new GraphValue(v.id(),new TensorDescriptor(type,v.descriptor().shape(),
                    Optional.of(LayoutDescriptor.of(v.descriptor().shape(),v.descriptor().layout().orElseThrow().strides(),2,false)),false))).toList();
            c = new PrepareContext<>(c.partition(),c.nodes(),values,c.memoryRequirements().stream().map(r ->
                    new LogicalMemoryRequirement(r.valueId(),values.get((int)r.valueId().value()).descriptor(),r.producerPartition(),r.consumerPartitions(),r.graphOutput())).toList(),Map.of(),c.backendInputs());
            var plan = new CpuPartitionPreparer().analyze(c).plan();
            assertEquals(1,plan.units().size());
            Object left = array(type,new double[]{99,99,1,2,3,4,5,6,99});
            Object right = array(type,new double[]{99,99,2,2,2,2,2,2,99});
            Object bias = array(type,new double[]{99,99,-1,99});
            Object output = array(type,new double[]{99,99,99,99,99});
            var physical = List.of(left,right,bias,output); var args = new ArrayList<Object>();
            for (int i = 0; i < 4; i++) args.add(patterns.get(i) == CarrierAccess.MEMORY_SEGMENT ? segment(physical.get(i)) : physical.get(i));
            invoke(plan,args,0,2);
            assertEquals(11,get(output,2)); assertEquals(29,get(output,3));
            assertEquals(99,get(output,0)); assertEquals(99,get(output,4));
        }
    }

    @Test void nativeMixedCarriersBroadcastAndInvocationBasesAreExplicit() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (int mask = 0; mask < 16; mask++) {
            var patterns = new ArrayList<CarrierAccess>();
            for (int i = 0; i < 4; i++) patterns.add((mask&(1<<i)) == 0 ? CarrierAccess.MEMORY_SEGMENT
                    : type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY);
            var c = context(type,Shape.of(2,3),new AxisReductionAttrs(1,false),Shape.of(2),true,1,false,Set.of(),patterns);
            var values = new ArrayList<>(c.values()); values.set(1,new GraphValue(new ValueId(1),new TensorDescriptor(type,Shape.of(3),
                    Optional.of(LayoutDescriptor.contiguous(Shape.of(3))),false))); c = descriptors(c,values);
            var plan = new CpuPartitionPreparer().analyze(c).plan();
            Object left = array(type,new double[]{99,99,1,2,3,4,5,6,99});
            Object right = array(type,new double[]{99,99,99,2,3,4,99});
            Object bias = array(type,new double[]{99,99,99,99,-1,99});
            Object output = array(type,new double[]{99,99,99,99,99,99,99,99});
            try (var arena = Arena.ofConfined()) {
                var args = new ArrayList<Object>();
                for (var a : List.of(left,right,bias,output)) {
                    if (patterns.get(args.size()) == CarrierAccess.MEMORY_SEGMENT) {
                        var from = segment(a); var nativeSegment = arena.allocate(from.byteSize(),type.byteWidth());
                        MemorySegment.copy(from,0,nativeSegment,0,from.byteSize()); args.add(nativeSegment);
                    } else args.add(a);
                }
                invoke(plan,args,0,2,new long[]{2,3,4,5});
                if (args.getLast() instanceof MemorySegment s) MemorySegment.copy(s,0,segment(output),0,s.byteSize());
                assertEquals(19,get(output,5)); assertEquals(46,get(output,6)); assertEquals(99,get(output,4)); assertEquals(99,get(output,7));
            }
        }
    }

    @Test void specializedCleanJavaHasSameStateAndClassFileHasNoHiddenRuntimeWork() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var plan = new CpuPartitionPreparer().analyze(context(type,Shape.of(64,256),new AxisReductionAttrs(1,false),Shape.of(64),true,0,false,Set.of(),
                    Collections.nCopies(3,type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY))).plan();
            var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
            byte[] bytes = generator.generateClassBytes(route.specialization(),route.kernelIr());
            String evidence = System.getenv("SYNAPTIK_CPU0049_EVIDENCE");
            if (evidence != null) {
                java.nio.file.Files.write(java.nio.file.Path.of(evidence).resolve("fusion-"+type+".class"),bytes);
                java.nio.file.Files.write(java.nio.file.Path.of(evidence).resolve("fusion-"+type+".compatibility"),route.specialization().compatibilityBytes());
                java.nio.file.Files.write(java.nio.file.Path.of(evidence).resolve("fusion-"+type+".classidentity"),route.specialization().classIdentityBytes());
            }
            assertArrayEquals(bytes,generator.generateClassBytes(route.specialization(),route.kernelIr()));
            var model = ClassFile.of().parse(bytes);
            assertEquals(1,model.methods().size()); assertTrue(model.fields().isEmpty());
            assertEquals(0,model.constantPool().bootstrapMethodCount());
            for (var entry : model.constantPool()) if (entry instanceof MemberRefEntry member) {
                String owner = member.owner().asInternalName();
                assertFalse(owner.startsWith("io/github/pho001/synaptik"),owner);
                assertTrue(owner.equals("java/lang/Float") || owner.equals("java/lang/Double") || owner.equals("java/lang/Long")
                        || owner.startsWith("java/lang/foreign/"),owner);
            }
            assertTrue(model.methods().getFirst().code().orElseThrow().elementStream()
                    .filter(java.lang.classfile.Instruction.class::isInstance).map(java.lang.classfile.Instruction.class::cast)
                    .noneMatch(i -> i.opcode().name().contains("NEW") || i.opcode().name().contains("SWITCH")));
            double[] a = new double[64*256], b = new double[a.length];
            for (int i = 0; i < a.length; i++) { a[i] = (i%17-8)*.25; b[i] = (i%7-3)*.5; }
            Object left = array(type,a), right = array(type,b), actual = array(type,new double[64]), expected = array(type,new double[64]);
            try (var arena = Arena.ofConfined()) {
                var state = arena.allocate(plan.aggregateGeometry().orElseThrow().workspaceBytes(1),8);
                if (type == DataType.FLOAT32) CpuAggregateFusionCleanJavaOracle.f32((float[])left,0,(float[])right,0,(float[])expected,0,state,0,64);
                else CpuAggregateFusionCleanJavaOracle.f64((double[])left,0,(double[])right,0,(double[])expected,0,state,0,64);
                invoke(plan,List.of(left,right,actual),0,64);
                for (int i = 0; i < 64; i++) assertRaw(type,get(expected,i),get(actual,i));
            }
            assertTrue(new String(route.specialization().compatibilityBytes()).contains("ordinary-aggregate-fusion-v1="));
            assertTrue(new String(route.specialization().classIdentityBytes()).contains("ordinary-aggregate-fusion-v1="));
            assertEquals(52,route.specialization().classIdentitySchema());
        }
    }

    private static void check(DataType type, Shape input, OperationAttrs attrs, Shape output, boolean producer,
            int tail, boolean mean, double[] left, double[] right, double bias) throws Throwable {
        double[] actual = execute(type,input,attrs,output,producer,tail,mean,left,right,bias,0);
        Object product = array(type,left), reduced = array(type,new double[actual.length]);
        if (producer) for (int i = 0; i < left.length; i++) set(product,i,type == DataType.FLOAT32
                ? (float)left[i]*(float)right[i] : left[i]*right[i]);
        var bare = CpuAggregateLoweringTest.lower(mean ? AggregateReductionKind.MEAN : AggregateReductionKind.SUM,type,input,attrs,output);
        CpuScalarReferenceKernel.execute((CpuAggregateIr)bare.portableKernelIr(),bare.aggregateGeometry().orElseThrow(),List.of(argument(type,product),argument(type,reduced)));
        // Also execute the current materialized exact aggregate, independently of fusion.
        var split = new CpuPartitionPreparer().analyze(CpuAggregateLoweringTest.context(mean ? AggregateReductionKind.MEAN : AggregateReductionKind.SUM,type,input,attrs,output)).plan();
        Object splitOut = array(type,new double[actual.length]); invoke(split,List.of(segment(product),segment(splitOut)),0,actual.length);
        for (int i = 0; i < actual.length; i++) {
            assertRaw(type,get(reduced,i),get(splitOut,i));
            double expected = get(reduced,i);
            if (tail == 1 || tail == 3 || tail == 4) expected = type == DataType.FLOAT32 ? (float)expected+(float)bias : expected+bias;
            if (tail >= 2) expected = type == DataType.FLOAT32 ? Math.max(0f,(float)expected) : Math.max(0d,expected);
            assertRaw(type,expected,actual[i]);
        }
    }

    private static double[] execute(DataType type, Shape input, OperationAttrs attrs, Shape output, boolean producer,
            int tail, boolean mean, double[] left, double[] right, double bias, int unused) throws Throwable {
        var plan = new CpuPartitionPreparer().analyze(context(type,input,attrs,output,producer,tail,mean,Set.of(),List.of())).plan();
        assertEquals(1,plan.units().size()); assertInstanceOf(CpuAggregateFusionIr.class,plan.units().getFirst().portablePlan().portableKernelIr());
        Object actual = array(type,new double[Math.toIntExact(output.knownElementCount().orElseThrow())]);
        var args = new ArrayList<Object>(); args.add(segment(array(type,left)));
        if (producer) args.add(segment(array(type,right)));
        if (tail == 1 || tail == 3 || tail == 4) args.add(segment(array(type,new double[]{bias})));
        args.add(segment(actual)); invoke(plan,args,0,Math.toIntExact(output.knownElementCount().orElseThrow()));
        double[] result = new double[Math.toIntExact(output.knownElementCount().orElseThrow())];
        for (int i = 0; i < result.length; i++) result[i] = get(actual,i);
        return result;
    }

    static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type, Shape input, OperationAttrs attrs,
            Shape output, boolean producer, int tail, boolean mean, Set<Integer> protectedValues, List<CarrierAccess> carriers) {
        var nodes = new ArrayList<CompiledNode>();
        if (producer) nodes.add(node(nodes.size(),new Operation(BinaryArithmeticKind.MUL,NoOperationAttrs.INSTANCE),List.of(0,1),2));
        nodes.add(node(nodes.size(),new Operation(mean ? AggregateReductionKind.MEAN : AggregateReductionKind.SUM,attrs),List.of(producer ? 2 : 0),3));
        int finalId = 3;
        if (tail == 1 || tail == 3 || tail == 4) { nodes.add(node(nodes.size(),new Operation(BinaryArithmeticKind.ADD,NoOperationAttrs.INSTANCE),tail == 4 ? List.of(4,3) : List.of(3,4),5)); finalId = 5; }
        if (tail >= 2) { nodes.add(node(nodes.size(),new Operation(UnaryElementwiseKind.RELU,NoOperationAttrs.INSTANCE),List.of(finalId),6)); finalId = 6; }
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<GraphValue>(); var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < 7; i++) {
            var id = new ValueId(i); Shape shape = i <= 2 ? input : i == 4 ? Shape.scalar() : output;
            var descriptor = new TensorDescriptor(type,shape,Optional.of(LayoutDescriptor.contiguous(shape)),false);
            values.add(new GraphValue(id,descriptor)); boolean produced = nodes.stream().anyMatch(n -> n.outputs().contains(id));
            memory.add(new LogicalMemoryRequirement(id,descriptor,produced ? Optional.of(partition) : Optional.empty(),
                    nodes.stream().anyMatch(n -> n.inputs().contains(id)) ? List.of(partition) : List.of(),i == finalId || protectedValues.contains(i)));
        }
        return new PrepareContext<>(partition,nodes,values,memory,Map.of(),new CpuPartitionAnalysisInputs(false,carriers));
    }

    private static CompiledNode node(int ordinal, Operation operation, List<Integer> inputs, int output) {
        return new CompiledNode(new NodeId(ordinal),operation,inputs.stream().map(ValueId::new).toList(),List.of(new ValueId(output)));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> descriptors(PrepareContext<CpuPartitionAnalysisInputs> c,List<GraphValue> values) {
        return new PrepareContext<>(c.partition(),c.nodes(),values,c.memoryRequirements().stream().map(r -> new LogicalMemoryRequirement(
                r.valueId(),values.get((int)r.valueId().value()).descriptor(),r.producerPartition(),r.consumerPartitions(),r.graphOutput())).toList(),Map.of(),c.backendInputs());
    }

    private static void invoke(CpuPartitionPreparationPlan plan, List<?> physical, long start, long end) throws Throwable {
        invoke(plan,physical,start,end,new long[physical.size()]);
    }

    private static void invoke(CpuPartitionPreparationPlan plan, List<?> physical, long start, long end,long[] bases) throws Throwable {
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        try (var arena = Arena.ofConfined()) {
            var args = new ArrayList<Object>(physical);
            if (route.specialization().scratchParameter()) args.add(arena.allocate(plan.aggregateGeometry().orElseThrow().workspaceBytes(1),8));
            args.add(route.specialization().aggregateFusionIr().isPresent() ? plan.aggregateGeometry().orElseThrow().packFusion(bases)
                    : plan.aggregateGeometry().orElseThrow().pack(new long[2]));
            long middle = (start+end)/2;
            args.add(start); args.add(middle); artifact.entryPoint().invokeWithArguments(args);
            args.set(args.size()-2,middle); artifact.entryPoint().invokeWithArguments(args);
            args.set(args.size()-1,end); artifact.entryPoint().invokeWithArguments(args);
        }
    }

    private static Object array(DataType type, double[] values) {
        if (type == DataType.FLOAT64) return values.clone();
        float[] result = new float[values.length]; for (int i = 0; i < result.length; i++) result[i] = (float)values[i]; return result;
    }
    private static MemorySegment segment(Object array) { return array instanceof float[] f ? MemorySegment.ofArray(f) : MemorySegment.ofArray((double[])array); }
    private static double get(Object array, int index) { return array instanceof float[] f ? f[index] : ((double[])array)[index]; }
    private static void set(Object array, int index, double value) { if (array instanceof float[] f) f[index] = (float)value; else ((double[])array)[index] = value; }
    private static CpuBufferArgument argument(DataType type, Object array) {
        var segment = segment(array); return new CpuBufferArgument.Segment(type,segment,segment.byteSize(),false);
    }
    private static void assertRaw(DataType type, double expected, double actual) {
        if (Double.isNaN(expected)) assertTrue(Double.isNaN(actual));
        else if (type == DataType.FLOAT32) assertEquals(Float.floatToRawIntBits((float)expected),Float.floatToRawIntBits((float)actual));
        else assertEquals(Double.doubleToRawLongBits(expected),Double.doubleToRawLongBits(actual));
    }
}
