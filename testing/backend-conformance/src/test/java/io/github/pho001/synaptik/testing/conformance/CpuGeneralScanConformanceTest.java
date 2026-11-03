package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Model-only body-bearing provider and exact saved-site execution witnesses; no Compiler edge. */
final class CpuGeneralScanConformanceTest {
    /**
     * Explicit selected-strategy witness with SUM and transpose-MATMUL terminals feeding
     * separate shared carries. The authentic transpose supplies the static row-role proof.
     * Only final carries and emission are demanded; the canonical complete residual inventory
     * validates the captured body without adding publication demands. Default contexts remain unchanged.
     */
    @Test void selectedFusedForwardUsesVectorAndParallelRowTerminals() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var h = new ValueId(0); var p = new ValueId(1); var x = new ValueId(2);
            var negative = new ValueId(3); var sum = new ValueId(4); var next = new ValueId(5);
            var transpose = new ValueId(6); var product = new ValueId(7); var nextProduct = new ValueId(8);
            var shared = descriptor(type,Shape.of(2)); var row = descriptor(type,Shape.of(7,2));
            var matrix = descriptor(type,Shape.of(2,2)); var transposed = Shape.of(2,7);
            var nodes = List.of(
                    new CompiledNode(new NodeId(0),new Operation(UnaryElementwiseKind.NEG,NoOperationAttrs.INSTANCE),List.of(x),List.of(negative)),
                    new CompiledNode(new NodeId(1),new Operation(io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.SUM,
                            new io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs(0,false)),List.of(negative),List.of(sum)),
                    new CompiledNode(new NodeId(2),new Operation(BinaryArithmeticKind.ADD,NoOperationAttrs.INSTANCE),List.of(h,sum),List.of(next)),
                    new CompiledNode(new NodeId(3),new Operation(io.github.pho001.synaptik.model.operation.layout.AxisTransformKind.PERMUTE,
                            new io.github.pho001.synaptik.model.operation.layout.PermutationAttrs(List.of(1,0))),List.of(x),List.of(transpose)),
                    new CompiledNode(new NodeId(4),new Operation(io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL,
                            NoOperationAttrs.INSTANCE),List.of(transpose,negative),List.of(product)),
                    new CompiledNode(new NodeId(5),new Operation(BinaryArithmeticKind.ADD,NoOperationAttrs.INSTANCE),List.of(p,product),List.of(nextProduct)));
            var residuals = List.of(h,negative,sum,next,p,transpose,product,nextProduct);
            var graphValues = List.of(new GraphValue(h,shared),new GraphValue(p,matrix),new GraphValue(x,row),new GraphValue(negative,row),
                    new GraphValue(sum,shared),new GraphValue(next,shared),new GraphValue(transpose,new TensorDescriptor(type,transposed,
                            Optional.of(LayoutDescriptor.of(transposed,new long[]{1,2},0,true)),false)),
                    new GraphValue(product,matrix),new GraphValue(nextProduct,matrix));
            var phases = new HashMap<NodeId,GraphPhase>(); nodes.forEach(n -> phases.put(n.id(),GraphPhase.FORWARD));
            var graph = new CompiledGraphModel(graphValues,nodes,List.of(h,p,x),residuals,phases);
            var body = new CompiledScanBody(graph,List.of(next,nextProduct),List.of(negative),residuals,List.of(0,1,2));
            var outputs = new ArrayList<TensorDescriptor>();
            outputs.add(shared); outputs.add(matrix); outputs.add(descriptor(type,Shape.of(3,7,2)));
            for (var id : residuals) {
                var local = graphValues.stream().filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor().shape();
                long[] stacked = new long[local.rank()+1]; stacked[0] = 3;
                System.arraycopy(local.toLongArray(),0,stacked,1,local.rank());
                outputs.add(descriptor(type,Shape.of(stacked)));
            }
            var q = new OperationCapabilityQuery(new Operation(ScanKind.SCAN,new ScanAttrs(3,2,1,0,1,8,false)),
                    List.of(shared,matrix,descriptor(type,Shape.of(3,7,2))),outputs,Optional.of(body),Optional.empty());
            assertTrue(new CpuCapabilityProvider().supports(q));
            for (boolean vector : List.of(false,true)) for (boolean parallel : List.of(false,true)) {
                var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(vector
                        ? CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                        : CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,parallel ? 4 : 1,4,2);
                var analysis = new CpuPartitionPreparer().analyze(context(q,new CpuPartitionAnalysisInputs(false,List.of(),config)));
                var unit = analysis.plan().units().getFirst();
                assertTrue(unit.runtimeFacts().generalScan().orElseThrow().ir().stages().orElseThrow().rowMapReductionIr().isPresent());
                assertEquals(parallel ? vector ? CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_VECTOR : CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR
                        : vector ? CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,unit.executionStrategy());
                try (var f = new CpuRecurrentScanConformanceTest.Fixture(analysis,1,true,true,Map.of())) {
                    fill(f,type,0,.25); fill(f,type,1,.25); fill(f,type,2,.125);
                    f.executable.bind(f.state).execute();
                    assertArrayEquals(new double[]{-2.375,-2.375},values(f,type,3));
                    assertArrayEquals(new double[]{-.078125,-.078125,-.078125,-.078125},values(f,type,4));
                    var expected = new double[42]; Arrays.fill(expected,-.125); assertArrayEquals(expected,values(f,type,5));
                }
            }
        }
    }
    @Test void combinedProviderQualificationDoesNotGrantAllValidFloat64OrNativeOwners() {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var query = CpuSymbolicScanConformanceTest.symbolicLengthQuery(type);
            assertTrue(new CpuCapabilityProvider().supports(query)); assertFalse(new MetalCapabilityProvider().supports(query));
            assertEquals(DataType.INT64,query.inputs().getLast().dataType());
            var wrong = new ArrayList<>(query.inputs());
            wrong.set(wrong.size()-1,new TensorDescriptor(DataType.INT64,Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.DynamicDimension("Other")),Optional.empty(),false));
            assertFalse(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(query.operation(),wrong,query.outputs(),query.scanBody(),query.scanActivityPlan())));
        }
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthControlRemainsInt64AndProvidersIndependentlyQualifyTypedPrefixes(DataType type) {
        var q = lengthQuery(type); var cpu = new CpuCapabilityProvider();
        assertTrue(cpu.supports(q));
        assertEquals(type == DataType.FLOAT32,new MetalCapabilityProvider().supports(q));
        assertEquals(DataType.INT64,q.inputs().getLast().dataType());
        for (var control : List.of(DataType.INT32,DataType.FLOAT32)) {
            var inputs = new ArrayList<>(q.inputs());
            inputs.set(inputs.size()-1,descriptor(control,Shape.of(2)));
            var malformed = new OperationCapabilityQuery(q.operation(),inputs,q.outputs(),q.scanBody(),q.scanActivityPlan());
            assertFalse(cpu.supports(malformed));
            assertFalse(new MetalCapabilityProvider().supports(malformed));
        }
        for (var low : List.of(DataType.BFLOAT16,DataType.FLOAT16)) {
            var negative = withType(q,low); assertEquals(DataType.INT64,negative.inputs().getLast().dataType());
            assertFalse(cpu.supports(negative));
        }
        for (int mode = 0; mode < 3; mode++) {
            var analysis = new CpuPartitionPreparer().analyze(context(q));
            var control = analysis.plan().units().getFirst().portablePlan().specialization().boundaryDataTypes();
            assertEquals(List.of(type,type,DataType.INT64),control.subList(0,3));
            try (var f = new CpuRecurrentScanConformanceTest.Fixture(analysis,mode,true,true,Map.of())) {
                values(f,type,0,.125,-0d); values(f,type,1,.25,Double.NaN,.5,Double.NaN,Double.NaN,Double.NaN);
                f.lengths(2,2,0); f.executable.bind(f.state).execute();
                double first = work(type,Math.tanh(work(type,.125+.25))), second = work(type,Math.tanh(work(type,first+.5)));
                assertArrayEquals(new double[]{second,-0d},values(f,type,3));
                assertArrayEquals(new double[]{first,0,second,0,0,0},values(f,type,4));
            }
        }
    }

    /**
     * Keeps a compact provider/ledger witness independent of Compiler source constructors.
     * @param type homogeneous floating body type
     * @return complete typed length query with two ROW(0) scalar-per-row cells
     */
    static OperationCapabilityQuery lengthQuery(DataType type) {
        var q = query(type,UnaryElementwiseKind.TANH,false,3,Shape.of(2));
        var a = (ScanAttrs)q.operation().attrs(); var inputs = new ArrayList<>(q.inputs());
        inputs.add(descriptor(DataType.INT64,Shape.of(2)));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN,new ScanAttrs(a.timeCount(), a.carryCount(), a.sequenceCount(), a.invariantCount(), a.emissionCount(), a.residualCount(), a.reverse(), true, 1, List.of(OptionalInt.of(0)))),inputs,q.outputs(),q.scanBody(),q.scanActivityPlan());
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void denseActualReadViewsHaveIndependentProviderAndTypedExecutionWitnesses(DataType type) {
        var fixtures = List.of(query(type,UnaryElementwiseKind.TANH,false,3,Shape.of(2)),withType(shapedQuery(),type),withType(matmulQuery(),type));
        for (int family = 0; family < fixtures.size(); family++) {
            var canonical = fixtures.get(family);
            var views = canonical.inputs().stream().map(d -> new TensorDescriptor(type,d.shape(),Optional.of(
                    LayoutDescriptor.of(d.shape(),d.layout().orElseThrow().strides(),0,true)),d.requiresGrad())).toList();
            var q = new OperationCapabilityQuery(canonical.operation(),views,canonical.outputs(),canonical.scanBody(),canonical.scanActivityPlan());
            for (var d : q.inputs()) assertTrue(d.layout().orElseThrow().isView(),"actual binder gets the view");
            assertTrue(new CpuCapabilityProvider().supports(q));
            assertEquals(type == DataType.FLOAT32,new MetalCapabilityProvider().supports(q));
            for (var low : List.of(DataType.BFLOAT16,DataType.FLOAT16)) assertFalse(new CpuCapabilityProvider().supports(withType(q,low)));
            var a = new CpuPartitionPreparer().analyze(context(canonical)); var b = new CpuPartitionPreparer().analyze(context(q));
            assertEquals(a.requirements(),b.requirements());
            assertEquals(a.plan().units().getFirst().runtimeFacts().generalScan(),b.plan().units().getFirst().runtimeFacts().generalScan());
            int reads = q.inputs().size();
            for (int mode = 0; mode < 3; mode++) try (var f = new CpuRecurrentScanConformanceTest.Fixture(b,mode,true,true,Map.of())) {
                fill(f,type,0,.125); fill(f,type,1,.0625); if (reads == 3) fill(f,type,2,.25);
                f.executable.bind(f.state).execute();
                double state = .125;
                double[] history = new double[family == 0 ? 6 : family == 1 ? 9 : 18];
                for (int t = 0; t < 3; t++) {
                    state = family == 0 ? work(type,Math.tanh(work(type,state+.0625)))
                            : work(type,state+(family == 1 ? .015625 : .03125));
                    int width = family == 0 ? 2 : family == 1 ? 3 : 6;
                    Arrays.fill(history,t*width,(t+1)*width,family == 1 ? work(type,state+state) : family == 2 ? .03125 : state);
                }
                double[] expected = new double[family == 0 ? 2 : 6]; Arrays.fill(expected,state);
                assertArrayEquals(expected,values(f,type,reads)); assertArrayEquals(history,values(f,type,reads+1));
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class, names={"FLOAT32","FLOAT64"})
    void rankTwoMatmulScanHasIndependentProviderAndRepresentedExecutionEvidence(DataType executionType) {
        var q = withType(matmulQuery(),executionType);
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64,DataType.BFLOAT16,DataType.FLOAT16)) {
            var typed = withType(q,type);
            assertEquals(type == DataType.FLOAT32 || type == DataType.FLOAT64,new CpuCapabilityProvider().supports(typed));
            assertEquals(type == DataType.FLOAT32,new MetalCapabilityProvider().supports(typed));
        }
        for (int mode = 0; mode < 3; mode++) {
            var analysis = new CpuPartitionPreparer().analyze(context(q));
            try (var f = new CpuRecurrentScanConformanceTest.Fixture(analysis,mode,true,true,Map.of())) {
                fill(f,executionType,0,.25f); fill(f,executionType,1,2); fill(f,executionType,2,.5f);
                f.executable.bind(f.state).execute();
                double[] expected = new double[6]; Arrays.fill(expected,6.25f);
                assertArrayEquals(expected,values(f,executionType,3));
                double[] products = new double[18]; Arrays.fill(products,2f);
                assertArrayEquals(products,values(f,executionType,4));
            }
        }
    }

    static OperationCapabilityQuery matmulQuery() {
        Shape s = Shape.of(2,3), step = Shape.of(2,2), weight = Shape.of(2,3), stacked = Shape.of(3,2,3);
        var h = new ValueId(0); var x = new ValueId(1); var w = new ValueId(2); var p = new ValueId(3); var next = new ValueId(4);
        var values = List.of(new GraphValue(h,descriptor(DataType.FLOAT32,s)),new GraphValue(x,descriptor(DataType.FLOAT32,step)),
                new GraphValue(w,descriptor(DataType.FLOAT32,weight)),new GraphValue(p,descriptor(DataType.FLOAT32,s)),new GraphValue(next,descriptor(DataType.FLOAT32,s)));
        var product = new CompiledNode(new NodeId(0),new Operation(io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL,
                NoOperationAttrs.INSTANCE),List.of(x,w),List.of(p));
        var add = new CompiledNode(new NodeId(1),new Operation(BinaryArithmeticKind.ADD,NoOperationAttrs.INSTANCE),List.of(h,p),List.of(next));
        var residuals = List.of(h,p,next);
        var graph = new CompiledGraphModel(values,List.of(product,add),List.of(h,x,w),residuals,
                Map.of(product.id(),GraphPhase.FORWARD,add.id(),GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph,List.of(next),List.of(p),residuals,List.of(0,1));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN,new ScanAttrs(3,1,1,1,1,3,false)),
                List.of(descriptor(DataType.FLOAT32,s),descriptor(DataType.FLOAT32,Shape.of(3,2,2)),descriptor(DataType.FLOAT32,weight)),
                List.of(descriptor(DataType.FLOAT32,s),descriptor(DataType.FLOAT32,stacked),descriptor(DataType.FLOAT32,stacked),
                        descriptor(DataType.FLOAT32,stacked),descriptor(DataType.FLOAT32,stacked)),Optional.of(body),Optional.empty());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class, names={"FLOAT32","FLOAT64"})
    void broadcastSumLayoutOccurrenceHasIndependentTypedProviderAndExecutionWitnesses(DataType executionType) {
        var q = withType(shapedQuery(),executionType);
        assertTrue(new CpuCapabilityProvider().supports(q));
        for (var type : List.of(DataType.BFLOAT16,DataType.FLOAT16))
            assertFalse(new CpuCapabilityProvider().supports(withType(q,type)));
        assertEquals(executionType == DataType.FLOAT32,new MetalCapabilityProvider().supports(q));
        for (int mode = 0; mode < 3; mode++) {
            var analysis = new CpuPartitionPreparer().analyze(context(q));
            try (var f = new CpuRecurrentScanConformanceTest.Fixture(analysis,mode,true,true,Map.of())) {
                values(f,executionType,0,new double[] {1,2,3,4,5,6});
                values(f,executionType,1,new double[] {1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3});
                values(f,executionType,2,new double[] {.5f,1f,2f});
                f.executable.bind(f.state).execute();
                assertArrayEquals(new double[] {4,8,15,7,11,18},values(f,executionType,3));
                assertArrayEquals(new double[] {6,9,13,8,13,21,11,19,33},values(f,executionType,4));
            }
        }
    }

    static OperationCapabilityQuery shapedQuery() {
        Shape s = Shape.of(2,3), transposed = Shape.of(3,2), v = Shape.of(3);
        var h = new ValueId(0); var x = new ValueId(1); var w = new ValueId(2); var m = new ValueId(3);
        var next = new ValueId(4); var transpose = new ValueId(5); var dense = new ValueId(6); var sum = new ValueId(7);
        var values = List.of(new GraphValue(h,descriptor(DataType.FLOAT32,s)),new GraphValue(x,descriptor(DataType.FLOAT32,s)),
                new GraphValue(w,descriptor(DataType.FLOAT32,v)),new GraphValue(m,descriptor(DataType.FLOAT32,s)),
                new GraphValue(next,descriptor(DataType.FLOAT32,s)),new GraphValue(transpose,new TensorDescriptor(DataType.FLOAT32,transposed,
                    Optional.of(LayoutDescriptor.of(transposed,new long[] {1,3},0,true)),false)),
                new GraphValue(dense,descriptor(DataType.FLOAT32,transposed)),new GraphValue(sum,descriptor(DataType.FLOAT32,v)));
        var nodes = List.of(
                new CompiledNode(new NodeId(0),new Operation(BinaryArithmeticKind.MUL,NoOperationAttrs.INSTANCE),List.of(x,w),List.of(m)),
                new CompiledNode(new NodeId(1),new Operation(BinaryArithmeticKind.ADD,NoOperationAttrs.INSTANCE),List.of(h,m),List.of(next)),
                new CompiledNode(new NodeId(2),new Operation(io.github.pho001.synaptik.model.operation.layout.AxisTransformKind.PERMUTE,
                    new io.github.pho001.synaptik.model.operation.layout.PermutationAttrs(List.of(1,0))),List.of(next),List.of(transpose)),
                new CompiledNode(new NodeId(3),new Operation(io.github.pho001.synaptik.model.operation.layout.ContiguousKind.CONTIGUOUS,NoOperationAttrs.INSTANCE),List.of(transpose),List.of(dense)),
                new CompiledNode(new NodeId(4),new Operation(io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.SUM,
                    new io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs(1,false)),List.of(dense),List.of(sum)));
        var phases = new HashMap<NodeId,GraphPhase>(); nodes.forEach(n -> phases.put(n.id(),GraphPhase.FORWARD));
        var residuals = List.of(h,m,next,transpose,dense,sum);
        var graph = new CompiledGraphModel(values,nodes,List.of(h,x,w),residuals,phases);
        var body = new CompiledScanBody(graph,List.of(next),List.of(sum),residuals,List.of(0,1));
        var outputs = new ArrayList<TensorDescriptor>(); outputs.add(descriptor(DataType.FLOAT32,s)); outputs.add(descriptor(DataType.FLOAT32,Shape.of(3,3)));
        for (ValueId id : residuals) {
            Shape local = values.stream().filter(value -> value.id().equals(id)).findFirst().orElseThrow().descriptor().shape();
            long[] stacked = new long[local.rank()+1]; stacked[0] = 3; System.arraycopy(local.toLongArray(),0,stacked,1,local.rank());
            outputs.add(descriptor(DataType.FLOAT32,Shape.of(stacked)));
        }
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN,new ScanAttrs(3,1,1,1,1,6,false)),
                List.of(descriptor(DataType.FLOAT32,s),descriptor(DataType.FLOAT32,Shape.of(3,2,3)),descriptor(DataType.FLOAT32,v)),outputs,Optional.of(body),Optional.empty());
    }
    @Test void independentFloatingTypesAndCpuLowLockQualifyThroughActualProviders() {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16)) {
            var q = query(type, UnaryElementwiseKind.TANH, false, 3, Shape.of(2));
            assertEquals(type == DataType.FLOAT32 || type == DataType.FLOAT64, new CpuCapabilityProvider().supports(q));
            assertEquals(type == DataType.FLOAT32,new MetalCapabilityProvider().supports(q));
            assertEquals(type == DataType.FLOAT32 || type == DataType.FLOAT64,
                    new CpuCapabilityProvider().supports(query(type, UnaryElementwiseKind.RELU, false, 0, Shape.of(2))));
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class, names={"FLOAT32","FLOAT64"})
    void completeRoutePreservesExactHistorySitesBothDirectionsAndEmptyDomains(DataType executionType) {
        for (boolean reverse : List.of(false, true)) for (int t : List.of(0, 1, 3, 257))
            for (int n : List.of(0, 2)) for (int mode = 0; mode < 3; mode++) {
                var q = query(executionType, UnaryElementwiseKind.TANH, reverse, t, Shape.of(n));
                var analysis = new CpuPartitionPreparer().analyze(context(q));
                try (var f = new CpuRecurrentScanConformanceTest.Fixture(analysis, mode, true, true, Map.of())) {
                    double[] initial = new double[n], x = new double[t * n], expected = new double[n];
                    for (int i = 0; i < n; i++) expected[i] = initial[i] = -.25f + i * .5f;
                    for (int i = 0; i < x.length; i++) x[i] = (i % 9 - 4) * .03125f;
                    values(f,executionType,0, initial); values(f,executionType,1, x);
                    double[] old = new double[x.length], sums = new double[x.length], history = new double[x.length];
                    for (int k = 0; k < t; k++) { int time = reverse ? t - 1 - k : k;
                        for (int i = 0; i < n; i++) { int p = time * n + i;
                            old[p] = expected[i]; sums[p] = work(executionType,expected[i] + x[p]);
                            history[p] = expected[i] = work(executionType,Math.tanh(sums[p])); }
                    }
                    f.executable.bind(f.state).execute();
                    assertArrayEquals(expected, values(f,executionType,2)); assertArrayEquals(history, values(f,executionType,3));
                    assertArrayEquals(old, values(f,executionType,4)); assertArrayEquals(sums, values(f,executionType,5));
                    assertArrayEquals(history, values(f,executionType,6));
                    assertTrue(analysis.plan().units().getFirst().runtimeFacts().workspaceDeclaration().isEmpty());
                }
            }
    }

    static OperationCapabilityQuery query(DataType type, UnaryElementwiseKind unary, boolean reverse, long t, Shape s) {
        var dimensions = new ArrayList<Dimension>(); dimensions.add(new StaticDimension(t)); dimensions.addAll(s.dimensions());
        Shape stacked = Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
        var h = new ValueId(0); var x = new ValueId(1); var sum = new ValueId(2); var next = new ValueId(3);
        var values = List.of(new GraphValue(h, descriptor(type, s)), new GraphValue(x, descriptor(type, s)),
                new GraphValue(sum, descriptor(type, s)), new GraphValue(next, descriptor(type, s)));
        var add = new CompiledNode(new NodeId(0), new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE), List.of(h, x), List.of(sum));
        var activation = new CompiledNode(new NodeId(1), new Operation(unary, NoOperationAttrs.INSTANCE), List.of(sum), List.of(next));
        var graph = new CompiledGraphModel(values, List.of(add, activation), List.of(h, x), List.of(h, sum, next),
                Map.of(add.id(), GraphPhase.FORWARD, activation.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(next), List.of(next), List.of(h, sum, next), List.of(0,1,2,3,4));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, new ScanAttrs(t, 1, 1, 0, 1, 3, reverse)),
                List.of(descriptor(type, s), descriptor(type, stacked)),
                List.of(descriptor(type, s), descriptor(type, stacked), descriptor(type, stacked), descriptor(type, stacked), descriptor(type, stacked)), Optional.of(body),Optional.empty());
    }

    static OperationCapabilityQuery withType(OperationCapabilityQuery q, DataType type) {
        var b = q.scanBody().orElseThrow(); var g = b.graph();
        var graph = new CompiledGraphModel(g.values().stream().map(v -> new GraphValue(v.id(),
                new TensorDescriptor(type,v.descriptor().shape(),v.descriptor().layout(),v.descriptor().requiresGrad()))).toList(), g.nodes(), g.inputs(), g.outputs(), g.nodePhases());
        return new OperationCapabilityQuery(q.operation(), q.inputs().stream().map(d -> new TensorDescriptor(
                    d.dataType() == DataType.INT64 ? DataType.INT64 : type,d.shape(),d.layout(),d.requiresGrad())).toList(),
                q.outputs().stream().map(d -> new TensorDescriptor(type,d.shape(),d.layout(),d.requiresGrad())).toList(),
                Optional.of(new CompiledScanBody(graph, b.nextCarry(), b.emissions(), b.residuals(), b.demandedOutputIndices())),q.scanActivityPlan());
    }

    private static double work(DataType type, double value) { return type == DataType.FLOAT64 ? value : (float)value; }
    private static void values(CpuRecurrentScanConformanceTest.Fixture f, DataType type, int id, double... data) {
        for (int i = 0; i < data.length; i++) {
            if (type == DataType.FLOAT64) f.segment(id).setAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,i,data[i]);
            else f.segment(id).setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,i,(float)data[i]);
        }
    }
    private static double[] values(CpuRecurrentScanConformanceTest.Fixture f, DataType type, int id) {
        double[] data = new double[Math.toIntExact(f.segment(id).byteSize()/type.byteWidth())];
        for (int i = 0; i < data.length; i++) data[i] = type == DataType.FLOAT64
                ? f.segment(id).getAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,i) : f.segment(id).getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,i);
        return data;
    }
    private static void fill(CpuRecurrentScanConformanceTest.Fixture f, DataType type, int id, double value) {
        double[] data = new double[Math.toIntExact(f.segment(id).byteSize()/type.byteWidth())]; Arrays.fill(data,value); values(f,type,id,data);
    }

    private static TensorDescriptor descriptor(DataType type, Shape s) {
        return new TensorDescriptor(type, s, Optional.of(LayoutDescriptor.contiguous(s)), false);
    }

    static PrepareContext<CpuPartitionAnalysisInputs> context(OperationCapabilityQuery q) {
        return context(q,new CpuPartitionAnalysisInputs(false,List.of()));
    }

    /**
     * Projects the same Model-only query and its scoped activity with explicit cold CPU execution inputs.
     * @param q complete body-bearing query
     * @param inputs explicit immutable backend analysis inputs
     * @return fresh checked prepare context, without Compiler or physical storage
     */
    static PrepareContext<CpuPartitionAnalysisInputs> context(OperationCapabilityQuery q,CpuPartitionAnalysisInputs inputs) {
        int reads = q.inputs().size();
        var node = new CompiledNode(new NodeId(0), q.operation(), java.util.stream.IntStream.range(0,reads).mapToObj(ValueId::new).toList(),
                java.util.stream.IntStream.range(reads,reads+q.outputs().size()).mapToObj(ValueId::new).toList(), q.scanBody());
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
        var descriptors = new ArrayList<>(q.inputs()); descriptors.addAll(q.outputs());
        var values = new ArrayList<GraphValue>(); var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int p = 0; p < descriptors.size(); p++) {
            var id = new ValueId(p); var d = descriptors.get(p);
            values.add(new GraphValue(id, d)); memory.add(new LogicalMemoryRequirement(id, d,
                    p >= reads ? Optional.of(partition) : Optional.empty(), p < reads ? List.of(partition) : List.of(),
                    p >= reads && q.scanBody().orElseThrow().demandedOutputIndices().contains(p-reads)));
        }
        return new PrepareContext<>(partition, List.of(node), values, memory, Map.of(), inputs,
                q.scanActivityPlan().map(plan -> Map.of(node.id(), plan)).orElseGet(Map::of));
    }
}
