package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import static java.lang.foreign.ValueLayout.*;
import io.github.pho001.synaptik.backend.contract.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest;
import io.github.pho001.synaptik.compiler.*;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Direct numerical qualification of ordinary AD kernels reused across invocation batches. */
class CpuOrdinaryDynamicBatchTest {
    /** Ordinary symbolic MATMUL authenticates its namespace through actual finalization. */
    @Test void ordinaryMatmulRetainsTypedEnvelopeAndRealizesEmptyAndNonemptyBatch() {
        Tensor x = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, symbolic(4), Optional.empty(), false));
        Tensor w = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, Shape.of(4, 3),
                Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(Shape.of(4, 3))), false));
        Tensor y = x.matmul(w);
        var backend = CpuCapabilityProvider.CPU_BACKEND_ID;
        var compiled = GraphCompilationPort.compile(CompileMode.FORWARD_ONLY, List.of(y), Optional.empty(),
                GraphOptimizationConfig.standard(), BackendIntent.unconstrained(), PartitionScoringConfig.neutral(),
                List.of(new CpuCapabilityProvider()), List.of(new BackendAvailabilitySnapshot(backend,
                        Map.of(new BackendDeviceId(backend, "cpu"), DeviceClass.CPU))));
        var analysis = new CpuPartitionPreparer().analyze(
                GraphPreparation.project(compiled, compiled.partitions().getFirst(), CpuPartitionAnalysisInputs.DEFAULT));
        var unit = analysis.plan().units().getFirst();
        var route = unit.portablePlan();
        assertEquals(route.specialization().matmulIr().orElseThrow().encodedKernelIr(), route.kernelIr());
        assertTrue(route.kernelIr().familyIdentity().endsWith(":ordinary-batch-v1"));
        for (int batch : new int[]{0, 3}) {
            try (var fixture = new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(
                    analysis, 0, false, false, Map.of(), java.util.OptionalLong.of(batch))) {
                var left = fixture.storage.get(unit.boundaryValues().get(0));
                var right = fixture.storage.get(unit.boundaryValues().get(1));
                var output = fixture.storage.get(unit.boundaryValues().get(2));
                for (int i = 0; i < batch * 4; i++) left.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i, i % 4 + 1);
                for (int i = 0; i < 12; i++) right.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i, 1);
                fixture.executable.bind(fixture.state).execute();
                for (int i = 0; i < batch * 3; i++)
                    assertEquals(10f, output.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i));
            }
        }
        var ir = route.kernelIr(); var s = route.specialization();
        var wrong = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr(
                ir.values(), ir.instructions(), ir.loop(), ir.stores(), ir.familyIdentity().replace(":ordinary-batch-v1", ""));
        var forged = new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization(
                io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint.fromHex(wrong.structuralKey()),
                s.executionStrategy(), s.boundaryDataTypes(), s.carrierPattern(), 0, -1, List.of(), false, 54,
                s.matmulIr(), Optional.empty(), true);
        var generator = new CpuClassFileKernelGenerator();
        byte[] current = generator.generateClassBytes(s, ir);
        assertThrows(IllegalArgumentException.class, () -> generator.generateClassBytes(forged, wrong));
        assertThrows(IllegalArgumentException.class, () -> generator.defineClassBytes(forged, wrong, current));
    }
    @Test void typedOuterPredicatesUseExactOccurrencesWithoutOriginAndValidatedGenerationRequiresTypedNamespace() {
        var b = new DynamicDimension("B");
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            Shape row = Shape.ofDimensions(b,new StaticDimension(3)), stack = Shape.ofDimensions(new StaticDimension(4),b,new StaticDimension(3));
            Shape rowOne = Shape.ofDimensions(new StaticDimension(3),b), rowOneStack = Shape.ofDimensions(new StaticDimension(4),new StaticDimension(3),b);
            var provider = new CpuCapabilityProvider();
            var yes = List.of(
                    occurrence(type,new Operation(io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.ADD,NoOperationAttrs.INSTANCE),List.of(row,row),row),
                    occurrence(type,new Operation(io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.MUL,NoOperationAttrs.INSTANCE),List.of(stack,Shape.of(3)),stack),
                    occurrence(type,new Operation(io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.EXPAND,new io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs(rowOneStack)),List.of(Shape.scalar()),rowOneStack),
                    occurrence(type,new Operation(io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.EXPAND,new io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs(rowOne)),List.of(Shape.scalar()),rowOne),
                    occurrence(type,new Operation(AggregateReductionKind.SUM,new AxisReductionAttrs(1,false)),List.of(stack),Shape.of(4,3)),
                    occurrence(type,new Operation(io.github.pho001.synaptik.model.operation.layout.AxisTransformKind.PERMUTE,new io.github.pho001.synaptik.model.operation.layout.PermutationAttrs(List.of(1,0))),List.of(row),rowOne));
            for (var q : yes) { io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertCurrentSupport(provider, q, q.toString());
                assertEquals(provider.supports(q),provider.supports(new io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery(q.operation(),q.inputs(),q.outputs())),"no Scan/AD-origin authority"); }
            var control = new io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery(new Operation(io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind.CAST,
                    new io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs(DataType.INT64)),
                    List.of(new TensorDescriptor(type,Shape.ofDimensions(b),Optional.empty(),false)),
                    List.of(new TensorDescriptor(DataType.INT64,Shape.ofDimensions(b),Optional.empty(),false)));
            io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertCurrentSupport(provider, control);
            var gradControl = new io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery(control.operation(),
                    List.of(new TensorDescriptor(type,Shape.ofDimensions(b),Optional.empty(),true)),control.outputs());
            assertFalse(provider.supports(gradControl));
            var other = occurrence(type,new Operation(io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.ADD,NoOperationAttrs.INSTANCE),
                    List.of(row,Shape.ofDimensions(new DynamicDimension("Other"),new StaticDimension(3))),row);
            assertFalse(provider.supports(other));
            if (type == DataType.FLOAT64) {
                assertFalse(provider.supports(occurrence(type,new Operation(io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL,NoOperationAttrs.INSTANCE),List.of(row,Shape.of(3,2)),Shape.ofDimensions(b,new StaticDimension(2)))));
                assertFalse(provider.supports(occurrence(type,new Operation(io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind.TANH,NoOperationAttrs.INSTANCE),List.of(row),row)));
                assertFalse(provider.supports(occurrence(type,new Operation(io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.ADD,NoOperationAttrs.INSTANCE),List.of(rowOneStack,rowOneStack),rowOneStack)));
            }
        }
        var route = aggregateTyped(DataType.FLOAT64,new Operation(AggregateReductionKind.MEAN,NoOperationAttrs.INSTANCE),symbolic(3),Shape.scalar());
        var ir = route.kernelIr(); var s = route.specialization();
        var wrong = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr(ir.values(),ir.instructions(),ir.loop(),ir.stores(),
                ir.familyIdentity().replace(":ordinary-batch-typed-v1",":ordinary-batch-v1"));
        var forged = new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization(
                io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint.fromHex(wrong.structuralKey()),s.executionStrategy(),s.boundaryDataTypes(),s.carrierPattern(),
                0,-1,List.of(),true,s.classIdentitySchema(),s.matmulIr(),s.generalScanIr(),true,true);
        assertThrows(IllegalArgumentException.class,() -> new CpuClassFileKernelGenerator().generateClassBytes(forged,wrong));
    }

    private static io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery occurrence(DataType type,Operation operation,List<Shape> inputs,Shape output) {
        return new io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery(operation,
                inputs.stream().map(shape -> new TensorDescriptor(type,shape,Optional.empty(),false)).toList(),
                List.of(new TensorDescriptor(type,output,Optional.empty(),false)));
    }

    /**
     * Checks typed exact reductions and retains ClassFiles under the optional generalScan evidence
     * root; without an override, the historical Prepare0015 destination remains the default.
     * @throws Throwable if generation, execution, numerical checks or evidence writing fails
     */
    @Test void typedFloat64ExactSumAndRationalMeanUse34LimbsWithoutChangingStaticSizing() throws Throwable {
        for (boolean mean : List.of(false,true)) for (int axis : new int[]{-1,0,1}) {
            Shape input = symbolic(3), output = axis == -1 ? Shape.scalar() : axis == 0 ? Shape.of(3)
                    : Shape.ofDimensions(new DynamicDimension("B"));
            var route = aggregateTyped(DataType.FLOAT64,new Operation(mean ? AggregateReductionKind.MEAN : AggregateReductionKind.SUM,
                    axis == -1 ? NoOperationAttrs.INSTANCE : new AxisReductionAttrs(axis,false)),input,output);
            byte[] bytes = bytes(route);
            // An opt-in destination keeps diagnostic reruns from replacing historical artifacts.
            var evidence = java.nio.file.Path.of(System.getProperty("synaptik.cpu.generalScan.evidenceRoot",
                    "/private/tmp/synaptik-prepare0015.gVDqlx")).resolve("classfiles"); java.nio.file.Files.createDirectories(evidence);
            java.nio.file.Files.write(evidence.resolve("ordinary-F64-"+mean+"-axis"+axis+".class"),bytes);
            java.nio.file.Files.write(evidence.resolve("ordinary-F64-"+mean+"-axis"+axis+".identity.bytes"),route.specialization().classIdentityBytes());
            java.nio.file.Files.write(evidence.resolve("ordinary-F64-"+mean+"-axis"+axis+".compatibility.bytes"),route.specialization().compatibilityBytes());
            assertTrue(route.kernelIr().familyIdentity().endsWith(":ordinary-batch-typed-v1"));
            for (byte[] identity : List.of(route.specialization().compatibilityBytes(),route.specialization().classIdentityBytes()))
                assertTrue(new String(identity,java.nio.charset.StandardCharsets.US_ASCII).contains("ordinary-batch-typed-v1"));
            var entry = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), route.kernelIr(), bytes).entryPoint();
            for (int batch : new int[]{0,1,7,32}) for (int dataCase = 0; dataCase < 5; dataCase++) {
                var g = route.batchAggregate().orElseThrow().bind(batch);
                assertEquals(34,g.stateLimbCount()); assertEquals(280,g.scratchSliceBytes());
                assertEquals(g.outputCount() == 0 ? 0 : 280,route.batchAggregate().orElseThrow().byteSize(batch));
                double[] data = new double[batch*3];
                double[][] cases = {{Double.MAX_VALUE,1,-Double.MAX_VALUE,Double.MIN_VALUE,-Double.MIN_VALUE,0x1.0000000000001p0},
                        {-0d,-0d,-0d},{Double.NaN,1,2},{Double.POSITIVE_INFINITY,Double.NEGATIVE_INFINITY,1},
                        {Double.MIN_VALUE,Double.MIN_VALUE,0,1,0x1.0p-53,0x1.0p-54}};
                for (int i = 0; i < data.length; i++) data[i] = cases[dataCase][i%cases[dataCase].length];
                var source = MemorySegment.ofArray(data); var actual = MemorySegment.ofArray(new double[(int)g.outputCount()]);
                var direct = MemorySegment.ofArray(new double[(int)g.outputCount()]);
                try (var arena = Arena.ofConfined()) {
                    entry.invokeExact(source,actual,arena.allocate(g.outputCount() == 0 ? 0 : 280,8),g.pack(new long[2]),0L,g.outputCount());
                    int domain = axis == -1 ? batch*3 : axis == 0 ? batch : 3;
                    int cellStep = axis == -1 ? 0 : axis == 0 ? 1 : 3, termStep = axis == 0 ? 3 : 1;
                    CpuOrdinaryDynamicBatchCleanJavaOracle.reduceFloat64(source,direct,arena.allocate(280,8),(int)g.outputCount(),domain,cellStep,termStep,mean);
                    for (int cell = 0; cell < g.outputCount(); cell++) {
                        double[] contributors = new double[domain]; for (int n = 0; n < domain; n++) contributors[n] = data[cell*cellStep+n*termStep];
                        double expected = decimalExact(contributors,mean), observed = actual.getAtIndex(JAVA_DOUBLE,cell);
                        if (Double.isNaN(expected)) assertTrue(Double.isNaN(observed));
                        else assertEquals(Double.doubleToRawLongBits(expected),Double.doubleToRawLongBits(observed));
                        assertEquals(Double.doubleToLongBits(observed),Double.doubleToLongBits(direct.getAtIndex(JAVA_DOUBLE,cell)),"independent semantic oracle also qualifies the timed clean-Java algorithm");
                    }
                }
            }
        }
        var staticShape = Shape.of(3); var context = CpuNonAffineMovementLoweringTest.context(new Operation(AggregateReductionKind.SUM,NoOperationAttrs.INSTANCE),List.of(0),
                List.of(new TensorDescriptor(DataType.FLOAT64,staticShape,Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(staticShape)),false)),
                new TensorDescriptor(DataType.FLOAT64,Shape.scalar(),Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(Shape.scalar())),false));
        var staticGeometry = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().runtimeFacts().aggregateGeometry().orElseThrow();
        assertEquals(33,staticGeometry.stateLimbCount()); assertEquals(272,staticGeometry.scratchSliceBytes());
    }

    @Test void typedFloat64RankThreeExactAxesAndSingletonSumToShapeRetainAllRepresentedCases() throws Throwable {
        Shape sourceShape = Shape.ofDimensions(new StaticDimension(4),new DynamicDimension("B"),new StaticDimension(3));
        for (List<Integer> axes : List.of(List.of(0),List.of(1),List.of(2),List.of(0,1),List.of(0,2),List.of(1,2),List.of(0,1,2))) {
            var dimensions = new ArrayList<Dimension>(); for (int i = 0; i < 3; i++) if (!axes.contains(i)) dimensions.add(sourceShape.dimension(i));
            Shape target = Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
            for (boolean mean : List.of(false,true)) {
                var op = new Operation(mean ? AggregateReductionKind.MEAN : AggregateReductionKind.SUM,new MultiAxisReductionAttrs(axes,false));
                var route = aggregateTyped(DataType.FLOAT64,op,sourceShape,target); byte[] bytes = bytes(route);
                var entry = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), route.kernelIr(), bytes).entryPoint();
                for (int batch : new int[]{0,1,7}) try (var arena = Arena.ofConfined()) {
                    var g = route.batchAggregate().orElseThrow().bind(batch); double[] data = new double[12*batch];
                    for (int i = 0; i < data.length; i++) data[i] = (i%11-5)*.125+Double.MIN_VALUE;
                    var input = MemorySegment.ofArray(data); var actual = MemorySegment.ofArray(new double[(int)g.outputCount()]);
                    entry.invokeExact(input,actual,arena.allocate(route.batchAggregate().orElseThrow().byteSize(batch),8),g.pack(new long[2]),0L,g.outputCount());
                    var contributors = new ArrayList<List<Double>>(); for (int i = 0; i < g.outputCount(); i++) contributors.add(new ArrayList<>());
                    for (int t = 0; t < 4; t++) for (int r = 0; r < batch; r++) for (int c = 0; c < 3; c++) {
                        int out = 0; int[] coordinate = {t,r,c}, extents = {4,batch,3};
                        for (int axis = 0; axis < 3; axis++) if (!axes.contains(axis)) out = out*extents[axis]+coordinate[axis];
                        contributors.get(out).add(data[(t*batch+r)*3+c]);
                    }
                    for (int cell = 0; cell < g.outputCount(); cell++) {
                        double expected = decimalExact(contributors.get(cell).stream().mapToDouble(Double::doubleValue).toArray(),mean);
                        assertEquals(Double.doubleToLongBits(expected),Double.doubleToLongBits(actual.getAtIndex(JAVA_DOUBLE,cell)),"axes="+axes+" mean="+mean+" B="+batch);
                    }
                }
            }
        }
        var route = aggregateTyped(DataType.FLOAT64,new Operation(AggregateReductionKind.SUM,new SumToShapeAttrs(Shape.of(1,3))),symbolic(3),Shape.of(1,3));
        var entry = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), route.kernelIr(), bytes(route)).entryPoint();
        for (int batch : new int[]{0,1,7}) try (var arena = Arena.ofConfined()) {
            double[] data = new double[batch*3]; if (batch > 0) { data[0] = -0d; data[1] = Double.longBitsToDouble(0x7ff8123456789abcL); data[2] = Double.MIN_VALUE; }
            var input = MemorySegment.ofArray(data); var actual = MemorySegment.ofArray(new double[3]); var g = route.batchAggregate().orElseThrow().bind(batch);
            entry.invokeExact(input,actual,arena.allocate(280,8),g.pack(new long[2]),0L,3L);
            if (batch == 1) assertEquals(-1,input.mismatch(actual),"singleton is raw movement, not an addition to +0");
            else if (batch == 0) assertArrayEquals(new double[3],actual.toArray(JAVA_DOUBLE));
            else { assertEquals(0d,actual.getAtIndex(JAVA_DOUBLE,0)); assertTrue(Double.isNaN(actual.getAtIndex(JAVA_DOUBLE,1))); assertEquals(Double.MIN_VALUE,actual.getAtIndex(JAVA_DOUBLE,2)); }
        }
    }

    /** Independent decimal arithmetic oracle; 1200 digits exceed the entire double exponent span. */
    private static double decimalExact(double[] values,boolean mean) {
        boolean positiveInfinity = false, negativeInfinity = false, negativeZeroOnly = values.length > 0;
        var sum = java.math.BigDecimal.ZERO;
        for (double value : values) {
            if (Double.isNaN(value)) return Double.NaN;
            if (value == Double.POSITIVE_INFINITY) positiveInfinity = true;
            else if (value == Double.NEGATIVE_INFINITY) negativeInfinity = true;
            else sum = sum.add(new java.math.BigDecimal(value));
            negativeZeroOnly &= Double.doubleToRawLongBits(value) == Long.MIN_VALUE;
        }
        if (positiveInfinity && negativeInfinity || mean && values.length == 0) return Double.NaN;
        if (positiveInfinity) return Double.POSITIVE_INFINITY; if (negativeInfinity) return Double.NEGATIVE_INFINITY;
        if (sum.signum() == 0 && negativeZeroOnly) return -0d;
        return (mean ? sum.divide(java.math.BigDecimal.valueOf(values.length),new java.math.MathContext(1200,java.math.RoundingMode.HALF_EVEN)) : sum).doubleValue();
    }

    static CpuPortableRoutePlan aggregateTyped(DataType type,Operation operation,Shape input,Shape output) {
        var context = CpuNonAffineMovementLoweringTest.context(operation,List.of(0),List.of(new TensorDescriptor(type,input,Optional.empty(),false)),
                new TensorDescriptor(type,output,Optional.empty(),false));
        return new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
    }

    @Test void fixedFactMatmulsMatchCleanJavaAndReuseOneEntryAcrossInvocationB() throws Throwable {
        for (int[] size : new int[][] {{4,3},{128,128}}) {
            int f = size[0], h = size[1];
            var matrices = routes(f,h).stream().filter(r -> r.batchMatmul().isPresent()).toList();
            assertEquals(3, matrices.size());
            for (var route : matrices) {
                byte[] bytes = bytes(route);
                var entry = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), route.kernelIr(), bytes).entryPoint();
                for (int b : new int[] {64,40,1,64,0}) {
                    var g = route.batchMatmul().orElseThrow().bind(b);
                    float[] xv = varied(b*f), wv = varied(f*h), dyv = varied(b*h);
                    MemorySegment x = MemorySegment.ofArray(xv), w = MemorySegment.ofArray(wv), dy = MemorySegment.ofArray(dyv);
                    MemorySegment left, right;
                    int family;
                    if (route.batchMatmul().orElseThrow().left().extentBatchFactors().get(1) == 1) {
                        left=x; right=dy; family=2;
                    } else if (route.batchMatmul().orElseThrow().right().extentConstants().get(1) == f
                            && route.batchMatmul().orElseThrow().right().strideConstants().get(0) == 1) {
                        left=dy; right=w; family=1;
                    } else { left=x; right=w; family=0; }
                    MemorySegment actual = MemorySegment.ofArray(new float[Math.toIntExact(g.outputCount())]);
                    MemorySegment expected = MemorySegment.ofArray(new float[Math.toIntExact(g.outputCount())]);
                    entry.invokeExact(left, right, actual, g.pack(0,0,0), 0L, g.outputCount());
                    oracle(f,h,family,x,w,dy,expected,b);
                    assertEquals(-1, actual.mismatch(expected), "matrix family="+family+" B="+b);
                }
            }
        }
        assertNotEquals(routes(4,3).getFirst().specialization().structuralKey(),
                routes(128,128).getFirst().specialization().structuralKey());
    }

    @Test void exactFiveLimbFullSelectedAndSumToShapeMatchCleanJavaAtZeroSingletonAndNonuniformDomains() throws Throwable {
        for (boolean mean : List.of(false,true)) for (int axis : new int[] {-1,0,1}) {
            Shape input = symbolic(3), output = axis == -1 ? Shape.scalar() : axis == 0 ? Shape.of(3)
                    : Shape.ofDimensions(new DynamicDimension("B"));
            var operation = new Operation(mean ? AggregateReductionKind.MEAN : AggregateReductionKind.SUM,
                    axis == -1 ? NoOperationAttrs.INSTANCE : new AxisReductionAttrs(axis, false));
            var route = aggregate(operation,input,output);
            var entry = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), route.kernelIr(), bytes(route)).entryPoint();
            for (int b : new int[] {0,1,40,64}) {
                var g=route.batchAggregate().orElseThrow().bind(b);
                assertEquals(5,g.stateLimbCount()); assertEquals(48,g.scratchSliceBytes());
                assertEquals(g.outputCount()==0?0:48,route.batchAggregate().orElseThrow().byteSize(b));
                float[] data = varied(b*3);
                if (data.length>=6) { data[0]=Float.MAX_VALUE; data[1]=1; data[2]=-Float.MAX_VALUE; data[3]=Float.MIN_VALUE; }
                MemorySegment source=MemorySegment.ofArray(data), actual=MemorySegment.ofArray(new float[(int)g.outputCount()]),
                        expected=MemorySegment.ofArray(new float[(int)g.outputCount()]);
                try (Arena arena=Arena.ofConfined()) {
                    MemorySegment scratch=arena.allocate(48,8), oracleScratch=arena.allocate(48,8);
                    entry.invokeExact(source,actual,scratch,g.pack(new long[2]),0L,g.outputCount());
                    CpuOrdinaryDynamicBatchCleanJavaOracle.reduce(source,expected,oracleScratch,(int)g.outputCount(),
                            axis==-1?b*3:axis==0?b:3,axis==-1?0:axis==0?1:3,axis==0?3:1,mean);
                    assertEquals(-1,actual.mismatch(expected),"axis="+axis+" mean="+mean+" B="+b);
                }
            }
        }
        var route=aggregate(new Operation(AggregateReductionKind.SUM,new SumToShapeAttrs(Shape.of(1,3))),symbolic(3),Shape.of(1,3));
        var entry=new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), route.kernelIr(), bytes(route)).entryPoint();
        for(int b:new int[]{0,1,40}) try(Arena arena=Arena.ofConfined()) {
            float[] data=varied(b*3); if(b==1) data[0]=Float.intBitsToFloat(0x7fc01234);
            var source=MemorySegment.ofArray(data); var actual=MemorySegment.ofArray(new float[3]);
            var g=route.batchAggregate().orElseThrow().bind(b);
            entry.invokeExact(source,actual,arena.allocate(48,8),g.pack(new long[2]),0L,3L);
            if(b==1) assertEquals(-1,source.mismatch(actual),"SUM-to-Shape B1 is a raw copy including NaN payload");
            else { var expected=MemorySegment.ofArray(new float[3]);
                CpuOrdinaryDynamicBatchCleanJavaOracle.reduce(source,expected,arena.allocate(48,8),3,b,1,3,false);
                assertEquals(-1,actual.mismatch(expected)); }
        }
    }

    @Test void matmulFullRangesAndSubrangesPreserveUntouchedPartialRows() throws Throwable {
        for (var route : routes(4,3).stream().filter(r -> r.batchMatmul().isPresent()).toList()) {
            byte[] bytes = bytes(route);
            var geometry = route.batchMatmul().orElseThrow().bind(5);
            MemorySegment a = MemorySegment.ofArray(varied(Math.toIntExact(geometry.leftSpan())));
            MemorySegment b = MemorySegment.ofArray(varied(Math.toIntExact(geometry.rightSpan())));
            int count = Math.toIntExact(geometry.outputCount());
            int width = Math.toIntExact(geometry.n());
            var entry = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), route.kernelIr(), bytes).entryPoint();
            float[] reference = new float[count];
            for (int row = 0; row < geometry.m(); row++) for (int column = 0; column < width; column++) {
                float sum = 0;
                for (int k = 0; k < geometry.k(); k++) {
                    long ai = geometry.leftOffset() + row * geometry.leftMStride() + k * geometry.leftKStride();
                    long bi = geometry.rightOffset() + k * geometry.rightKStride() + column * geometry.rightNStride();
                    sum += a.get(JAVA_FLOAT_UNALIGNED, ai * 4) * b.get(JAVA_FLOAT_UNALIGNED, bi * 4);
                }
                reference[row * width + column] = sum;
            }
            for (long[] range : new long[][] {{0,count}, {1,width-1}, {1,width+1},
                    {width,width*2L}, {width-1,count-1}, {count-1,count}, {1,1}}) {
                float[] actual = new float[count]; Arrays.fill(actual, Float.intBitsToFloat(0x7fc01234));
                MemorySegment output = MemorySegment.ofArray(actual);
                entry.invokeExact(a,b,output,geometry.pack(0,0,0),range[0],range[1]);
                for (int i = 0; i < count; i++) assertEquals(
                        i >= range[0] && i < range[1] ? Float.floatToRawIntBits(reference[i]) : 0x7fc01234,
                        Float.floatToRawIntBits(actual[i]), "range="+Arrays.toString(range)+" cell="+i);
            }
        }
    }

    static CpuPortableRoutePlan aggregate(Operation op,Shape input,Shape output) {
        var context=CpuNonAffineMovementLoweringTest.context(op,List.of(0),List.of(descriptor(input)),descriptor(output));
        return new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
    }
    /**
     * Builds standalone forward/dX/dW controls with a direct tensor cotangent. A mean-loss
     * derivative now has a computed row producer and is an eligible CPU0073 group, so it no
     * longer isolates the three unchanged whole-MATMUL entry contracts tested here. Public
     * mean-loss execution remains covered independently by the Engine and conformance fixtures.
     * @param f fixed feature count
     * @param h fixed output width
     * @return standalone unchanged matrix routes plus any movement of the explicit seed
     */
    static List<CpuPortableRoutePlan> routes(int f,int h) {
        Tensor x=TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,symbolic(f),Optional.empty(),true));
        Tensor w=TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,Shape.of(f,h),
                Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(Shape.of(f,h))),true));
        Tensor y=x.matmul(w),seed=TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,symbolic(h),Optional.empty(),false));
        var stage=new FunctionalGradientRequest.Stage(List.of(new FunctionalGradientRequest.ForwardTensorReference(y)),
                List.of(Optional.of(seed)),List.of(x,w),false,FunctionalGradientRequest.DisconnectedPolicy.ERROR);
        var backend=CpuCapabilityProvider.CPU_BACKEND_ID;
        var artifacts=GraphCompilationPort.compile(CompileMode.FORWARD_AND_BACKWARD,List.of(y),
                Optional.of(new FunctionalGradientRequest(List.of(stage))),GraphOptimizationConfig.standard(),
                BackendIntent.unconstrained(),PartitionScoringConfig.neutral(),List.of(new CpuCapabilityProvider()),
                List.of(new BackendAvailabilitySnapshot(backend,Map.of(new BackendDeviceId(backend,"cpu"),DeviceClass.CPU))));
        return artifacts.partitions().stream().flatMap(p -> new CpuPartitionPreparer().analyze(
                GraphPreparation.project(artifacts,p,CpuPartitionAnalysisInputs.DEFAULT)).plan().units().stream())
                .map(CpuPartitionPreparationPlan.ExecutionUnitPlan::portablePlan).toList();
    }
    static Shape symbolic(int f) { return Shape.ofDimensions(new DynamicDimension("B"),new StaticDimension(f)); }
    static TensorDescriptor descriptor(Shape shape) { return new TensorDescriptor(DataType.FLOAT32,shape,Optional.empty(),false); }
    static byte[] bytes(CpuPortableRoutePlan route) { return new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(),route.kernelIr()); }
    static float[] varied(int n) { float[] a=new float[n]; for(int i=0;i<n;i++)a[i]=(i%13-6)*.125f; return a; }
    static void oracle(int f,int h,int family,MemorySegment x,MemorySegment w,MemorySegment dy,MemorySegment out,int b) {
        if(f==4&&h==3) {
            if(family==0)CpuOrdinaryDynamicBatchCleanJavaOracle.matmul4x3(x,w,out,b);
            else if(family==1)CpuOrdinaryDynamicBatchCleanJavaOracle.dx4x3(dy,w,out,b);
            else CpuOrdinaryDynamicBatchCleanJavaOracle.dw4x3(x,dy,out,b);
        } else {
            if(family==0)CpuOrdinaryDynamicBatchCleanJavaOracle.matmul128x128(x,w,out,b);
            else if(family==1)CpuOrdinaryDynamicBatchCleanJavaOracle.dx128x128(dy,w,out,b);
            else CpuOrdinaryDynamicBatchCleanJavaOracle.dw128x128(x,dy,out,b);
        }
    }
}
