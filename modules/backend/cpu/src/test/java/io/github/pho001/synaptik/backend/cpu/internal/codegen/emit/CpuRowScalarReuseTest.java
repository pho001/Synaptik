package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.*;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Unmeasured scalar-reuse dominance, literal-control and matched-entry preflight coverage. */
class CpuRowScalarReuseTest {
    /**
     * Ordinary F32/F64 heap carriers use the same generic proof and reread changed invariants.
     * @throws Throwable if the exact generated typed entry fails
     */
    @Test void typedOrdinaryArrayStagingRereadsWeightsOnEveryInvocation() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var region = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRowMapReductionLoweringTest.candidate(type,false,7);
            var carrier = type == DataType.FLOAT64
                    ? io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY
                    : io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
            var spec = new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization(
                    io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint.fromHex(region.structuralKey()),
                    CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,Collections.nCopies(4,type),Collections.nCopies(4,carrier),
                    0,-1,List.of(),false,72,Optional.empty(),Optional.empty(),false,false,Optional.of(region));
            assertTrue(new String(spec.classIdentityBytes(),java.nio.charset.StandardCharsets.US_ASCII)
                    .contains("|ordinary-row-invariant-load-staging-v1="));
            var generator = new CpuClassFileKernelGenerator();
            var entry = generator.defineClassBytes(spec, region.encodedKernelIr(), generator.generateClassBytes(spec, region.encodedKernelIr())).entryPoint();
            Object x = type == DataType.FLOAT64 ? new double[14] : new float[14];
            Object w = type == DataType.FLOAT64 ? new double[4] : new float[4];
            Object rows = type == DataType.FLOAT64 ? new double[14] : new float[14];
            Object sums = type == DataType.FLOAT64 ? new double[4] : new float[4];
            for (int c = 0; c < 14; c++) java.lang.reflect.Array.set(x,c,type == DataType.FLOAT64 ? (Object)(c*.25) : (Object)(c*.25f));
            for (int invocation = 1; invocation <= 2; invocation++) {
                for (int c = 0; c < 4; c++) java.lang.reflect.Array.set(w,c,type == DataType.FLOAT64
                        ? (Object)(c%3 == 0 ? (double)invocation : 0d) : (Object)(c%3 == 0 ? (float)invocation : 0f));
                entry.invokeWithArguments(x,w,rows,sums,new long[]{7,0,0,0,0},0L,1L);
                for (int c = 0; c < 14; c++) assertEquals(2*(c*.25)*invocation,((Number)java.lang.reflect.Array.get(rows,c)).doubleValue(),0);
                for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
                    double expected = 0;
                    for (int r = 0; r < 7; r++) expected += ((2*r+i)*.25)*(2*((2*r+j)*.25));
                    assertEquals(expected,((Number)java.lang.reflect.Array.get(sums,2*i+j)).doubleValue(),0);
                }
            }
        }
    }

    /** Differing SHARED history values belong to original time, never traversal position. */
    @Test void scalarSharedHistoriesStageAtOriginalTimeInBothTraversals() {
        scalarTemporalHistory(false,false);
        scalarTemporalHistory(true,false);
    }

    /** A globally active time does not authorize reads in an entirely inactive worker range. */
    @Test void scalarHelperStagesOnlyAfterItsOwnFirstActiveConsumer() {
        scalarTemporalHistory(true,true);
    }

    /** Forwarded histories keep LONG bases even though they are not regional producing stores. */
    @Test void forwardedHistoryCallerLifetimeFallsBackBeforeStaging() {
        var type = DataType.FLOAT32;
        Tensor h = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(32,8));
        Tensor c = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,8));
        Tensor x = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(32,1));
        Tensor w = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,8));
        Tensor a = x.add(x), y = a.matmul(w);
        var body = new ScanBody(List.of(h,c),List.of(x),List.of(w),
                List.of(y,c.add(x.permute(1,0).matmul(y))),Collections.nCopies(60,x));
        Tensor xs = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,32,1));
        Tensor actualW = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,8));
        Tensor lengths = CpuSymbolicLengthScanFixture.tensor(DataType.INT64,false,Shape.of(32));
        var result = Scan.apply(List.of(CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(32,8)),
                CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,8))),List.of(xs),List.of(actualW),lengths,1,
                List.of(OptionalInt.of(0)),body,false);
        var roots = new ArrayList<Tensor>(result.finalCarry()); roots.addAll(result.outputs());
        var compiled = CpuGeneralScanCapabilityTest.compile(roots,Optional.empty(),false,new CpuCapabilityProvider());
        var analysis = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT));
        var spec = analysis.plan().units().stream().filter(u -> u.runtimeFacts().generalScan().isPresent())
                .findFirst().orElseThrow().portablePlan().specialization();
        var scan = spec.generalScanIr().orElseThrow();
        assertTrue(scan.stages().orElseThrow().rowMapReductionIr().isPresent());
        assertEquals(60,scan.demandedPorts().stream().filter(p -> p >= scan.carryCount()).count());
        assertEquals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,spec.executionStrategy());
        for (byte[] identity : List.of(spec.compatibilityBytes(),spec.classIdentityBytes())) assertFalse(
                new String(identity,java.nio.charset.StandardCharsets.US_ASCII).contains("-row-invariant-load-staging-v1="));
        try (var run = new CpuRecurrentScanExecutionTest.Fixture(analysis,1,true,true,Map.of())) {
            for (var input : compiled.constants().bindableInputBindings()) {
                var segment = run.storage.get(input.valueId());
                if (input.tensorId().equals(lengths.id())) for (int r = 0; r < 32; r++) segment.setAtIndex(ValueLayout.JAVA_LONG,r,1);
                else for (int i = 0; i < segment.byteSize()/4; i++) segment.setAtIndex(ValueLayout.JAVA_FLOAT,i,
                        input.tensorId().equals(xs.id()) ? .5f : input.tensorId().equals(actualW.id()) ? .25f : 0f);
            }
            run.executable.bind(run.state).execute();
            for (int p = 0; p < roots.size(); p++) {
                var segment = run.storage.get(compiled.publication().forwardBindings().get(p).valueId());
                for (int i = 0; i < segment.byteSize()/4; i++) assertEquals(p == 0 ? .25f : p == 1 ? 4f : .5f,
                        segment.getAtIndex(ValueLayout.JAVA_FLOAT,i),0,"forwarded lifetime port="+p);
            }
        }
    }

    /**
     * Twenty-five wrappers add 24 distinct carrier/base triples. Staging must not consume
     * the helper's last three free slots, and must not silently retire its eligible strategy.
     * @throws Exception if the existing private phase-bound cross-check cannot be inspected
     */
    @Test void duplicateWrapperHelperBudgetFallsBackWithoutStrategyShrink() throws Exception {
        var type = DataType.FLOAT32;
        Tensor row = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(4,1));
        Tensor carry = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,8));
        Tensor x = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(4,1));
        Tensor w = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,8));
        Tensor a = x.add(x), y = a.matmul(w);
        var body = new ScanBody(List.of(row,carry),List.of(x),List.of(w),
                List.of(row,carry.add(x.permute(1,0).matmul(y))),Collections.nCopies(25,y));
        Tensor xs = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,4,1));
        Tensor actualW = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,8));
        Tensor lengths = CpuSymbolicLengthScanFixture.tensor(DataType.INT64,false,Shape.of(4));
        var result = Scan.apply(List.of(CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(4,1)),
                CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(1,8))),List.of(xs),List.of(actualW),lengths,1,
                List.of(OptionalInt.of(0)),body,false);
        var roots = new ArrayList<Tensor>(); roots.add(result.finalCarry().get(1)); roots.addAll(result.outputs());
        var compiled = CpuGeneralScanCapabilityTest.compile(roots,Optional.empty(),false,new CpuCapabilityProvider());
        var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,2,2,1);
        var analysis = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),
                new CpuPartitionAnalysisInputs(false,List.of(),config)));
        var unit = analysis.plan().units().stream().filter(u -> u.runtimeFacts().generalScan().isPresent()).findFirst().orElseThrow();
        var spec = unit.portablePlan().specialization(); var scan = spec.generalScanIr().orElseThrow();
        var region = scan.stages().orElseThrow().rowMapReductionIr().orElseThrow();
        assertEquals(CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR,unit.executionStrategy());
        assertEquals(2,spec.rowFusionEmission().orElseThrow().maximumPartialCount());
        var helperBound = CpuPartitionPreparer.class.getDeclaredMethod("scanRowHelperSlots",region.getClass(),scan.getClass(),
                boolean.class,long.class,long.class,long.class,int.class);
        helperBound.setAccessible(true);
        assertEquals(125L,helperBound.invoke(null,region,scan,false,9L,8L,0L,0));
        for (byte[] identity : List.of(spec.compatibilityBytes(),spec.classIdentityBytes())) assertFalse(
                new String(identity,java.nio.charset.StandardCharsets.US_ASCII).contains("-row-invariant-load-staging-v1="));
        assertTrue(io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRowMapReductionLowering
                .invariantLoadStaging(region,List.of(scan.formalOrdinals().getLast()),List.of(),184,188).isEmpty());
        try (var run = new CpuRecurrentScanExecutionTest.Fixture(analysis,1,true,true,Map.of())) {
            for (var input : compiled.constants().bindableInputBindings()) {
                var segment = run.storage.get(input.valueId());
                if (input.tensorId().equals(lengths.id())) for (int r = 0; r < 4; r++) segment.setAtIndex(ValueLayout.JAVA_LONG,r,2);
                else for (int c = 0; c < segment.byteSize()/4; c++) segment.setAtIndex(ValueLayout.JAVA_FLOAT,c,
                        input.tensorId().equals(xs.id()) ? .5f : input.tensorId().equals(actualW.id()) ? .25f : 0f);
            }
            run.executable.bind(run.state).execute();
            for (int p = 0; p < roots.size(); p++) {
                var segment = run.storage.get(compiled.publication().forwardBindings().get(p).valueId());
                for (int c = 0; c < segment.byteSize()/4; c++) assertEquals(p == 0 ? 1f : .25f,
                        segment.getAtIndex(ValueLayout.JAVA_FLOAT,c),0,"checked no-staging wrapper="+p);
            }
        }
    }

    /**
     * Qualifies a non-square F2/H3 body against an independent ordered scalar equation.
     * First worker range [0,4) is entirely inactive beside active rows [4,7).
     * @param reverse whether original time decreases
     * @param parallel whether the fixed two-range scalar helper is selected
     */
    private static void scalarTemporalHistory(boolean reverse,boolean parallel) {
        var type = DataType.FLOAT32;
        Tensor x = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(7,2));
        Tensor q = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(3,2));
        Tensor row = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(7,2));
        Tensor sum = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(3));
        Tensor product = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,3));
        Tensor z = x.matmul(q.permute(1,0));
        var body = new ScanBody(List.of(row,sum,product),List.of(x,q),List.of(),
                List.of(x,sum.add(z.sumToShape(Shape.of(3))),product.add(x.permute(1,0).matmul(z))),List.of(z));
        Tensor initialRow = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(7,2));
        Tensor initialSum = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(3));
        Tensor initialProduct = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,3));
        Tensor xs = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(4,7,2));
        Tensor qs = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(4,3,2));
        Tensor lengths = CpuSymbolicLengthScanFixture.tensor(DataType.INT64,false,Shape.of(7));
        var result = Scan.apply(List.of(initialRow,initialSum,initialProduct),List.of(xs,qs),List.of(),lengths,1,
                List.of(OptionalInt.of(0),OptionalInt.empty()),body,reverse);
        var compiled = CpuGeneralScanCapabilityTest.compile(List.of(result.finalCarry().get(1),result.finalCarry().get(2),result.outputs().getFirst()),
                Optional.empty(),false,new CpuCapabilityProvider());
        var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,parallel ? 2 : 1,2,4);
        var analysis = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),
                new CpuPartitionAnalysisInputs(false,List.of(),config)));
        var spec = analysis.plan().units().stream().filter(u -> u.runtimeFacts().generalScan().isPresent())
                .findFirst().orElseThrow().portablePlan().specialization();
        assertEquals(parallel ? CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR
                : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,spec.executionStrategy());
        assertTrue(new String(spec.compatibilityBytes(),java.nio.charset.StandardCharsets.US_ASCII)
                .contains("|scan-row-invariant-load-staging-v1="));
        try (var run = new CpuRecurrentScanExecutionTest.Fixture(analysis,1,true,true,Map.of())) {
            for (var input : compiled.constants().bindableInputBindings()) {
                var segment = run.storage.get(input.valueId());
                if (input.tensorId().equals(lengths.id())) {
                    for (int r = 0; r < 7; r++) segment.setAtIndex(ValueLayout.JAVA_LONG,r,r < 4 ? 0 : 7-r);
                } else for (int c = 0; c < segment.byteSize()/4; c++) segment.setAtIndex(ValueLayout.JAVA_FLOAT,c,
                        input.tensorId().equals(xs.id()) ? (c%7-3)*.125f
                        : input.tensorId().equals(qs.id()) ? .125f*(1+c/6)*(1+c%6) : .25f);
            }
            run.executable.bind(run.state).execute();
            float[] expectedSum = {.25f,.25f,.25f}, expectedProduct = new float[6], expectedRows = new float[84];
            Arrays.fill(expectedProduct,.25f);
            for (int visit = 0; visit < 4; visit++) {
                int t = reverse ? 3-visit : visit; float[] stepSum = new float[3], stepProduct = new float[6]; boolean active = false;
                for (int r = 4; r < 7; r++) if (t < 7-r) {
                    active = true;
                    for (int j = 0; j < 3; j++) {
                        float value = 0;
                        for (int k = 0; k < 2; k++) value += (((t*7+r)*2+k)%7-3)*.125f*(.125f*(t+1)*(1+2*j+k));
                        expectedRows[(t*7+r)*3+j] = value; stepSum[j] += value;
                        for (int k = 0; k < 2; k++) stepProduct[3*k+j] += (((t*7+r)*2+k)%7-3)*.125f*value;
                    }
                }
                if (active) {
                    for (int c = 0; c < 3; c++) expectedSum[c] += stepSum[c];
                    for (int c = 0; c < 6; c++) expectedProduct[c] += stepProduct[c];
                }
            }
            var expected = List.of(expectedSum,expectedProduct,expectedRows);
            for (int p = 0; p < 3; p++) {
                var segment = run.storage.get(compiled.publication().forwardBindings().get(p).valueId());
                for (int c = 0; c < expected.get(p).length; c++) assertEquals(Float.floatToRawIntBits(expected.get(p)[c]),
                        Float.floatToRawIntBits(segment.getAtIndex(ValueLayout.JAVA_FLOAT,c)),"original-time raw port="+p+" cell="+c);
            }
        }
    }

    /** Four-case structural/cache/equality preflight; never calls the measured main/window. */
    @Test void authenticOldCandidateJavaPreflightAndLiteralSimdControls() throws Exception {
        String supplied = System.getenv("SYNAPTIK_CPU0075_BASELINE");
        org.junit.jupiter.api.Assumptions.assumeTrue(supplied != null,"authentic frozen external baseline required");
        var baseline = java.nio.file.Path.of(supplied);
        var root = java.nio.file.Files.createTempDirectory(baseline.getParent(),"source-preflight.");
        System.out.println("CPU0075 UNMEASURED evidence="+root);
        CpuRowScalarReuseComparison.preflight(baseline,root);
        var report = new StringBuilder();
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean scan : List.of(false,true))
                for (boolean parallel : List.of(false,true)) {
            int mode = parallel ? 3 : 1;
            String id = (scan ? "scan" : "ordinary")+"-"+type+"-B32-"+mode;
            var dir = java.nio.file.Files.createDirectory(root.resolve(id));
            try (var fixture = CpuRowScalarReuseComparison.fixture(scan,type,32,true,parallel,dir)) {
                byte[] before = java.nio.file.Files.readAllBytes(baseline.resolve("cpu0074-measure").resolve(id).resolve("selected.class"));
                byte[] current = java.nio.file.Files.readAllBytes(dir.resolve("selected.class"));
                assertArrayEquals(before,current,"literal SIMD CF, not a candidate-derived control: "+id);
                ((Runnable)CpuRowScalarReuseComparison.property(fixture,"generated")).run();
                ((Runnable)CpuRowScalarReuseComparison.property(fixture,"java")).run();
                ((Runnable)CpuRowScalarReuseComparison.property(fixture,"check")).run();
                report.append(id).append('\t').append(CpuRowScalarReuseComparison.sha(current)).append("\tliteral ClassFile and embedded key unchanged\n");
            }
        }
        java.nio.file.Files.writeString(root.resolve("SIMD-LITERAL-PASS.tsv"),report);
    }

    /**
     * The SUM singleton and ordinary branches must each define their own row*3 address.
     * The preceding contraction leaves a row*2 expression; its slot is not a dominating
     * definition of the other branch's row*3 value. Exercises both typed widths and B=1/>1.
     */
    @Test void externalSumAlternativesWithDifferentRowStridesHaveDominatingAddresses() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (int batch : List.of(1,3,7))
                for (boolean symbolic : List.of(false,true)) {
            var b = symbolic ? new DynamicDimension("B") : new StaticDimension(batch);
            Tensor x = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(b,new StaticDimension(2)));
            Tensor y = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(b,new StaticDimension(3)));
            Tensor product = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,2));
            Tensor sum = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(3));
            Tensor a = x.add(x);
            var body = new ScanBody(List.of(product,sum),List.of(x,y),List.of(),
                    List.of(product.add(x.permute(1,0).matmul(a)),sum.add(y.sumToShape(Shape.of(3)))),List.of(a));
            Tensor sx = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(new StaticDimension(2),b,new StaticDimension(2)));
            Tensor sy = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(new StaticDimension(2),b,new StaticDimension(3)));
            Tensor ip = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2,2));
            Tensor is = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(3));
            // All-valid symbolic F64 remains unqualified: do not invent admission for the test.
            if (symbolic && type == DataType.FLOAT64) continue;
            var result = Scan.apply(List.of(ip,is),List.of(sx,sy),List.of(),body,false);
            var compiled = CpuGeneralScanCapabilityTest.compile(
                    List.of(result.finalCarry().get(0),result.finalCarry().get(1),result.outputs().getFirst()),
                    Optional.empty(),false,new CpuCapabilityProvider());
            var analysis = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,
                    compiled.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT));
            var scan = analysis.plan().units().stream().filter(u -> u.runtimeFacts().generalScan().isPresent())
                    .findFirst().orElseThrow().runtimeFacts().generalScan().orElseThrow().ir();
            var region = scan.stages().orElseThrow().rowMapReductionIr().orElseThrow();
            assertEquals(1,region.sums().size(),"external SUM must execute in the fused row");
            assertEquals(1,region.contractions().size());
            try (var run = new CpuRecurrentScanExecutionTest.Fixture(analysis,1,true,true,Map.of(),
                    symbolic ? OptionalLong.of(batch) : OptionalLong.empty())) {
                for (var binding : compiled.constants().bindableInputBindings()) {
                    var segment = run.storage.get(binding.valueId());
                    for (int cell = 0; cell < segment.byteSize()/type.byteWidth(); cell++) {
                        double value = binding.tensorId().equals(sy.id()) ? cell+1 : binding.tensorId().equals(sx.id()) ? .25*(cell+1) : 0;
                        if (type == DataType.FLOAT64) segment.setAtIndex(ValueLayout.JAVA_DOUBLE,cell,value);
                        else segment.setAtIndex(ValueLayout.JAVA_FLOAT,cell,(float)value);
                    }
                }
                run.executable.bind(run.state).execute();
                var actual = run.storage.get(compiled.publication().forwardBindings().get(1).valueId());
                for (int j = 0; j < 3; j++) {
                    double expected = 0;
                    for (int t = 0; t < 2; t++) for (int r = 0; r < batch; r++) expected += (t*batch+r)*3+j+1;
                    double value = type == DataType.FLOAT64 ? actual.getAtIndex(ValueLayout.JAVA_DOUBLE,j) : actual.getAtIndex(ValueLayout.JAVA_FLOAT,j);
                    assertEquals(expected,value,0,type+" B="+batch+" symbolic="+symbolic+" cell="+j);
                }
            }
        }
    }
}
