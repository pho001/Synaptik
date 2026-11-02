package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.tensor;
import io.github.pho001.synaptik.backend.cpu.*;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.*;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Opt-in joint production-bound AD measurements; genuine raw forward probes remain separate. */
public class CpuGeneralScanTimingTest {
    private static volatile float consumed;
    private static volatile double consumedDouble;

    /** JFR batch interval; allocation/commit occurs outside the 10,000-call timed loop. */
    @jdk.jfr.Name("scan.Block") @jdk.jfr.StackTrace(false) @jdk.jfr.Threshold("0 ns")
    static class ScanBlock extends jdk.jfr.Event {
        /** Fixed phase, warmup or measurement. */ public String phase;
        /** Exact implementation identity, G or O. */ public String side;
        /** Zero-based phase pair. */ public int pair;
        /** Calls in this batch. */ public int reps;
        /** Timed loop start in monotonic nanoseconds. */ public long startNano;
        /** Timed loop end in monotonic nanoseconds. */ public long endNano;
        /** JVM uptime immediately before the timed loop, milliseconds. */ public long startUptimeMillis;
        /** JVM uptime immediately after the timed loop, milliseconds. */ public long endUptimeMillis;
    }

    /** Clock brackets correlate JVM uptime, monotonic time and JFR event wall time. */
    @jdk.jfr.Name("scan.Clock") @jdk.jfr.StackTrace(false)
    static class ScanClock extends jdk.jfr.Event {
        /** Phase boundary name. */ public String phase;
        /** Monotonic bracket start. */ public long nanoBefore;
        /** Monotonic bracket end. */ public long nanoAfter;
        /** JVM uptime sampled within the bracket. */ public long uptimeMillis;
    }

    /**
     * Launches only the CPU0036 bounded matched-work probe in a fresh JVM. The caller supplies
     * the evidence directory and explicit opt-in property; historical timing methods never run.
     * @param args empty; all inputs and counts are fixed by the verified CPU0036 brief
     * @throws Exception if opt-in, evidence, generation, output checks or diagnostic capture fails
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 0 || !Boolean.getBoolean("synaptik.cpu.generalScan.performance"))
            throw new IllegalArgumentException("CPU0036 requires explicit opt-in and no workload arguments");
        new CpuGeneralScanTimingTest().boundedSharedFloat64ShapedForwardCpu0036();
    }

    /**
     * Measures the actual production-generated v4 F64 shaped forward against the B4/H8 clean
     * Java oracle with the same entry ABI and runtime-long T. The generic oracle and its
     * historical evidence are unchanged; they do not qualify this equally specialized pair.
     * Both sides borrow identical shared native inputs, write distinct nonaliasing shared native
     * outputs, and need zero workspace. Each paired batch performs 10,000 calls
     * per side; 100 warmup pairs precede all 300 retained measured pairs. Side order alternates.
     * No pilot, trial, forced compilation or fixed-byte patching is performed. The coordinator
     * launches exactly three fresh sequential JVMs after independent review, never from this test.
     * Normal compilation evidence uses external JFR/LogCompilation and clock/block correlation;
     * no in-process diagnostic attachment can interrupt warmup or measurement.
     * @throws Exception if generation/definition, exact outputs, diagnostics or evidence output fails
     */
    @Test @EnabledIfSystemProperty(named="synaptik.cpu.generalScan.performance", matches="true")
    void boundedSharedFloat64ShapedForwardCpu0036() throws Exception {
        final int t = 8, batch = 4, width = 8, cells = batch*width;
        final int warmup = 100, samples = 300, calls = 10_000;
        var type = io.github.pho001.synaptik.model.datatype.DataType.FLOAT64;
        Path root = Path.of(System.getProperty("synaptik.cpu.generalScan.evidenceRoot")).toAbsolutePath(); Files.createDirectories(root);
        var policy = "CPU0036 F64 shaped forward next=h+xStep*w; runtime-long T=8 static B=4 H=8; matched entry ABI\n"
                +"inputs=Arena.ofShared/native; generated outputs=Arena.ofShared/native; oracle outputs=Arena.ofShared/native\n"
                +"Both output sets use the same shared arena policy, distinct nonaliasing allocations, same final-carry plus stacked-emission demand.\n"
                +"Both workspace policies=zero bytes; no binding/public Engine measurements or subtraction.\n"
                +"warmup_pairs=100 measured_pairs=300 calls_per_side_per_pair=10000; alternating paired side order\n"
                +"Only production generation/definition in this fresh JVM; no frozen-byte patch, pilot, trial or forced C2.\n";
        Files.writeString(root.resolve("policy.txt"),policy); System.out.print(policy);
        jdk.jfr.EventType.getEventType(ScanBlock.class); jdk.jfr.EventType.getEventType(ScanClock.class);
        clock("setup");
        var report = new StringBuilder(policy).append("pid=").append(ProcessHandle.current().pid())
                .append(" runtime=").append(System.getProperty("java.runtime.version"))
                .append(" vm=").append(System.getProperty("java.vm.name"))
                .append(" os=").append(System.getProperty("os.name")).append(' ').append(System.getProperty("os.arch"))
                .append("\nflags=").append(java.lang.management.ManagementFactory.getRuntimeMXBean().getInputArguments()).append('\n');
        var facts = CpuGeneralScanGeneratedKernelTest.broadcastFacts(type,t,batch,width,false,true);
        var spec = CpuGeneralScanGeneratedKernelTest.specialization(facts);
        assertEquals(0,facts.geometry().scratchBytes()); assertFalse(spec.scratchParameter());
        assertEquals(List.of(0,1),facts.ir().demandedPorts()); assertEquals(5,facts.spans().size());
        var generator = new CpuClassFileKernelGenerator(); long started = System.nanoTime();
        byte[] bytes = generator.generateClassBytes(spec,facts.ir().encodedKernelIr());
        long generation = System.nanoTime()-started; started = System.nanoTime();
        var generated = generator.defineClassBytes(spec, spec.generalScanIr().orElseThrow().encodedKernelIr(), bytes); long loading = System.nanoTime()-started;
        Files.writeString(root.resolve("identity.txt"),"class="+generated.hiddenClass().getName()+"\nhash="
                +CpuGeneralScanGeneratedKernelTest.sha256(bytes)+"\nentry="+generated.entryPoint().type()+"\n");
        CpuGeneralScanGeneratedKernelTest.assertDoubleClass(bytes);
        Files.write(root.resolve("production.class"),bytes);
        Files.write(root.resolve("compatibility.bytes"),spec.compatibilityBytes());
        Files.write(root.resolve("class-identity.bytes"),spec.classIdentityBytes());
        Files.writeString(root.resolve("facts.txt"),facts.toString());
        var javap = new ProcessBuilder(Path.of(System.getProperty("java.home"),"bin","javap").toString(),"-c","-v","-p",root.resolve("production.class").toString())
                .redirectErrorStream(true).start();
        Files.writeString(root.resolve("production.javap.txt"),new String(javap.getInputStream().readAllBytes())); assertEquals(0,javap.waitFor());
        report.append("schema=").append(io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema.CURRENT_VERSION)
                .append(" family=").append(spec.classIdentitySchema()).append(" payload=").append(facts.ir().bakedSpecializationIdentity())
                .append("\nkey=").append(spec.structuralKey()).append(" class_sha256=").append(CpuGeneralScanGeneratedKernelTest.sha256(bytes))
                .append("\ngeneration_ns=").append(generation).append(" definition_ns=").append(loading).append('\n');
        try (var inputs = Arena.ofShared(); var outputs = Arena.ofShared()) {
            MemorySegment initial = inputs.allocate(cells*8L,8), sequence = inputs.allocate(t*cells*8L,8), weight = inputs.allocate(width*8L,8);
            MemorySegment result = outputs.allocate(cells*8L,8), emissions = outputs.allocate(t*cells*8L,8);
            MemorySegment oracleResult = outputs.allocate(cells*8L,8), oracleEmissions = outputs.allocate(t*cells*8L,8);
            for (int i = 0; i < cells; i++) initial.setAtIndex(ValueLayout.JAVA_DOUBLE,i,(i-7)*.03125+0x1p-40);
            for (int i = 0; i < t*cells; i++) sequence.setAtIndex(ValueLayout.JAVA_DOUBLE,i,(i%19-9)*.015625+0x1p-40);
            for (int i = 0; i < width; i++) weight.setAtIndex(ValueLayout.JAVA_DOUBLE,i,(i+1)*.0625+0x1p-40);
            MethodHandle handle = generated.entryPoint(); long[] geometry = {t,cells};
            Runnable kernel = () -> {
                try { handle.invokeExact(initial,sequence,weight,result,emissions,geometry,0L,(long)cells); }
                catch (RuntimeException | Error e) { throw e; } catch (Throwable e) { throw new AssertionError(e); }
            };
            Runnable oracle = () -> CpuGeneralScanCleanJavaOracle.broadcastForwardFloat64B4H8(initial,sequence,weight,oracleResult,oracleEmissions,geometry,0L,(long)cells);
            started = System.nanoTime(); kernel.run(); long first = System.nanoTime()-started; oracle.run();
            assertEquals(-1L,result.mismatch(oracleResult)); assertEquals(-1L,emissions.mismatch(oracleEmissions));
            report.append("first_entry_ns=").append(first).append(" exact_outputs_before=true\n");
            var blocks = new ArrayList<ScanBlock>(2*(warmup+samples)); clock("warmup-start");
            for (int i = 0; i < warmup; i++) {
                if ((i&1) == 0) { blocks.add(measureBatch(kernel,calls,"warmup",i,"G")); blocks.add(measureBatch(oracle,calls,"warmup",i,"O")); }
                else { blocks.add(measureBatch(oracle,calls,"warmup",i,"O")); blocks.add(measureBatch(kernel,calls,"warmup",i,"G")); }
            }
            clock("warmup-end");
            long[] gn = new long[samples], on = new long[samples];
            var csv = new StringBuilder("pair,first,calls_per_side,generated_total_ns,oracle_total_ns,generated_ns_per_call,oracle_ns_per_call,ratio,generated_start_nano,generated_end_nano,oracle_start_nano,oracle_end_nano,generated_start_uptime_ms,generated_end_uptime_ms,oracle_start_uptime_ms,oracle_end_uptime_ms\n");
            clock("measurement-start");
            for (int i = 0; i < samples; i++) {
                ScanBlock g, o;
                if ((i&1) == 0) { blocks.add(g = measureBatch(kernel,calls,"measurement",i,"G")); blocks.add(o = measureBatch(oracle,calls,"measurement",i,"O")); }
                else { blocks.add(o = measureBatch(oracle,calls,"measurement",i,"O")); blocks.add(g = measureBatch(kernel,calls,"measurement",i,"G")); }
                gn[i] = g.endNano-g.startNano; on[i] = o.endNano-o.startNano;
                csv.append(i).append(',').append((i&1) == 0 ? "generated" : "oracle").append(',').append(calls).append(',')
                        .append(gn[i]).append(',').append(on[i]).append(',').append(gn[i]/(double)calls).append(',')
                        .append(on[i]/(double)calls).append(',').append(gn[i]/(double)on[i]).append(',')
                        .append(g.startNano).append(',').append(g.endNano).append(',').append(o.startNano).append(',').append(o.endNano).append(',')
                        .append(g.startUptimeMillis).append(',').append(g.endUptimeMillis).append(',').append(o.startUptimeMillis).append(',').append(o.endUptimeMillis).append('\n');
                consumedDouble = result.get(ValueLayout.JAVA_DOUBLE,0)+oracleResult.get(ValueLayout.JAVA_DOUBLE,0);
                if ((i+1)%50 == 0) System.out.println("CPU0036 measured pairs "+(i+1)+"/300 pid="+ProcessHandle.current().pid());
            }
            clock("measurement-end");
            Files.writeString(root.resolve("paired-samples.csv"),csv);
            var blockCsv = new StringBuilder("phase,pair,side,reps,startNano,endNano,elapsedNs\n");
            for (var block : blocks) blockCsv.append(block.phase).append(',').append(block.pair).append(',').append(block.side).append(',')
                    .append(block.reps).append(',').append(block.startNano).append(',').append(block.endNano).append(',').append(block.endNano-block.startNano).append('\n');
            Files.writeString(root.resolve("blocks.csv"),blockCsv);
            kernel.run(); oracle.run();
            assertEquals(-1L,result.mismatch(oracleResult)); assertEquals(-1L,emissions.mismatch(oracleEmissions));
            report.append("exact_outputs_after=true\n");
            for (int[] window : List.of(new int[] {0,300},new int[] {0,100},new int[] {100,200},new int[] {200,300})) {
                int from = window[0], to = window[1];
                double[] g = new double[to-from], o = new double[to-from], ratios = new double[to-from];
                for (int i = from; i < to; i++) { g[i-from] = gn[i]/(double)calls; o[i-from] = on[i]/(double)calls; ratios[i-from] = gn[i]/(double)on[i]; }
                report.append("window=[").append(from).append(',').append(to).append(") generated_ns ").append(distribution(g))
                        .append(" oracle_ns ").append(distribution(o)).append(" paired_ratio ").append(distribution(ratios)).append('\n');
            }
        }
        report.append("All 300 pairs retained. Normal tiered compilation evidence is observational, not a parity threshold.\n")
                .append("Cold generation, definition and first entry are separate; no inferred binding/Engine costs.\n");
        Files.writeString(root.resolve("summary.txt"),report); System.out.print(report);
        clock("complete");
    }

    /**
     * Records elapsed batch time without truncating each call's cost.
     * @param call prebound generated or oracle invocation
     * @param count exact positive call count
     * @param phase fixed warmup or measurement phase
     * @param pair zero-based pair index
     * @param side exact implementation identity, G or O
     * @return committed batch event retaining exact start/end times and uptime brackets
     */
    private static ScanBlock measureBatch(Runnable call, int count, String phase, int pair, String side) {
        var event = new ScanBlock(); event.phase = phase; event.pair = pair; event.side = side; event.reps = count;
        event.begin(); event.startUptimeMillis = java.lang.management.ManagementFactory.getRuntimeMXBean().getUptime();
        event.startNano = System.nanoTime(); for (int i = 0; i < count; i++) call.run(); event.endNano = System.nanoTime();
        event.endUptimeMillis = java.lang.management.ManagementFactory.getRuntimeMXBean().getUptime();
        event.end(); event.commit(); return event;
    }

    /**
     * Commits a clock bracket without changing compilation state.
     * @param phase fixed phase-boundary name
     */
    private static void clock(String phase) {
        var event = new ScanClock(); event.phase = phase; event.begin(); event.nanoBefore = System.nanoTime();
        event.uptimeMillis = java.lang.management.ManagementFactory.getRuntimeMXBean().getUptime();
        event.nanoAfter = System.nanoTime(); event.end(); event.commit();
    }

    /**
     * Summarizes every value in a window without trimming outliers.
     * @param values nonempty finite samples in original order, borrowed
     * @return full-distribution count/min/quantiles/max/mean/population-standard-deviation text
     */
    private static String distribution(double[] values) {
        double mean = Arrays.stream(values).average().orElseThrow(), variance = 0;
        for (double value : values) variance += (value-mean)*(value-mean);
        double[] sorted = values.clone(); Arrays.sort(sorted);
        return "n="+values.length+" min="+sorted[0]+" p50="+sorted[(int)Math.ceil(.50*sorted.length)-1]
                +" p90="+sorted[(int)Math.ceil(.90*sorted.length)-1]+" p95="+sorted[(int)Math.ceil(.95*sorted.length)-1]
                +" p99="+sorted[(int)Math.ceil(.99*sorted.length)-1]+" max="+sorted[sorted.length-1]
                +" mean="+mean+" sd="+Math.sqrt(variance/values.length);
    }

    /** Measures the complete bound pipeline against independent complete forward/adjoint work. */
    @Test @EnabledIfSystemProperty(named="synaptik.cpu.generalScan.performance",matches="true")
    void symbolicLengthForwardReversePrepare0015() throws Exception {
        long deadline=System.nanoTime()+60_000_000_000L;
        var report=new StringBuilder("Joint production binding/preflight/helper execution; independent Java forward plus adjoint. No phase-kernel ABI, historical-byte parity, subtraction or timing threshold.\n");
        for (var type:List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            for (boolean reverse:List.of(false,true)) {
                var family=new CpuSymbolicLengthScanFixture.Family(type,reverse);
                for (int batch:new int[]{32,7}) try (var fixture=new CpuSymbolicLengthScanFixture(family,batch)) {
                    fixture.verify();
                    Runnable oracle=() -> { fixture.oracleForward(); fixture.oracleReverse(); };
                    paired(type+"-reverse"+reverse+"-B"+batch+"-joint",fixture.preboundExecution(),oracle,20,90,100,deadline,report);
                    fixture.verifyLatest();
                }
            }
            for (boolean mean:List.of(false,true)) aggregate(type,mean,deadline,report);
        }
        retain("prepare0015-report.txt",report);
    }

    @Test @EnabledIfSystemProperty(named="synaptik.cpu.generalScan.performance",matches="true")
    void lengthAwareForwardReverseCpu0042() throws Exception {
        for (var type:List.of(DataType.FLOAT32,DataType.FLOAT64)) probe(type,1,5,3,8,8,true,1,true,2000,100,10);
    }
    @Test @EnabledIfSystemProperty(named="synaptik.cpu.generalScan.performance",matches="true")
    void matchedFloat64PointwiseForwardAndActualReverse() throws Exception { probe(DataType.FLOAT64,0,8,1,32,32,true,1,false,8000,100,3); }
    @Test @EnabledIfSystemProperty(named="synaptik.cpu.generalScan.performance",matches="true")
    void matchedFloat64ShapedForwardAndActualReverse() throws Exception { probe(DataType.FLOAT64,1,8,4,8,8,true,1,false,8000,100,3); }
    @Test @EnabledIfSystemProperty(named="synaptik.cpu.generalScan.performance",matches="true")
    void matchedFloat64MatmulForwardAndActualReverse() throws Exception { probe(DataType.FLOAT64,2,8,4,6,8,true,1,false,8000,100,3); }
    @Test @EnabledIfSystemProperty(named="synaptik.cpu.generalScan.performance",matches="true")
    void matchedMatmulForwardAndActualReverse() throws Exception {
        for (int[] d:List.of(new int[]{4,2,3,4},new int[]{4,16,24,32}))
            probe(DataType.FLOAT32,2,d[0],d[1],d[2],d[3],true,2,false,1200,d[1]==2?100:20,3);
    }
    @Test @EnabledIfSystemProperty(named="synaptik.cpu.generalScan.performance",matches="true")
    void matchedShapedBroadcastReverseAndLargerReduction() throws Exception {
        for (int[] d:List.of(new int[]{8,2,8},new int[]{8,64,128}))
            probe(DataType.FLOAT32,1,d[0],d[1],d[2],d[2],true,2,false,d[1]==2?1500:100,d[1]==2?100:5,5);
    }
    @Test @EnabledIfSystemProperty(named="synaptik.cpu.generalScan.pointwisePerformance",matches="true")
    void matchedSegmentOnlyForwardAndActualReverse() throws Exception {
        for (int[] d:List.of(new int[]{8,32},new int[]{64,256})) for (boolean backward:List.of(false,true)) for (int mode:List.of(1,2))
            probe(DataType.FLOAT32,0,d[0],1,d[1],d[1],backward,mode,false,d[0]==8?5000:500,d[0]==8?400:20,7);
    }

    /** Retains every original family/domain/carrier case, measuring the honest joint ABI. */
    private static void probe(DataType type,int family,int time,int rows,int k,int n,boolean backward,int mode,
            boolean lengthAware,int warmup,int repetitions,int samples) throws Exception {
        long[] stateShape=family==0 ? new long[]{n} : new long[]{rows,n};
        long[] stepShape=family==0 ? new long[]{n} : new long[]{rows,k};
        long[] weightShape=family==2 ? new long[]{k,n} : new long[]{n};
        Tensor h=tensor(type,true,stateShape), x=tensor(type,true,stepShape), w=tensor(type,true,weightShape);
        Tensor next=family==0 ? h.add(x).tanh() : h.add(family==1 ? x.mul(w) : x.matmul(w));
        var body=new ScanBody(List.of(h),List.of(x),family==0?List.of():List.of(w),List.of(next),List.of(next));
        long[] stack=new long[stepShape.length+1]; stack[0]=time; System.arraycopy(stepShape,0,stack,1,stepShape.length);
        Tensor initial=tensor(type,true,stateShape), sequence=tensor(type,true,stack), weight=tensor(type,true,weightShape);
        Tensor lengths=tensor(DataType.INT64,false,rows);
        var invariants=family==0 ? List.<Tensor>of() : List.of(weight);
        var scan=lengthAware ? Scan.apply(List.of(initial),List.of(sequence),invariants,lengths,1,List.of(OptionalInt.of(0)),body,false)
                : Scan.apply(List.of(initial),List.of(sequence),invariants,body,false);
        var roots=List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
        Tensor fs=tensor(type,false,stateShape), es=tensor(type,false,roots.get(1).descriptor().shape().toLongArray());
        var targets=family==0 ? List.of(initial,sequence) : List.of(initial,sequence,weight);
        var request=new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(roots.stream()
                .map(v -> (FunctionalGradientRequest.OutputReference)new FunctionalGradientRequest.ForwardTensorReference(v)).toList(),
                List.of(Optional.of(fs),Optional.of(es)),targets,false,FunctionalGradientRequest.DisconnectedPolicy.ZERO)));
        var artifact=CpuGeneralScanCapabilityTest.compile(roots,backward?Optional.of(request):Optional.empty(),false,new CpuCapabilityProvider());
        var analysis=new CpuPartitionPreparer().analyze(GraphPreparation.project(artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT));
        int cells=rows*n, width=type.byteWidth();
        try (var fixture=new CpuRecurrentScanExecutionTest.Fixture(analysis,mode,false,false,Map.of()); var arena=Arena.ofConfined()) {
            var ids=new HashMap<TensorId,ValueId>(); artifact.constants().bindableInputBindings().forEach(b -> ids.put(b.tensorId(),b.valueId()));
            MemorySegment hi=fixture.storage.get(ids.get(initial.id())), xs=fixture.storage.get(ids.get(sequence.id()));
            MemorySegment weights=family==0?null:fixture.storage.get(ids.get(weight.id()));
            MemorySegment finalSeed=backward?fixture.storage.get(ids.get(fs.id())):null, emissionSeed=backward?fixture.storage.get(ids.get(es.id())):null;
            double epsilon=type==DataType.FLOAT64 ? 0x1p-40 : 0;
            fill(type,hi,(type==DataType.FLOAT64?.03125:family==2?.125:.2)+epsilon);
            fill(type,xs,(type==DataType.FLOAT64?.03125:family==2?.03125:.125)+epsilon);
            if (weights!=null) fill(type,weights,(family==1 && type==DataType.FLOAT32?.75:.0625)+epsilon);
            if (backward) { fill(type,finalSeed,.5+epsilon); fill(type,emissionSeed,.25+epsilon); }
            MemorySegment controls=lengthAware?fixture.storage.get(ids.get(lengths.id())):null;
            if (lengthAware) {
                controls.copyFrom(MemorySegment.ofArray(new long[]{5,3,0}));
                for (var input:List.of(hi,xs,weights,finalSeed,emissionSeed)) for (long i=0;i<input.byteSize()/width;i++) put(type,input,i,.125+(i%7)*.015625);
            }
            for (var constant:artifact.constants().constantSources()) if (fixture.storage.containsKey(constant.valueId()))
                fill(type,fixture.storage.get(constant.valueId()),type==DataType.FLOAT64?constant.value().float64Value():constant.value().float32Value());
            MemorySegment result=arena.allocate((long)cells*width,width), emissions=arena.allocate((long)time*cells*width,width);
            MemorySegment saved=arena.allocate((long)time*cells*width,width), dh=arena.allocate((long)cells*width,width);
            MemorySegment dx=arena.allocate((long)time*rows*k*width,width), dw=arena.allocate((long)(family==2?k*n:n)*width,width);
            // Independent reference initializers/work are not production boundary operands.
            MemorySegment zero=arena.allocate(width,width), work=arena.allocate(2L*width*(cells+(family==2?k*n:n)),width);
            long[] geometry={time};
            Runnable forward=() -> {
                if (lengthAware) {
                    if (type==DataType.FLOAT64) CpuGeneralScanCleanJavaOracle.lengthForwardFloat64B3F8(hi,xs,weights,controls,result,emissions,geometry,0,cells);
                    else CpuGeneralScanCleanJavaOracle.lengthForwardFloat32B3F8(hi,xs,weights,controls,result,emissions,geometry,0,cells);
                } else if (family==0) {
                    if (type==DataType.FLOAT64) CpuGeneralScanCleanJavaOracle.forwardFloat64(hi,xs,result,emissions,saved,time,cells,false);
                    else CpuGeneralScanCleanJavaOracle.forward(hi,xs,result,emissions,saved,time,cells,false);
                } else if (family==2) {
                    if (type==DataType.FLOAT64) CpuGeneralScanCleanJavaOracle.matmulForwardFloat64(hi,xs,weights,result,emissions,time,rows,k,n);
                    else CpuGeneralScanCleanJavaOracle.matmulForward(hi,xs,weights,result,emissions,time,rows,k,n);
                } else if (type==DataType.FLOAT64) CpuGeneralScanCleanJavaOracle.broadcastForwardFloat64(hi,xs,weights,result,emissions,time,rows,n);
                else {
                    result.copyFrom(hi);
                    for (int t=0;t<time;t++) for (int i=0;i<cells;i++) {
                        float value=(float)get(type,result,i)+(float)get(type,xs,(long)t*cells+i)*(float)get(type,weights,i%n);
                        put(type,result,i,value); put(type,emissions,(long)t*cells+i,value);
                    }
                }
            };
            Runnable oracle=() -> {
                forward.run(); if (!backward) return;
                if (lengthAware) {
                    if (type==DataType.FLOAT64) CpuGeneralScanCleanJavaOracle.lengthReverseFloat64B3F8(finalSeed,zero,xs,emissionSeed,weights,controls,dh,dw,dx,geometry,0,cells);
                    else CpuGeneralScanCleanJavaOracle.lengthReverseFloat32B3F8(finalSeed,zero,xs,emissionSeed,weights,controls,dh,dw,dx,geometry,0,cells);
                } else if (family==0) {
                    if (type==DataType.FLOAT64) CpuGeneralScanCleanJavaOracle.reverseFloat64(finalSeed,emissionSeed,saved,dh,dx,time,cells,false);
                    else CpuGeneralScanCleanJavaOracle.reverse(finalSeed,emissionSeed,saved,dh,dx,time,cells,false);
                } else if (family==1) {
                    if (type==DataType.FLOAT64) CpuGeneralScanCleanJavaOracle.broadcastReverseFloat64(finalSeed,emissionSeed,xs,weights,zero,dh,dw,dx,work,time,rows,n);
                    else CpuGeneralScanCleanJavaOracle.broadcastReverse(finalSeed,emissionSeed,xs,weights,zero,dh,dw,dx,work,time,rows,n);
                } else {
                    if (type==DataType.FLOAT64) CpuGeneralScanCleanJavaOracle.matmulReverseFloat64(finalSeed,emissionSeed,xs,weights,zero,dh,dw,dx,work,time,rows,k,n);
                    else CpuGeneralScanCleanJavaOracle.matmulReverse(finalSeed,emissionSeed,xs,weights,zero,dh,dw,dx,work,time,rows,k,n);
                }
            };
            long cold=System.nanoTime(); var bound=fixture.executable.bind(fixture.state); long binding=System.nanoTime()-cold;
            Runnable generated=bound::execute;
            var expected=new ArrayList<MemorySegment>(List.of(result,emissions));
            if (backward) { expected.add(dh); expected.add(dx); if (family!=0) expected.add(dw); }
            var publications=new ArrayList<ValueId>(); artifact.publication().forwardBindings().forEach(p -> publications.add(p.valueId()));
            artifact.publication().gradientBindings().forEach(p -> publications.add(p.valueId()));
            Runnable compare=() -> { for (int p=0;p<expected.size();p++) {
                var actual=fixture.storage.get(publications.get(p));
                if (type==DataType.FLOAT64) assertArrayEquals(expected.get(p).toArray(ValueLayout.JAVA_DOUBLE),actual.toArray(ValueLayout.JAVA_DOUBLE));
                else assertArrayEquals(expected.get(p).toArray(ValueLayout.JAVA_FLOAT),actual.toArray(ValueLayout.JAVA_FLOAT));
            }};
            generated.run(); oracle.run(); compare.run();
            for (int i=0;i<warmup;i++) { if ((i&1)==0) { generated.run(); oracle.run(); } else { oracle.run(); generated.run(); } }
            long[] gn=new long[samples], on=new long[samples];
            for (int sample=0;sample<samples;sample++) {
                if ((sample&1)==0) { gn[sample]=measure(generated,repetitions); on[sample]=measure(oracle,repetitions); }
                else { on[sample]=measure(oracle,repetitions); gn[sample]=measure(generated,repetitions); }
                compare.run();
            }
            String name=type+"-family"+family+"-T"+time+"B"+rows+"K"+k+"N"+n+"-AD"+backward+"-mode"+mode+"-length"+lengthAware;
            retain(name+".txt",new StringBuilder("Complete production-bound forward/requested-AD including preflight; independent complete Java reference. No separate reverse-kernel ABI or old-byte claim.\n")
                    .append("binding_ns=").append(binding).append(" bound_joint_ns=").append(Arrays.toString(gn)).append(" oracle_joint_ns=").append(Arrays.toString(on)).append('\n'));
        }
    }

    private static void aggregate(DataType type,boolean mean,long deadline,StringBuilder report) throws Exception {
        var route=CpuOrdinaryDynamicBatchTest.aggregateTyped(type,new io.github.pho001.synaptik.model.operation.Operation(
                mean?io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.MEAN:io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.SUM,
                io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE),CpuOrdinaryDynamicBatchTest.symbolic(2),io.github.pho001.synaptik.model.shape.Shape.scalar());
        var generator=new CpuClassFileKernelGenerator();
        var entry=generator.defineClassBytes(route.specialization(),route.kernelIr(),generator.generateClassBytes(route.specialization(),route.kernelIr())).entryPoint();
        for (int batch:new int[]{32,7}) try (var arena=Arena.ofShared()) {
            var geometry=route.batchAggregate().orElseThrow().bind(batch); long[] pack=geometry.pack(new long[2]);
            var input=arena.allocate(2L*batch*type.byteWidth(),8); var actual=arena.allocate(type.byteWidth(),8); var expected=arena.allocate(type.byteWidth(),8);
            var scratch=arena.allocate(geometry.scratchSliceBytes(),8); var referenceScratch=arena.allocate(geometry.scratchSliceBytes(),8);
            for (int i=0;i<2*batch;i++) put(type,input,i,(i%13-6)*.125);
            Runnable generated=() -> { try { entry.invokeExact(input,actual,scratch,pack,0L,1L); } catch (Throwable failure) { throw new AssertionError(failure); } };
            Runnable oracle=type==DataType.FLOAT64 ? () -> CpuOrdinaryDynamicBatchCleanJavaOracle.reduceFloat64(input,expected,referenceScratch,1,batch*2,0,1,mean)
                    : () -> CpuOrdinaryDynamicBatchCleanJavaOracle.reduce(input,expected,referenceScratch,1,batch*2,0,1,mean);
            generated.run(); oracle.run(); assertEquals(-1,actual.mismatch(expected));
            paired(type+"-B"+batch+"-exact-"+(mean?"mean":"sum"),generated,oracle,20,90,100,deadline,report);
            assertEquals(-1,actual.mismatch(expected));
        }
    }
    private static void paired(String name,Runnable generated,Runnable oracle,int warmup,int samples,int repetitions,long deadline,StringBuilder report) {
        for (int pair=0;pair<warmup+samples;pair++) {
            if (System.nanoTime()>=deadline) { report.append(name).append(" censored=true\n"); return; }
            long g,o;
            if ((pair&1)==0) { g=measure(generated,repetitions); o=measure(oracle,repetitions); }
            else { o=measure(oracle,repetitions); g=measure(generated,repetitions); }
            if (pair>=warmup) report.append(name).append(" pair=").append(pair-warmup).append(" bound_joint_ns=").append(g).append(" oracle_joint_ns=").append(o).append('\n');
        }
    }
    private static long measure(Runnable work,int repetitions) { long start=System.nanoTime(); for (int i=0;i<repetitions;i++) work.run(); return (System.nanoTime()-start)/repetitions; }
    private static void retain(String name,StringBuilder report) throws Exception {
        var root=Path.of(System.getProperty("synaptik.cpu.generalScan.evidenceRoot")); Files.createDirectories(root);
        Files.writeString(root.resolve(name),report); System.out.print(report);
    }
    private static void fill(DataType type,MemorySegment segment,double value) { for (long i=0;i<segment.byteSize()/type.byteWidth();i++) put(type,segment,i,value); }
    private static void put(DataType type,MemorySegment segment,long index,double value) { if (type==DataType.FLOAT64) segment.setAtIndex(ValueLayout.JAVA_DOUBLE,index,value); else segment.setAtIndex(ValueLayout.JAVA_FLOAT,index,(float)value); }
    private static double get(DataType type,MemorySegment segment,long index) { return type==DataType.FLOAT64?segment.getAtIndex(ValueLayout.JAVA_DOUBLE,index):segment.getAtIndex(ValueLayout.JAVA_FLOAT,index); }
}
