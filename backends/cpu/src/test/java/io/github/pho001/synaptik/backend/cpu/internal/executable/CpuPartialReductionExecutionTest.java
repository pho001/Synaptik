package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPartialReductionIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.MemorySegment;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGenerator;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class CpuPartialReductionExecutionTest {
    /*
     * This is intentionally an opt-in main rather than a normal test.  Its root is a sealed,
     * caller-owned experiment: no ordinary test invocation can create, append to, or interpret
     * profitability evidence.  Keeping the harness beside the execution test gives its Gp body
     * access to the actual worker-group execution rather than timing a substitute loop.
     */
    private static final String ROOT_PROPERTY = "synaptik.cpu.0008p.evidenceRoot";
    private static final long SEED = 0x0000000000080050L;
    private static final int FORKS = 5, WARMUPS = 5, PAIRS = 9;
    private static final long MIN_SIDE_NS = 25_000_000L, CALIBRATION_NS = 50_000_000L;
    private static final long LIMIT_NS = TimeUnit.MINUTES.toNanos(30);

    /** Launches exactly one sealed parent or one named child process for retained 0008P evidence. */
    public static void main(String[] args) throws Exception {
        if (args.length == 3 && args[0].equals("--fork")) {
            child(Path.of(requiredRoot()), Integer.parseInt(args[1]), Integer.parseInt(args[2]));
            return;
        }
        if (args.length != 0) throw new IllegalArgumentException("expected no arguments or --fork row fork");
        parent(Path.of(requiredRoot()));
    }

    /** Runs the one authorized immutable protocol only when its explicit root property is supplied. */
    @Test void sealedTwentyFourRowPerformanceProtocol() throws Exception {
        String root = System.getProperty(ROOT_PROPERTY);
        if (root == null || root.isBlank()) return;
        parent(Path.of(root));
    }

    private static void parent(Path root) throws Exception {
        requireInitiallyEmptyExternalRoot(root);
        long started = System.nanoTime();
        List<Row> rows = rows();
        String protocol = protocol(rows);
        Files.writeString(root.resolve("protocol.json"), protocol, StandardCharsets.UTF_8);
        Files.createDirectory(root.resolve("forks"));
        Files.writeString(root.resolve("environment.json"), environment("started", started), StandardCharsets.UTF_8);
        Files.writeString(root.resolve("classes.json"), classes(rows), StandardCharsets.UTF_8);
        Files.writeString(root.resolve("inputs.json"), inputs(rows), StandardCharsets.UTF_8);
        seal(root, rows);
        boolean interrupted = false;
        String failure = "";
        outer: for (int row = 0; row < rows.size(); row++) for (int fork = 0; fork < FORKS; fork++) {
            if (System.nanoTime() - started > LIMIT_NS) { failure = "ENVIRONMENT_LIMIT"; break outer; }
            List<String> command = new ArrayList<>();
            command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
            command.addAll(List.of("-Xms1g", "-Xmx1g", "-XX:-TieredCompilation", "-Xbatch",
                    "-D" + ROOT_PROPERTY + "=" + root, "-cp", System.getProperty("java.class.path"),
                    CpuPartialReductionExecutionTest.class.getName(), "--fork", Integer.toString(row),
                    Integer.toString(fork)));
            Process process = new ProcessBuilder(command).redirectOutput(root.resolve("forks/row-"
                    + rows.get(row).id + "-fork-" + fork + ".stdout").toFile()).redirectError(root.resolve("forks/row-"
                    + rows.get(row).id + "-fork-" + fork + ".stderr").toFile()).start();
            long remaining = LIMIT_NS - (System.nanoTime() - started);
            if (remaining <= 0 || !process.waitFor(remaining, TimeUnit.NANOSECONDS)) {
                process.destroyForcibly();
                failure = "ENVIRONMENT_LIMIT row=" + rows.get(row).id + " fork=" + fork;
                break outer;
            }
            if (process.exitValue() != 0) { failure = "FORK_FAILURE row=" + rows.get(row).id + " fork=" + fork; break outer; }
        }
        if (Thread.currentThread().isInterrupted()) interrupted = true;
        String summary = summarize(root, rows, failure.isEmpty() && !interrupted, failure.isEmpty()
                ? (interrupted ? "INTERRUPTED" : "") : failure);
        Files.writeString(root.resolve("summary.json"), summary, StandardCharsets.UTF_8);
        sums(root);
    }

    private static void child(Path root, int rowOrdinal, int fork) throws Exception {
        List<Row> rows = rows(); verifySeal(root, rows);
        if (rowOrdinal < 0 || rowOrdinal >= rows.size() || fork < 0 || fork >= FORKS) throw new IllegalArgumentException("fork ordinal");
        Row row = rows.get(rowOrdinal); Work work = new Work(row, fork);
        Path csv = root.resolve("forks/row-" + row.id + "-fork-" + fork + ".csv");
        if (Files.exists(csv)) throw new IllegalStateException("append-only fork CSV already exists");
        try (BufferedWriter writer = Files.newBufferedWriter(csv, StandardCharsets.UTF_8)) {
            writer.write("row_id,fork,comparison,pair,side,order,iterations,ns,input_sha256,output_sha256\n");
            long state = mix(work.seed);
            for (Comparison comparison : Comparison.values()) {
                for (int warmup = 0; warmup < WARMUPS; warmup++) { state = mix(state); pair(work, comparison, 1, (state & 1) == 0, null, -1); }
            }
            int iterations = 1;
            while (true) {
                long[] probe = new long[4];
                for (Implementation implementation : Implementation.values()) probe[implementation.ordinal()] = duration(work, implementation, iterations);
                if (Arrays.stream(probe).allMatch(value -> value >= CALIBRATION_NS)) break;
                if (iterations > (1 << 26)) throw new IllegalStateException("CALIBRATION_FAILURE");
                iterations = Math.multiplyExact(iterations, 2);
            }
            for (Comparison comparison : Comparison.values()) for (int pair = 0; pair < PAIRS; pair++) {
                state = mix(state); pair(work, comparison, iterations, (state & 1) == 0, writer, pair);
            }
        }
    }

    private static void pair(Work work, Comparison comparison, int iterations, boolean aFirst,
            BufferedWriter writer, int pair) throws Exception {
        Implementation a = comparison.a, b = comparison.b;
        Implementation[] order = aFirst ? new Implementation[]{a,b,b,a} : new Implementation[]{b,a,a,b};
        long[] durations = new long[4]; String outputHash = "";
        for (int side = 0; side < 4; side++) {
            durations[side] = duration(work, order[side], iterations);
            if (durations[side] < MIN_SIDE_NS && writer != null) throw new IllegalStateException("TIMING_FLOOR");
            outputHash = work.outputHash;
            if (writer != null) writer.write(String.format(Locale.ROOT, "%s,%d,%s,%d,%d,%s,%d,%d,%s,%s%n",
                    work.row.id, work.fork, comparison.name(), pair, side, aFirst ? "A-B-B-A" : "B-A-A-B",
                    iterations, durations[side], work.inputHash, outputHash));
        }
    }

    private static long duration(Work work, Implementation implementation, int iterations) {
        try {
            long start = System.nanoTime();
            for (int i = 0; i < iterations; i++) implementation.run(work);
            return System.nanoTime() - start;
        } catch (Throwable failure) {
            throw new IllegalStateException("timed implementation failed: " + implementation, failure);
        }
    }

    private enum Comparison { GP_WG(Implementation.GP, Implementation.WG, .90), GP_DP(Implementation.GP, Implementation.DP, 1.15), WG_DW(Implementation.WG, Implementation.DW, 1.15);
        final Implementation a,b; final double gate; Comparison(Implementation a, Implementation b, double gate){this.a=a;this.b=b;this.gate=gate;} }
    private enum Implementation { WG { void run(Work w) throws Throwable { w.whole(); } }, DW { void run(Work w) { w.directWhole(); } }, GP { void run(Work w) { w.partial(); } }, DP { void run(Work w) { w.directPartial(); } }; abstract void run(Work work) throws Throwable; }

    private static final class Work {
        final Row row; final int fork; final long seed; final String inputHash; String outputHash = "";
        final CpuGeneratedKernel.PartialReductionArtifact partialArtifact; final CpuGeneratedKernel wholeArtifact;
        final long[] wholeGeometry;
        final int[] ints; final long[] longs; final int[] intOutput; final long[] longOutput;
        Work(Row row, int fork) throws Exception { this.row=row;this.fork=fork;seed=mix(SEED ^ row.id.hashCode()) ^ fork; var generator=new CpuClassFileKernelGenerator(); partialArtifact=generator.generatePartialReduction(new CpuPartialReductionIr(row.kind, row.type, row.form, row.cells, row.domain, row.partials)); var whole=wholeRoute(row); wholeArtifact=generator.defineClassBytes(whole.route.specialization(),generator.generateClassBytes(whole.route.specialization(),whole.route.kernelIr())); wholeGeometry=whole.geometry.pack(new long[]{0,0}); ints=row.type==DataType.INT32?valuesInt(row,seed):null;longs=row.type==DataType.INT64?valuesLong(row,seed):null;intOutput=ints==null?null:new int[row.cells];longOutput=longs==null?null:new long[row.cells];inputHash=sha256(ints!=null?little(ints):little(longs)); }
        void whole() throws Throwable { wholeArtifact.entryPoint().invokeWithArguments(ints!=null?ints:longs,ints!=null?intOutput:longOutput,wholeGeometry,0L,(long)row.cells); outputHash=sha256(ints!=null?little(intOutput):little(longOutput)); }
        void directWhole() { if(ints!=null){for(int c=0;c<row.cells;c++){int v=row.kind==CpuPartialReductionIr.Kind.SUM?0:1;for(int i=0;i<row.domain;i++)v=row.kind==CpuPartialReductionIr.Kind.SUM?v+ints[c*row.domain+i]:v*ints[c*row.domain+i];intOutput[c]=v;}outputHash=sha256(little(intOutput));}else{for(int c=0;c<row.cells;c++){long v=row.kind==CpuPartialReductionIr.Kind.SUM?0:1;for(int i=0;i<row.domain;i++)v=row.kind==CpuPartialReductionIr.Kind.SUM?v+longs[c*row.domain+i]:v*longs[c*row.domain+i];longOutput[c]=v;}outputHash=sha256(little(longOutput));} }
        void partial() { try (var workers=new CpuWorkerGroup(4)) { if(ints!=null) CpuPartialReductionExecution.executeInt(partialArtifact,ints,0,intOutput,0,MemorySegment.ofArray(new long[row.cells*row.partials]),workers); else CpuPartialReductionExecution.executeLong(partialArtifact,longs,0,longOutput,0,MemorySegment.ofArray(new long[row.cells*row.partials]),workers); outputHash=sha256(ints!=null?little(intOutput):little(longOutput)); } }
        void directPartial() { if(ints!=null){for(int c=0;c<row.cells;c++){int v=row.kind==CpuPartialReductionIr.Kind.SUM?0:1;for(int p=0;p<row.partials;p++){int q=row.domain/row.partials,r=row.domain%row.partials,b=p*q+Math.min(p,r),e=(p+1)*q+Math.min(p+1,r),s=row.kind==CpuPartialReductionIr.Kind.SUM?0:1;for(int i=b;i<e;i++)s=row.kind==CpuPartialReductionIr.Kind.SUM?s+ints[c*row.domain+i]:s*ints[c*row.domain+i];v=row.kind==CpuPartialReductionIr.Kind.SUM?v+s:v*s;}intOutput[c]=v;}outputHash=sha256(little(intOutput));}else{for(int c=0;c<row.cells;c++){long v=row.kind==CpuPartialReductionIr.Kind.SUM?0L:1L;for(int p=0;p<row.partials;p++){int q=row.domain/row.partials,r=row.domain%row.partials,b=p*q+Math.min(p,r),e=(p+1)*q+Math.min(p+1,r);long s=row.kind==CpuPartialReductionIr.Kind.SUM?0L:1L;for(int i=b;i<e;i++)s=row.kind==CpuPartialReductionIr.Kind.SUM?s+longs[c*row.domain+i]:s*longs[c*row.domain+i];v=row.kind==CpuPartialReductionIr.Kind.SUM?v+s:v*s;}longOutput[c]=v;}outputHash=sha256(little(longOutput));} }
    }

    private record PreparedWhole(io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route, io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLowering.Geometry geometry) { }
    private static PreparedWhole wholeRoute(Row row) {
        Shape input; Shape output; OperationAttrs attributes;
        switch (row.form) {
            case FULL -> { input=Shape.of(row.domain); output=Shape.scalar(); attributes=NoOperationAttrs.INSTANCE; }
            case SINGLE_AXIS -> { input=Shape.of(64,8192); output=Shape.of(64); attributes=new AxisReductionAttrs(1,false); }
            case MULTI_AXIS -> { input=Shape.of(4,16,2048); output=Shape.of(4); attributes=new MultiAxisReductionAttrs(List.of(1,2),false); }
            default -> throw new AssertionError(row.form);
        }
        var base=CpuAggregateLoweringTest.context(row.kind==CpuPartialReductionIr.Kind.SUM ? AggregateReductionKind.SUM : AggregateReductionKind.PROD,row.type,input,attributes,output);
        PrepareContext<CpuPartitionAnalysisInputs> context=new PrepareContext<>(base.partitionDag(),base.values(),base.memoryRequirements(),base.constants(),new CpuPartitionAnalysisInputs(false,List.of(row.type==DataType.INT32?CarrierAccess.INT_ARRAY:CarrierAccess.LONG_ARRAY,row.type==DataType.INT32?CarrierAccess.INT_ARRAY:CarrierAccess.LONG_ARRAY)));
        var plan=new CpuPartitionPreparer().analyze(context).plan();
        return new PreparedWhole(plan.units().getFirst().portablePlan(),plan.aggregateGeometry().orElseThrow());
    }

    private record Row(String id, DataType type, CpuPartialReductionIr.Kind kind, CpuAggregateIr.Form form, int cells, int domain, int partials) { }
    private static List<Row> rows(){List<Row> r=new ArrayList<>();for(DataType t:List.of(DataType.INT32,DataType.INT64))for(CpuPartialReductionIr.Kind k:CpuPartialReductionIr.Kind.values())for(Object[] f:new Object[][]{{CpuAggregateIr.Form.FULL,1,524288},{CpuAggregateIr.Form.SINGLE_AXIS,64,8192},{CpuAggregateIr.Form.MULTI_AXIS,4,32768}})for(int p:new int[]{2,4})r.add(new Row(t+"-"+k+"-"+f[0]+"-P"+p,t,k,(CpuAggregateIr.Form)f[0],(int)f[1],(int)f[2],p));return List.copyOf(r);}
    private static int[] valuesInt(Row r,long seed){int[] a=new int[r.cells*r.domain];long x=seed;for(int i=0;i<a.length;i++){x=mix(x);a[i]=(int)Math.floorMod(x,7)-3;}return a;} private static long[] valuesLong(Row r,long seed){long[] a=new long[r.cells*r.domain];long x=seed;for(int i=0;i<a.length;i++){x=mix(x);a[i]=Math.floorMod(x,7)-3;}return a;}
    private static String protocol(List<Row> rows){return "{\"schema\":\"synaptik.cpu.partial-reduction-performance.v1\",\"rows\":"+jsonRows(rows)+",\"comparisons\":[\"Gp/Wg<=0.90\",\"Gp/Dp<=1.15\",\"Wg/Dw<=1.15\"],\"forks\":5,\"warmup_pairs\":5,\"measured_pairs\":9,\"flags\":[\"-Xms1g\",\"-Xmx1g\",\"-XX:-TieredCompilation\",\"-Xbatch\"],\"seed\":\"0x0000000000080050\",\"timing_floor_ns\":25000000,\"calibration_ns\":50000000}";}
    private static String classes(List<Row> rows)throws Exception{return "{\"harness_sha256\":\""+sha256(Files.readAllBytes(harnessSourcePath()))+"\",\"descriptors\":[\"([IIILjava/lang/foreign/MemorySegment;J)V\",\"([JIILjava/lang/foreign/MemorySegment;J)V\",\"(Ljava/lang/foreign/MemorySegment;II[II)V\",\"(Ljava/lang/foreign/MemorySegment;II[JI)V\"],\"rows\":"+rows.size()+"}";}
    private static Path harnessSourcePath() throws Exception {
        Path classes = Path.of(CpuPartialReductionExecutionTest.class.getProtectionDomain()
                .getCodeSource().getLocation().toURI()).toAbsolutePath().normalize();
        for (Path candidate = classes; candidate != null; candidate = candidate.getParent()) {
            Path source = candidate.resolve("backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPartialReductionExecutionTest.java");
            if (Files.isRegularFile(source)) return source;
        }
        throw new IllegalStateException("cannot locate 0008P harness source from its code source");
    }
    private static String inputs(List<Row> rows){StringBuilder b=new StringBuilder("{\"mapping\":\"SplitMix64 floorMod(next,7)-3\",\"rows\":[");for(int i=0;i<rows.size();i++){Row r=rows.get(i);long s=mix(SEED^r.id.hashCode());b.append(i==0?"":",").append("{\"id\":\"").append(r.id).append("\",\"seed\":\"").append(Long.toUnsignedString(s)).append("\",\"input_sha256\":\"").append(sha256(r.type==DataType.INT32?little(valuesInt(r,s)):little(valuesLong(r,s)))).append("\"}");}return b.append("]}").toString();}
    private static String environment(String phase,long started){return "{\"phase\":\""+phase+"\",\"wall_clock\":\""+Instant.now()+"\",\"os\":\""+esc(System.getProperty("os.name"))+" "+esc(System.getProperty("os.version"))+"\",\"architecture\":\""+esc(System.getProperty("os.arch"))+"\",\"processors\":"+Runtime.getRuntime().availableProcessors()+",\"jdk\":\""+esc(System.getProperty("java.vendor")+" "+System.getProperty("java.version"))+"\",\"jvm_flags\":\""+esc(String.join(" ",ManagementFactory.getRuntimeMXBean().getInputArguments()))+"\",\"affinity\":\"unavailable\",\"governor\":\"unavailable\"}";}
    private static void seal(Path root,List<Row> rows)throws Exception{String s=sha256(Files.readAllBytes(root.resolve("protocol.json")))+sha256(Files.readAllBytes(root.resolve("classes.json")))+sha256(Files.readAllBytes(root.resolve("inputs.json")));Files.writeString(root.resolve(".seal"),s);}
    private static void verifySeal(Path root,List<Row> rows)throws Exception{if(!Files.isRegularFile(root.resolve(".seal"))||!Files.readString(root.resolve(".seal")).equals(sha256(Files.readAllBytes(root.resolve("protocol.json")))+sha256(Files.readAllBytes(root.resolve("classes.json")))+sha256(Files.readAllBytes(root.resolve("inputs.json")))))throw new IllegalStateException("PROTOCOL_SEAL_FAILURE");if(!Files.readString(root.resolve("protocol.json")).contains(jsonRows(rows)))throw new IllegalStateException("row inventory changed");}
    private static String summarize(Path root,List<Row> rows,boolean complete,String failure)throws Exception{StringBuilder out=new StringBuilder("{\"complete\":"+complete+",\"failure\":\""+esc(failure)+"\",\"expected_pairs\":3240,\"expected_timings\":12960,\"expected_fork_medians\":360,\"expected_aggregates\":72,\"rows\":[");boolean all=true;for(int ri=0;ri<rows.size();ri++){Row row=rows.get(ri);StringBuilder details=new StringBuilder();boolean pass=complete;for(Comparison c:Comparison.values()){double[] fm=new double[FORKS];for(int f=0;f<FORKS;f++){Path p=root.resolve("forks/row-"+row.id+"-fork-"+f+".csv");if(!Files.isRegularFile(p)){pass=false;continue;}List<String> lines=Files.readAllLines(p);double[] ratios=new double[PAIRS];int n=0;for(int i=1;i<lines.size();i+=4){String[] a=lines.get(i).split(","),b=lines.get(i+1).split(","),d=lines.get(i+2).split(","),e=lines.get(i+3).split(",");if(!a[2].equals(c.name()))continue;long an=Long.parseLong(a[7]),bn=Long.parseLong(b[7]),dn=Long.parseLong(d[7]),en=Long.parseLong(e[7]);boolean af=a[5].equals("A-B-B-A");ratios[n++]=(double)(af?an+en:bn+dn)/(af?bn+dn:an+en);}if(n!=PAIRS){pass=false;continue;}Arrays.sort(ratios);fm[f]=ratios[PAIRS/2];for(double x:ratios)if(x>c.gate)pass=false;if(fm[f]>c.gate)pass=false;}Arrays.sort(fm);if(fm[FORKS/2]>c.gate)pass=false;details.append("\\\"").append(c.name()).append("_median\\\":").append(fm[FORKS/2]).append(',');}all&=pass;out.append(ri==0?"":",").append("{\"id\":\"").append(row.id).append("\",\"decision\":\"").append(pass?"PASS":"KEEP_WHOLE_CELL").append("\",").append(details.substring(0,details.length()-1)).append('}');}return out.append("],\"admission\":\"").append(all?"ADMIT_ALL_24":"KEEP_WHOLE_CELL").append("\"}").toString();}
    private static void sums(Path root)throws Exception{List<Path> files;try(var s=Files.walk(root)){files=s.filter(Files::isRegularFile).filter(p->!p.getFileName().toString().equals("SHA256SUMS")).sorted().toList();}StringBuilder b=new StringBuilder();for(Path p:files)b.append(sha256(Files.readAllBytes(p))).append("  ").append(root.relativize(p)).append('\n');Files.writeString(root.resolve("SHA256SUMS"),b);}
    private static void requireInitiallyEmptyExternalRoot(Path root)throws Exception{Path checkout=Path.of("").toAbsolutePath().normalize();if(root.toAbsolutePath().normalize().startsWith(checkout)||!Files.isDirectory(root))throw new IllegalArgumentException("evidence root must be initially empty and outside checkout");try(var entries=Files.list(root)){if(entries.findAny().isPresent())throw new IllegalArgumentException("evidence root must be initially empty and outside checkout");}}
    private static String requiredRoot(){String r=System.getProperty(ROOT_PROPERTY);if(r==null||r.isBlank())throw new IllegalArgumentException("missing "+ROOT_PROPERTY);return r;}
    private static long mix(long z){z+=(long)0x9E3779B97F4A7C15L;z=(z^(z>>>30))*0xBF58476D1CE4E5B9L;z=(z^(z>>>27))*0x94D049BB133111EBL;return z^(z>>>31);} private static byte[] little(int[] a){ByteBuffer b=ByteBuffer.allocate(a.length*4).order(ByteOrder.LITTLE_ENDIAN);for(int x:a)b.putInt(x);return b.array();}private static byte[] little(long[] a){ByteBuffer b=ByteBuffer.allocate(a.length*8).order(ByteOrder.LITTLE_ENDIAN);for(long x:a)b.putLong(x);return b.array();}private static String sha256(byte[] b){try{return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(b));}catch(Exception e){throw new IllegalStateException(e);}}private static String jsonRows(List<Row> rows){return rows.stream().map(r->"\\\""+r.id+"\\\"").collect(java.util.stream.Collectors.joining(",","[","]"));}private static String esc(String s){return s.replace("\\","\\\\").replace("\"","\\\"");}
    @Test void combinesFixedPartialStatesInOrdinalOrderWithModularIntArithmetic() {
        var ir = new CpuPartialReductionIr(CpuPartialReductionIr.Kind.SUM, DataType.INT32,
                CpuAggregateIr.Form.SINGLE_AXIS, 2, 5, 2);
        int[] input = {Integer.MAX_VALUE, 1, -2, 3, 4, 7, 8, 9, 10, 11};
        int[] output = {-1, -1};
        try (var workers = new CpuWorkerGroup(2)) {
            CpuPartialReductionExecution.executeInt(new CpuClassFileKernelGenerator()
                    .generatePartialReduction(ir), input, 0, output, 0,
                    MemorySegment.ofArray(new long[4]), workers);
        }
        assertArrayEquals(new int[]{-2_147_483_643, 45}, output);
    }

    @Test void combinesLongProductWithFixedFourWayRanges() {
        var ir = new CpuPartialReductionIr(CpuPartialReductionIr.Kind.PROD, DataType.INT64,
                CpuAggregateIr.Form.FULL, 1, 5, 4);
        long[] output = {-1L};
        try (var workers = new CpuWorkerGroup(4)) {
            CpuPartialReductionExecution.executeLong(new CpuClassFileKernelGenerator()
                    .generatePartialReduction(ir), new long[]{Long.MAX_VALUE, 2L, -1L, 3L, 5L},
                    0, output, 0, MemorySegment.ofArray(new long[4]), workers);
        }
        assertArrayEquals(new long[]{30L}, output);
    }
}
