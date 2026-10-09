package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Opt-in five-fork generated-versus-independent-clean-Java performance gate for CPU 0009C1.
 *
 * <p>The eight cases are mechanically selected representatives of the 48-row CONCAT/STACK
 * inventory: each operation has one FLOAT32 representative for each distinct selected emitter
 * path and ordered carrier shape. The source-derived inventory assertion prevents a benchmark of
 * one operation or carrier shape from being projected onto another. Timed actions invoke the
 * real generated entry and the independently javac-compiled clean-oracle entry through matching
 * typed method-handle adapters; carrier construction and equality checks occur outside timing.</p>
 */
class CpuConcatStackPerformanceTest {
    private static final String ENABLE = "synaptik.cpu.concatStack.performance";
    private static final String ROOT = "synaptik.cpu.concatStack.performanceEvidenceRoot";
    private static final int FORKS = 5, WARMUPS = 5, SAMPLES = 7;
    /* The source matrix has five- and four-element kernels.  Keep the exact source mapping,
       but time the same prepared operation on a large geometry so each entry invocation exposes
       its selected copy loop rather than its method-handle, guard, and timer costs. */
    private static final int SCALE = 8_192, COMPILE_STABILIZATION_INVOCATIONS = 12_000;
    private static final long MINIMUM_NANOS = 25_000_000L;
    private static final double LIMIT = 1.15d;
    private static volatile long sink;

    enum Case {
        CONCAT_DENSE_HEAP("CONCAT", "heap-contiguous-scalar", "DENSE_HEAP_ARRAY_INT"),
        CONCAT_BOUNDED_HEAP("CONCAT", "heap-general-materialization-candidate", "BOUNDED_INT_TARGET"),
        CONCAT_BOUNDED_SEGMENT("CONCAT", "segment-contiguous-vector", "BOUNDED_INT_TARGET"),
        CONCAT_BOUNDED_MIXED("CONCAT", "mixed-general-parallel-vector", "BOUNDED_INT_TARGET"),
        STACK_DENSE_HEAP("STACK", "heap-contiguous-scalar", "DENSE_HEAP_ARRAY_INT"),
        STACK_GENERAL_HEAP("STACK", "heap-general-materialization-candidate", "GENERAL_LONG"),
        STACK_GENERAL_SEGMENT("STACK", "segment-contiguous-vector", "GENERAL_LONG"),
        STACK_GENERAL_MIXED("STACK", "mixed-general-parallel-vector", "GENERAL_LONG");
        final String form, request, addressing;
        Case(String form, String request, String addressing) { this.form=form; this.request=request; this.addressing=addressing; }
    }

    public static void main(String[] args) throws Throwable {
        if (args.length == 2 && args[0].equals("--fork")) runFork(root(), Integer.parseInt(args[1]));
        else runParent(root());
    }

    @Test void retainedFiveForkEvidence() throws Throwable {
        Assumptions.assumeTrue(Boolean.getBoolean(ENABLE)); runParent(root());
    }

    @Test void sourceDerivedFortyEightRowMappingSelectsEveryChangedPathAndCarrierShape() throws Exception {
        List<CpuOrdinaryNonPointwiseGeneratedMatrixTest.MovementOrFoldCandidate> rows = compositionRows();
        assertEquals(48, rows.size());
        for (String form : List.of("CONCAT", "STACK")) {
            List<CpuOrdinaryNonPointwiseGeneratedMatrixTest.MovementOrFoldCandidate> formRows = rows.stream()
                    .filter(row -> row.operationForm().equals(form)).toList();
            assertEquals(24, formRows.size(), form);
            for (Case c : Case.values()) if (c.form.equals(form)) {
                List<?> matches = formRows.stream().filter(row -> row.ownerId().contains("/FLOAT32/")
                        && row.ownerId().endsWith('/' + c.request)).toList();
                assertEquals(1, matches.size(), c + " exact FLOAT32 source representative");
                var plan = new CpuPartitionPreparer().analyze(((CpuOrdinaryNonPointwiseGeneratedMatrixTest.MovementOrFoldCandidate) matches.getFirst()).context())
                        .plan();
                var route = plan.units().getFirst().portablePlan();
                String actual = actualPath(c, plan, route, ((CpuOrdinaryNonPointwiseGeneratedMatrixTest.MovementOrFoldCandidate) matches.getFirst()));
                assertEquals(c.addressing, actual, c + " actual selected emitter path");
            }
        }
    }

    private static Path root() {
        String value = System.getProperty(ROOT);
        if (value == null || value.isBlank()) throw new IllegalArgumentException(ROOT + " is required");
        return Path.of(value).toAbsolutePath();
    }

    private static void runParent(Path root) throws Throwable {
        Path evidence = root.resolve("cpu-0009c1-concat-stack-performance");
        if (Files.exists(evidence)) throw new IllegalStateException("fresh evidence root required: " + evidence);
        Files.createDirectories(evidence);
        mapping(evidence);
        Files.writeString(evidence.resolve("protocol.txt"), "task=0009C1\nforks=5\nwarmups=5\nmeasured_rounds=7\nminimum_sample_ns=25000000\nfixed_heap=-Xms1g,-Xmx1g\ncompiler=-XX:-TieredCompilation,-Xbatch\norder=seeded-alternating-generated-direct\nretry=false\ndiscard=false\nthreshold=1.15\ncases=" + Arrays.toString(Case.values()) + "\n", StandardCharsets.UTF_8);
        Files.writeString(evidence.resolve("environment.txt"), "java.version=" + System.getProperty("java.version")
                + "\nos.arch=" + System.getProperty("os.arch") + "\nos.name=" + System.getProperty("os.name") + "\n", StandardCharsets.UTF_8);
        Files.copy(source(), evidence.resolve("CpuConcatStackPerformanceTest.java"), StandardCopyOption.REPLACE_EXISTING);
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        StringBuilder commands = new StringBuilder();
        for (int fork=0; fork<FORKS; fork++) {
            List<String> command = List.of(java, "-Xms1g", "-Xmx1g", "-XX:-TieredCompilation", "-Xbatch", "--add-modules", "jdk.incubator.vector", "-cp", System.getProperty("java.class.path"), "-D" + ROOT + "=" + root, CpuConcatStackPerformanceTest.class.getName(), "--fork", Integer.toString(fork));
            commands.append(String.join(" ", command)).append('\n');
            Process process = new ProcessBuilder(command).redirectOutput(evidence.resolve("fork-"+fork+".stdout").toFile()).redirectError(evidence.resolve("fork-"+fork+".stderr").toFile()).start();
            assertEquals(0, process.waitFor(), "fork " + fork);
            validateFork(evidence.resolve("raw-fork-" + fork + ".csv"), fork);
        }
        Files.writeString(evidence.resolve("commands.txt"), commands, StandardCharsets.UTF_8);
        aggregate(evidence); seal(evidence);
    }

    private static void runFork(Path root, int fork) throws Throwable {
        if (fork < 0 || fork >= FORKS) throw new IllegalArgumentException("fork");
        Path evidence = root.resolve("cpu-0009c1-concat-stack-performance");
        Random random = new Random(0x0009_c1aL + fork);
        StringBuilder raw = new StringBuilder("case,iterations,median_ratio,checksum\n");
        for (Case c : Case.values()) {
            Prepared p = prepare(c); verify(p); stabilize(p); int iterations = calibrate(p, random);
            for (int warmup=0; warmup<WARMUPS; warmup++) measure(p, iterations, random);
            double[] ratios = new double[SAMPLES];
            StringBuilder samples = new StringBuilder("case,sample,iterations,generated_ns,direct_ns,ratio\n");
            for (int sample=0; sample<SAMPLES; sample++) {
                Measurement m=measure(p, iterations, random); ratios[sample]=m.ratio();
                samples.append(c).append(',').append(sample).append(',').append(iterations).append(',').append(m.generated).append(',').append(m.direct).append(',').append(m.ratio()).append('\n');
            }
            Files.writeString(evidence.resolve("samples-fork-"+fork+"-"+c+".csv"), samples, StandardCharsets.UTF_8);
            for (int sample = 0; sample < SAMPLES; sample++) {
                String[] fields = samples.toString().lines().skip(sample + 1).findFirst().orElseThrow().split(",");
                assertTrue(Long.parseLong(fields[3]) >= MINIMUM_NANOS && Long.parseLong(fields[4]) >= MINIMUM_NANOS,
                        c + " minimum sample duration");
                assertTrue(Double.isFinite(ratios[sample]) && ratios[sample] > 0d, c + " finite ratio");
            }
            Arrays.sort(ratios); double median=ratios[SAMPLES/2]; assertTrue(median <= LIMIT, c + " fork " + fork + "=" + median);
            raw.append(c).append(',').append(iterations).append(',').append(median).append(',').append(sink).append('\n');
        }
        Files.writeString(evidence.resolve("raw-fork-"+fork+".csv"), raw, StandardCharsets.UTF_8);
    }

    private static int calibrate(Prepared p, Random random) throws Throwable {
        int iterations=1;
        while (true) { Measurement m=measure(p, iterations, random); if (m.generated>=MINIMUM_NANOS && m.direct>=MINIMUM_NANOS) return iterations; iterations=Math.multiplyExact(iterations,2); }
    }
    private static void stabilize(Prepared p) throws Throwable {
        for (int invocation=0; invocation<COMPILE_STABILIZATION_INVOCATIONS; invocation++) {
            p.generated.run(); p.direct.run();
        }
    }
    private static Measurement measure(Prepared p, int iterations, Random random) throws Throwable {
        long generated, direct;
        if(random.nextBoolean()) {
            generated=timeBatch(p.generated, iterations); sink ^= checksum(p.generatedArrays);
            direct=timeBatch(p.direct, iterations); sink ^= checksum(p.directArrays);
        } else {
            direct=timeBatch(p.direct, iterations); sink ^= checksum(p.directArrays);
            generated=timeBatch(p.generated, iterations); sink ^= checksum(p.generatedArrays);
        }
        return new Measurement(generated,direct);
    }
    private static long timeBatch(Timed action, int iterations) throws Throwable {
        long started = System.nanoTime();
        for (int invocation = 0; invocation < iterations; invocation++) action.run();
        return System.nanoTime() - started;
    }

    private static Prepared prepare(Case c) throws Throwable {
        var sourceCandidate=compositionRows().stream().filter(row -> row.operationForm().equals(c.form)
                && row.ownerId().contains("/FLOAT32/") && row.ownerId().endsWith('/' + c.request)).findFirst().orElseThrow();
        var candidate = new CpuOrdinaryNonPointwiseGeneratedMatrixTest.MovementOrFoldCandidate(
                sourceCandidate.ownerId(), sourceCandidate.operationForm(), scaledContext(sourceCandidate.context(), c));
        var plan=new CpuPartitionPreparer().analyze(candidate.context()).plan(); var route=plan.units().getFirst().portablePlan();
        assertEquals(c.addressing, actualPath(c, plan, route, candidate), c + " scaled selected emitter path");
        var movement=(CpuDataMovementIr) route.portableKernelIr(); var output=candidate.context().nodes().getFirst().outputs().getFirst();
        DataType type=candidate.context().values().stream().filter(value -> value.id().equals(output)).findFirst().orElseThrow().descriptor().dataType();
        String family = c == Case.STACK_DENSE_HEAP ? "dense-stack"
                : c.addressing.equals("BOUNDED_INT_TARGET") ? "bounded-concat" : c.form.toLowerCase();
        var row=new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row(candidate.ownerId(), family, route.specialization().entryType().descriptorString(), type.name(), false, candidate.context().values().getLast().descriptor().shape().rank(), 0L, movement.plan().occurrenceToBoundary());
        var clean=CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(List.of(row));
        var generator=new CpuClassFileKernelGenerator(); MethodHandle generated=generator.defineClassBytes(route.specialization(), generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
        MethodHandle direct=CpuAffineMovementIndexingScatterRandomCleanJavaOracle.entry(clean, candidate.ownerId(), route.specialization().entryType().descriptorString());
        CarrierSet generatedCarriers=carriers(candidate, plan, route.specialization()); CarrierSet directCarriers=carriers(candidate, plan, route.specialization());
        List<Object> generatedArgs=generatedCarriers.arguments; List<Object> directArgs=directCarriers.arguments;
        long[] geometry=plan.movementGeometry().orElseThrow().pack(new long[plan.boundaryValues().size()],0,plan.elementCount()); generatedArgs.add(geometry); generatedArgs.add(0L); generatedArgs.add(plan.elementCount()); directArgs.add(geometry.clone()); directArgs.add(0L); directArgs.add(plan.elementCount());
        return new Prepared(c, generatedArgs, directArgs, generatedCarriers.arrays, directCarriers.arrays,
                action(generated, generatedArgs.toArray()), action(direct, directArgs.toArray()));
    }
    private static Timed action(MethodHandle handle, Object[] args) {
        MethodHandle spread=handle.asSpreader(Object[].class,args.length).asType(MethodType.methodType(void.class,Object[].class));
        return () -> invokeVoid(spread, args);
    }
    private static void invokeVoid(MethodHandle handle, Object[] arguments) throws Throwable { handle.invokeExact(arguments); }
    private static CarrierSet carriers(CpuOrdinaryNonPointwiseGeneratedMatrixTest.MovementOrFoldCandidate candidate, CpuPartitionPreparationPlan plan, CpuKernelSpecialization specialization) {
        var arguments=new ArrayList<Object>(); var arrays=new ArrayList<Object>();
        for(int i=0;i<plan.boundaryValues().size();i++) { var id=plan.boundaryValues().get(i); TensorDescriptor d=candidate.context().values().stream().filter(value -> value.id().equals(id)).findFirst().orElseThrow().descriptor(); Object array=values(d.dataType(), Math.toIntExact(maxAddress(d)+1)); arrays.add(array); arguments.add(specialization.carrierPattern().get(i)==CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT ? segment(array) : array); }
        return new CarrierSet(arguments, arrays);
    }
    private static void verify(Prepared p) {
        try { p.generated.run(); p.direct.run(); assertEquals(checksum(p.generatedArrays), checksum(p.directArrays), p.c+" scaled semantic checksum"); }
        catch(Throwable failure) { throw new AssertionError(p.c+" semantic check",failure); }
    }
    private static long checksum(List<Object> arrays) { long result=0x9e3779b97f4a7c15L; for(Object array:arrays) for(float value:(float[])array) result=Long.rotateLeft(result,7)^Float.floatToRawIntBits(value); return result; }
    private static long maxAddress(TensorDescriptor d) { long total=d.layout().orElseThrow().storageOffset(); long[] shape=d.shape().toLongArray(), strides=d.layout().orElseThrow().strides(); for(int i=0;i<shape.length;i++) total+=Math.max(0,shape[i]-1)*strides[i]; return total; }
    private static Object values(DataType type, int n) { if(type!=DataType.FLOAT32) throw new AssertionError(type); float[] values=new float[n]; for(int i=0;i<n;i++) values[i]=Float.intBitsToFloat(0x3f000000+i); return values; }
    private static MemorySegment segment(Object array) { return MemorySegment.ofArray((float[])array); }
    private static List<CpuOrdinaryNonPointwiseGeneratedMatrixTest.MovementOrFoldCandidate> compositionRows() { return CpuOrdinaryNonPointwiseGeneratedMatrixTest.movementOrFoldCandidates().stream().filter(row -> row.operationForm().equals("CONCAT")||row.operationForm().equals("STACK")).toList(); }

    private static String actualPath(Case c, CpuPartitionPreparationPlan plan,
            io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route,
            CpuOrdinaryNonPointwiseGeneratedMatrixTest.MovementOrFoldCandidate candidate) {
        if (route.specialization().loopAddressing(route.kernelIr()) == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT) return "DENSE_HEAP_ARRAY_INT";
        if (!c.form.equals("CONCAT")) return "GENERAL_LONG";
        long[] geometry = plan.movementGeometry().orElseThrow().pack(new long[plan.boundaryValues().size()], 0L,
                plan.elementCount());
        for (long value : geometry) if (value < 0L || value > Integer.MAX_VALUE) return "GENERAL_LONG";
        return "BOUNDED_INT_TARGET";
    }
    private static void mapping(Path evidence) throws Exception { StringBuilder out=new StringBuilder("case,form,request,source_emitter_path,scaled_emitter_path,carrier_pattern,source_dimensions,scaled_dimensions,represented_rows\n"); for(Case c:Case.values()) { var source=compositionRows().stream().filter(row->row.operationForm().equals(c.form)&&row.ownerId().endsWith('/'+c.request)&&row.ownerId().contains("/FLOAT32/")).findFirst().orElseThrow(); var sourcePlan=new CpuPartitionPreparer().analyze(source.context()).plan(); var sourceRoute=sourcePlan.units().getFirst().portablePlan(); String sourceActual=actualPath(c,sourcePlan,sourceRoute,source); assertEquals(c.addressing,sourceActual,c+" exact source path"); var scaled=new CpuOrdinaryNonPointwiseGeneratedMatrixTest.MovementOrFoldCandidate(source.ownerId(),source.operationForm(),scaledContext(source.context(),c)); var scaledPlan=new CpuPartitionPreparer().analyze(scaled.context()).plan(); var scaledRoute=scaledPlan.units().getFirst().portablePlan(); String scaledActual=actualPath(c,scaledPlan,scaledRoute,scaled); assertEquals(c.addressing,scaledActual,c+" exact scaled path"); long represented=compositionRows().stream().filter(row->row.operationForm().equals(c.form)&&row.ownerId().endsWith('/'+c.request)).count(); assertEquals(6,represented,c+" six type-width rows"); out.append(c).append(',').append(c.form).append(',').append(c.request).append(',').append(sourceActual).append(',').append(scaledActual).append(',').append(scaledRoute.specialization().carrierPattern()).append(',').append(dimensions(source.context())).append(',').append(dimensions(scaled.context())).append(',').append(represented).append('\n'); } Files.writeString(evidence.resolve("case-mapping.csv"),out,StandardCharsets.UTF_8); }
    private static PrepareContext<io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs> scaledContext(PrepareContext<io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs> source, Case c) {
        var values = new ArrayList<GraphValue>(); var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < source.values().size(); index++) { var old = source.values().get(index); TensorDescriptor descriptor = scaledDescriptor(old.descriptor(), c); values.add(new GraphValue(old.id(), descriptor)); var requirement = source.memoryRequirements().get(index); memory.add(new LogicalMemoryRequirement(requirement.valueId(), descriptor, requirement.producerPartition(), requirement.consumerPartitions(), requirement.graphOutput())); }
        return new PrepareContext<>(source.partition(), source.nodes(), values, memory, source.constants(), source.backendInputs());
    }
    private static TensorDescriptor scaledDescriptor(TensorDescriptor original, Case c) { long[] dimensions=original.shape().toLongArray(); dimensions[0]=Math.multiplyExact(dimensions[0], SCALE); Shape shape=Shape.of(dimensions); LayoutDescriptor layout=original.layout().orElseThrow(); return new TensorDescriptor(original.dataType(),shape,Optional.of(LayoutDescriptor.of(shape,layout.strides(),layout.storageOffset(),true)),original.requiresGrad()); }
    private static String dimensions(PrepareContext<?> context) { return context.values().stream().map(value -> Arrays.toString(value.descriptor().shape().toLongArray()).replace(", ", "x").replace("[", "").replace("]", "")).reduce((left,right)->left+"|"+right).orElseThrow(); }
    private static void validateFork(Path file,int fork) throws Exception { List<String> lines=Files.readAllLines(file); assertEquals(Case.values().length+1,lines.size(),"fork "+fork+" complete rows"); for(int i=1;i<lines.size();i++){String[] f=lines.get(i).split(","); assertTrue(Double.parseDouble(f[2])<=LIMIT,"fork "+fork+" "+f[0]);} }
    private static void aggregate(Path evidence) throws Exception { StringBuilder out=new StringBuilder("case,fork0,fork1,fork2,fork3,fork4,median\n"); List<Double> all=new ArrayList<>(); for(Case c:Case.values()){double[] ratios=new double[FORKS];for(int fork=0;fork<FORKS;fork++){String[] found=Files.readAllLines(evidence.resolve("raw-fork-"+fork+".csv")).stream().filter(line->line.startsWith(c+",")).findFirst().orElseThrow().split(",");ratios[fork]=Double.parseDouble(found[2]);all.add(ratios[fork]);}double[] ordered=ratios.clone();Arrays.sort(ordered);assertTrue(ordered[2]<=LIMIT,c+" five-fork median");out.append(c);for(double ratio:ratios)out.append(',').append(ratio);out.append(',').append(ordered[2]).append('\n');} all.sort(Comparator.naturalOrder()); double aggregate=all.get(all.size()/2);assertTrue(aggregate<=LIMIT,"aggregate median of fork medians");out.append("aggregate_median,,,,,,").append(aggregate).append('\n');Files.writeString(evidence.resolve("aggregate.csv"),out,StandardCharsets.UTF_8); }
    private static void seal(Path evidence) throws Exception { List<Path> files=Files.list(evidence).filter(Files::isRegularFile).filter(p->!p.getFileName().toString().equals("SHA256SUMS")&&!p.getFileName().toString().equals("manifest.txt")).sorted().toList();StringBuilder sums=new StringBuilder(),manifest=new StringBuilder("task=0009C1\nsealed_files="+files.size()+"\n");for(Path file:files){String hash=HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file)));sums.append(hash).append("  ").append(file.getFileName()).append('\n');manifest.append(file.getFileName()).append('=').append(hash).append('\n');}Files.writeString(evidence.resolve("SHA256SUMS"),sums,StandardCharsets.UTF_8);Files.writeString(evidence.resolve("manifest.txt"),manifest,StandardCharsets.UTF_8); }
    private static Path source(){return Path.of(System.getProperty("user.dir"), "src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConcatStackPerformanceTest.java");}
    private record Measurement(long generated,long direct){double ratio(){return (double)generated/direct;}}
    @FunctionalInterface private interface Timed {void run() throws Throwable;}
    private record CarrierSet(List<Object> arguments,List<Object> arrays){}
    private record Prepared(Case c,List<Object> generatedArgs,List<Object> directArgs,List<Object> generatedArrays,List<Object> directArrays,Timed generated,Timed direct){}
}
