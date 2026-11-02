package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.*;
import io.github.pho001.synaptik.backend.cpu.internal.executable.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import java.lang.classfile.*;
import java.lang.classfile.attribute.CodeAttribute;
import java.lang.classfile.instruction.*;
import java.lang.foreign.*;
import java.lang.invoke.*;
import java.lang.management.ManagementFactory;
import java.lang.reflect.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;

/**
 * Test-only, explicitly opt-in CPU0075 A/B/C handoff. No old production emitter or fabricated
 * specialization is used: A is the literal authenticated old hidden class, B the current class,
 * and C the unchanged independent Java oracle. Preflight executes each equation once, untimed.
 */
public final class CpuRowScalarReuseComparison {
    private static final String[] OLD_SHA = {
        "61eb357c1d6409879c28997695659793a5782c3eef8b1300cbc1f65e8576be24",
        "e70659e6c52307ea94f93d986ccb43546366b9ede98e491ae980d59350ab3a77",
        "486a51c6a415bf5a48b084dc8b5ffad030a070c1851e7aa18d1ef5ed1d0072d8",
        "486a51c6a415bf5a48b084dc8b5ffad030a070c1851e7aa18d1ef5ed1d0072d8"
    };
    private static final String ORACLE_SHA = "84cab8e2f48c57e7d67d4339cba0807d396a37f7fe6c836d7810009f296fc148";
    private static volatile long sink;

    private CpuRowScalarReuseComparison() { }

    /**
     * Measures only after an explicit MAIN release, with no calibration, fork or retry.
     * The supplied external launcher imposes the sole JVM's 90-second wall-clock watchdog.
     * @param args authentic baseline directory and fresh existing evidence directory
     * @throws Exception if release, literal provenance, ABI, equality or persistence fails
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 2 || !"MAIN_RELEASED".equals(System.getenv("SYNAPTIK_CPU0075_MEASURE")))
            throw new IllegalArgumentException("MAIN_RELEASED and baseline/evidence arguments required");
        run(Path.of(args[0]),Path.of(args[1]),true);
    }

    /**
     * Runs exactly the four approved cases without any timing/warmup window.
     * @param baseline authentic CPU0075 baseline root
     * @param root fresh private evidence directory
     * @throws Exception if literal provenance, ABI, bytecode or equality validation fails
     */
    static void preflight(Path baseline,Path root) throws Exception { run(baseline,root,false); }

    private static void run(Path baseline,Path root,boolean measure) throws Exception {
        Files.createDirectories(root);
        verifyOracle(baseline);
        captureClasspath(root);
        Files.writeString(root.resolve("protocol.txt"),
                "CPU0075 only: ordinary/actual Scan reverse, FLOAT32, B32/B256, scalar, F3/H2, Scan T4.\n"
                +"A=literal authentic old CF; B=current CF; C=unchanged Java oracle.\n"
                +"A/B: identical Runnable -> invokeExact -> MemorySegment[] spreader -> exact original ABI.\n"
                +"A/B use fresh distinct hidden-class instances, not the setup fixture's already executed candidate instance.\n"
                +"C: existing Runnable -> specialized private Java instance method; calling overhead is not subtracted.\n"
                +"Cold binder/preflight and all snapshots/checks outside windows; direct A/B entry excludes per-call Runtime admission/lease overhead.\n"
                +"Original checked segments/interior slices, same saved inputs and geometry. Distinct A/B writable outputs/scratch; C owns existing independent outputs.\n"
                +"Entries initialize/reset complete logical outputs each call, including Scan carries/padding. No growing accumulation across calls.\n"
                +"Ordinary measures selected region, not separate exact bias SUM. Java stages ten W/U cells per call, uses scalar seed/derivative instead of expanded seed/one tensor.\n"
                +"Scan Java keeps twelve carry cells in locals across T; generated A/B retain twelve separate per-time carry ADDs and storage traffic. No invented dataflow parity.\n"
                +"Warm: 3 blocks x10000 calls/arm/case, cyclic ABC/BCA/CAB. Measured: 3x5000, same cyclic order.\n"
                +"One symmetric untimed validation call/arm before warm, after warm and after each measured round; these are controls, not extra warm windows. C's class is shared across cases; no JIT-state parity is claimed.\n"
                +"No retry/restart/fork/cartesian expansion. Fixed calls do not establish C2; use same-JVM jit.xml with events.tsv uptime bounds.\n"
                +"C1 failures, normal C2 entry, C2 OSR and unresolved attribution must be reported separately; source/bytecode counts are not machine-code speed proof.\n"
                +"mode="+(measure ? "MAIN released measurement" : "UNMEASURED preflight; no timing windows")+"\n"
                +"java="+System.getProperty("java.runtime.version")+"\n"
                +"vmArgs="+ManagementFactory.getRuntimeMXBean().getInputArguments()+"\n");
        if (measure) {
            Files.writeString(root.resolve("events.tsv"),"case\tphase\tround\tarm\tedge\tuptime_ms\tnano_time\tcalls\n");
            Files.writeString(root.resolve("samples.tsv"),"case\tphase\tround\tarm\tcalls\telapsed_ns\tns_per_call\tuptime_start_ms\tuptime_end_ms\n");
        }
        int caseIndex = 0;
        for (boolean scan : List.of(false,true)) for (int batch : List.of(32,256)) {
            String id = (scan ? "scan" : "ordinary")+"-FLOAT32-B"+batch+"-0";
            Path dir = Files.createDirectory(root.resolve(id));
            try (var bound = bind(baseline,dir,scan,batch,OLD_SHA[caseIndex++])) {
                bound.validate();
                if (measure) {
                    for (int round = 0; round < 3; round++) for (int order = 0; order < 3; order++)
                        window(root,id,"warm",round,(round+order)%3,10000,bound.actions[(round+order)%3]);
                    bound.validate();
                    for (int round = 0; round < 3; round++) {
                        for (int order = 0; order < 3; order++)
                            window(root,id,"sample",round,(round+order)%3,5000,bound.actions[(round+order)%3]);
                        bound.validate();
                    }
                }
                Files.writeString(dir.resolve("EQUALITY-PASS.txt"),"Literal old == candidate raw outputs; candidate == original bound entry; original bound entry == unchanged independent Java.\n");
            }
        }
        Files.writeString(root.resolve("COMPLETE.txt"),measure ? "All four finite cases completed; no retry.\n" : "Four untimed structural/equality/cache preflights passed. No measurement released.\n");
    }

    /** Captures incremental start/end uptime bounds; file I/O and controls are outside timing. */
    private static void window(Path root,String id,String phase,int round,int arm,int calls,Runnable action) throws Exception {
        String label = "ABC".substring(arm,arm+1);
        long uptimeStart = ManagementFactory.getRuntimeMXBean().getUptime();
        Files.writeString(root.resolve("events.tsv"),id+"\t"+phase+"\t"+round+"\t"+label+"\tstart\t"+uptimeStart+"\t"+System.nanoTime()+"\t"+calls+"\n",StandardOpenOption.APPEND);
        long start = System.nanoTime();
        for (int call = 0; call < calls; call++) action.run();
        long elapsed = System.nanoTime()-start;
        long uptimeEnd = ManagementFactory.getRuntimeMXBean().getUptime();
        sink ^= calls;
        Files.writeString(root.resolve("events.tsv"),id+"\t"+phase+"\t"+round+"\t"+label+"\tend\t"+uptimeEnd+"\t"+System.nanoTime()+"\t"+calls+"\n",StandardOpenOption.APPEND);
        Files.writeString(root.resolve("samples.tsv"),id+"\t"+phase+"\t"+round+"\t"+label+"\t"+calls+"\t"+elapsed+"\t"+(double)elapsed/calls+"\t"+uptimeStart+"\t"+uptimeEnd+"\n",StandardOpenOption.APPEND);
    }

    /**
     * Borrows only the old campaign's unchanged source fixture factories, never its matrix,
     * measured test or timing main. No CPU0074 window can be reached from this method.
     */
    static AutoCloseable fixture(boolean scan,DataType type,int batch,boolean vector,boolean parallel,Path dir) throws Exception {
        Files.createDirectories(dir);
        var method = CpuRowFusionSimdParallelTimingTest.class.getDeclaredMethod(scan ? "scan" : "ordinary",
                DataType.class,int.class,boolean.class,boolean.class,Path.class);
        method.setAccessible(true);
        try { return (AutoCloseable)method.invoke(null,type,batch,vector,parallel,dir); }
        catch (InvocationTargetException failure) { throw new IllegalStateException("fixture construction failed",failure.getCause()); }
    }

    static Object property(Object object,String name) throws Exception {
        var accessor = object.getClass().getDeclaredMethod(name); accessor.setAccessible(true); return accessor.invoke(object);
    }

    private static Object field(Object object,String name) throws Exception {
        for (Class<?> owner = object.getClass(); owner != null; owner = owner.getSuperclass()) try {
            var field = owner.getDeclaredField(name); field.setAccessible(true); return field.get(object);
        } catch (NoSuchFieldException absent) { }
        throw new NoSuchFieldException(name);
    }

    private static BoundCase bind(Path baseline,Path dir,boolean scan,int batch,String oldSha) throws Exception {
        AutoCloseable fixture = fixture(scan,DataType.FLOAT32,batch,false,false,dir);
        Arena arena = Arena.ofShared();
        try {
            var run = (CpuRecurrentScanExecutionTest.Fixture)property(fixture,"run");
            var java = (Runnable)property(fixture,"java");
            var originalCheck = (Runnable)property(fixture,"check");
            PreparedExecutable child = null; CpuGeneratedKernel artifact = null;
            for (var candidate : ((CpuPreparedPartitionExecutable)run.executable).children()) {
                CpuGeneratedKernel a = candidate instanceof CpuPreparedExecutable ordinary ? ordinary.artifact()
                        : candidate instanceof CpuGeneralScanPreparedExecutable ? (CpuGeneratedKernel)field(candidate,"artifact") : null;
                if (a != null && (scan ? a.specialization().generalScanIr().flatMap(s -> s.stages().flatMap(st -> st.rowMapReductionIr())).isPresent()
                        : a.specialization().rowMapReductionIr().isPresent())) { child = candidate; artifact = a; break; }
            }
            assertNotNull(child); assertNotNull(artifact);
            var spec = artifact.specialization();
            assertEquals(0,spec.vectorSpeciesBitSize()); assertTrue(spec.rowFusionEmission().isEmpty());
            String marker = "|"+(scan ? "scan" : "ordinary")+"-row-scalar-load-address-reuse-v1=";
            assertTrue(new String(spec.compatibilityBytes(),StandardCharsets.US_ASCII).contains(marker));
            assertTrue(new String(spec.classIdentityBytes(),StandardCharsets.US_ASCII).contains(marker));
            assertEquals(scan ? 71 : 72,spec.classIdentitySchema());
            var region = scan ? spec.generalScanIr().orElseThrow().stages().orElseThrow().rowMapReductionIr().orElseThrow() : spec.rowMapReductionIr().orElseThrow();
            int reads = scan ? spec.generalScanIr().orElseThrow().scalarReads().size()
                    : (int)region.arithmetic().values().stream().filter(v -> v.kind() == CpuKernelIr.Value.Kind.INPUT).count();
            int boundaries = spec.carrierPattern().size();
            MemorySegment[] original; long[] geometry;
            long end;
            if (scan) {
                var invocation = child.bind(run.state);
                original = ((MemorySegment[])field(invocation,"segments")).clone();
                geometry = ((long[])field(invocation,"geometry")).clone(); end = geometry[1];
            } else {
                original = new MemorySegment[boundaries]; geometry = new long[1+boundaries]; geometry[0] = batch; end = 1;
                var storage = List.copyOf(run.storage.values());
                Method selection = null;
                for (Class<?> owner = child.getClass(); owner != null && selection == null; owner = owner.getSuperclass()) try {
                    selection = owner.getDeclaredMethod("bufferSelection",int.class);
                } catch (NoSuchMethodException absent) { }
                assertNotNull(selection); selection.setAccessible(true);
                for (int p = 0; p < boundaries; p++) {
                    var selected = (PreparedExecutable.BufferSelection)selection.invoke(child,p);
                    // Fixture.storage contains exact logical interior slices, not the arena
                    // allocation's enclosing capacity. No extra byte offset is introduced.
                    original[p] = storage.get(selected.bufferIndex());
                }
            }
            MemorySegment[][] carriers = {original.clone(),original.clone()};
            for (int arm = 0; arm < 2; arm++) for (int p = 0; p < original.length; p++) {
                if (p < reads) carriers[arm][p] = original[p].asReadOnly();
                else carriers[arm][p] = arena.allocate(original[p].byteSize(),8);
            }
            byte[] old = Files.readAllBytes(baseline.resolve("cpu0074-measure").resolve(dir.getFileName()).resolve("selected.class"));
            assertEquals(oldSha,sha(old),"authentic old ClassFile, not candidate-regenerated control");
            byte[] current = Files.readAllBytes(dir.resolve("selected.class"));
            assertArrayEquals(artifact.classBytes(),current,"inspection and actual bound candidate must agree");
            assertThrows(IllegalArgumentException.class,() -> new CpuClassFileKernelGenerator().defineClassBytes(spec, scan ? spec.generalScanIr().orElseThrow().encodedKernelIr() : region.encodedKernelIr(), old),"affected old cache entry is stale");
            assertTrue(ClassFile.of().verify(old).isEmpty());
            var lookup = MethodHandles.privateLookupIn(CpuClassFileKernelGenerator.class,MethodHandles.lookup());
            var oldLookup = lookup.defineHiddenClass(old,false,MethodHandles.Lookup.ClassOption.NESTMATE);
            var oldHandle = oldLookup.findStatic(oldLookup.lookupClass(),"invoke",spec.entryType());
            // Setup has already executed its candidate. Define the measured candidate afresh,
            // with its real metadata, so neither A nor B inherits setup execution/JIT history.
            var newHandle = new CpuClassFileKernelGenerator().defineClassBytes(spec, scan ? spec.generalScanIr().orElseThrow().encodedKernelIr() : region.encodedKernelIr(), current).entryPoint();
            String oldDescriptor = ClassFile.of().parse(old).methods().stream().filter(m -> m.methodName().equalsString("invoke")).findFirst().orElseThrow().methodType().stringValue();
            assertEquals(spec.entryType().descriptorString(),oldDescriptor,"literal old ABI matches, no metadata fabrication");
            MethodHandle a = oldHandle.asSpreader(0,MemorySegment[].class,original.length);
            MethodHandle b = newHandle.asSpreader(0,MemorySegment[].class,original.length);
            Runnable[] actions = {entry(a,carriers[0],geometry.clone(),end),entry(b,carriers[1],geometry.clone(),end),java};
            String candidateClass = MethodHandles.privateLookupIn(CpuClassFileKernelGenerator.class,MethodHandles.lookup())
                    .revealDirect(newHandle).getDeclaringClass().getName();
            Files.write(dir.resolve("authentic-old.class"),old);
            Files.writeString(dir.resolve("identities.txt"),"A.sha256="+sha(old)+"\nA.hidden="+oldLookup.lookupClass().getName()
                    +"\nB.sha256="+sha(current)+"\nB.hidden="+candidateClass+"\nC.class="+CpuRowFusionSimdParallelJavaOracle.class.getName()+"$F32"
                    +"\nC.sha256="+ORACLE_SHA+"\nentry="+oldDescriptor+"\nreads="+reads+" boundaries="+boundaries
                    +"\ngeometry="+Arrays.toString(geometry)+"\ncompatibility="+new String(spec.compatibilityBytes(),StandardCharsets.US_ASCII)
                    +"\nclassIdentity="+new String(spec.classIdentityBytes(),StandardCharsets.US_ASCII)+"\n");
            census(dir,old,current,scan);
            trace(dir,region);
            cacheProof(dir,spec,scan ? spec.generalScanIr().orElseThrow().encodedKernelIr() : region.encodedKernelIr(),old);
            int firstWrite = reads, lastWrite = boundaries;
            Runnable validation = () -> {
                for (int arm = 0; arm < 2; arm++) actions[arm].run();
                actions[2].run(); originalCheck.run();
                for (int p = firstWrite; p < lastWrite; p++) {
                    assertEquals(-1,carriers[0][p].mismatch(carriers[1][p]),"old/candidate raw output "+p);
                    assertEquals(-1,original[p].mismatch(carriers[1][p]),"direct ABI/original checked bound entry "+p);
                }
            };
            return new BoundCase(fixture,arena,actions,validation);
        } catch (Throwable failure) {
            arena.close(); fixture.close(); throw failure;
        }
    }

    private static Runnable entry(MethodHandle entry,MemorySegment[] carriers,long[] geometry,long end) {
        return () -> {
            try { entry.invokeExact(carriers,geometry,0L,end); }
            catch (Throwable failure) { throw new IllegalStateException("exact captured entry failed",failure); }
        };
    }

    private static void verifyOracle(Path baseline) throws Exception {
        byte[] loaded;
        try (var input = CpuRowFusionSimdParallelJavaOracle.class.getResourceAsStream("CpuRowFusionSimdParallelJavaOracle$F32.class")) {
            assertNotNull(input); loaded = input.readAllBytes();
        }
        assertEquals(ORACLE_SHA,sha(loaded),"unchanged independently compiled original oracle");
        assertArrayEquals(Files.readAllBytes(baseline.resolve("oracle-classes/CpuRowFusionSimdParallelJavaOracle$F32.class")),loaded);
    }

    private static void cacheProof(Path dir,CpuKernelSpecialization spec,CpuKernelIr ir,byte[] old) throws Exception {
        Path cache = Files.createDirectory(dir.resolve("cache-proof"));
        var envelope = Class.forName("io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest")
                .getDeclaredMethod("envelope",String.class,byte[].class,byte[].class);
        envelope.setAccessible(true);
        Files.write(cache.resolve(spec.structuralKey()+".artifact"),(byte[])envelope.invoke(null,spec.structuralKey(),spec.compatibilityBytes(),old));
        var clear = CpuGeneratedKernelArtifactStore.class.getDeclaredMethod("clearLoadedForTests"); clear.setAccessible(true); clear.invoke(null);
        var observed = CpuGeneratedKernelArtifactStore.class.getDeclaredMethod("loadOrGenerateObserved",CpuKernelSpecialization.class,CpuKernelIr.class); observed.setAccessible(true);
        var store = new CpuGeneratedKernelArtifactStore(Optional.of(cache));
        var first = observed.invoke(store,spec,ir);
        assertEquals("GENERATED",property(first,"source").toString());
        assertSame(property(first,"artifact"),store.loadOrGenerate(spec,ir));
        clear.invoke(null);
        assertEquals("PERSISTED_HIT",property(observed.invoke(store,spec,ir),"source").toString());
        Files.writeString(dir.resolve("CACHE-PASS.txt"),"Authentic old under current envelope rejected; generated -> same live artifact -> persisted reload.\n");
    }

    /** Counts actual minimal numeric row loops, not whole-method or source-level estimates. */
    private static void census(Path dir,byte[] old,byte[] current,boolean scan) throws Exception {
        var report = new StringBuilder("arm\tloads\tstores\tFMUL\tFADD\tFSUB\tLMUL\tLADD\trow_instructions\tcode_bytes\tmax_locals\twhole_FADD\n");
        for (int arm = 0; arm < 2; arm++) {
            byte[] bytes = arm == 0 ? old : current;
            var method = ClassFile.of().parse(bytes).methods().stream().filter(m -> m.methodName().equalsString("invoke")).findFirst().orElseThrow();
            var code = (CodeAttribute)method.code().orElseThrow();
            var at = new TreeMap<Integer,Instruction>(); int bci = 0;
            for (var e : code) if (e instanceof Instruction i) { at.put(bci,i); bci += i.sizeInBytes(); }
            NavigableMap<Integer,Instruction> row = null;
            for (var e : at.entrySet()) if (e.getValue() instanceof BranchInstruction back && back.opcode() == Opcode.GOTO && code.labelToBci(back.target()) < e.getKey()) {
                var loop = at.subMap(code.labelToBci(back.target()),true,e.getKey(),false);
                if (count(loop.values(),Opcode.FMUL) == 0 || loop.entrySet().stream().anyMatch(x -> x.getValue() instanceof BranchInstruction br && br.opcode() == Opcode.GOTO && code.labelToBci(br.target()) < x.getKey())) continue;
                assertNull(row,"one actual row traversal"); row = loop;
            }
            assertNotNull(row);
            long loads = calls(row.values(),"get"), stores = calls(row.values(),"set");
            assertEquals(scan ? 24 : 28,count(row.values(),Opcode.FMUL));
            assertEquals(scan ? 24 : 22,count(row.values(),Opcode.FADD));
            assertEquals(2,count(row.values(),Opcode.FSUB)); assertEquals(scan ? 5 : 7,stores);
            if (arm == 0) assertEquals(scan ? 27 : 32,loads); else if (!scan) assertEquals(20,loads); else assertTrue(loads < 27);
            if (scan) assertEquals(36,count(at.values(),Opcode.FADD),"twelve distinct per-time carry ADD sites retained");
            report.append(arm == 0 ? "A" : "B").append('\t').append(loads).append('\t').append(stores);
            for (var op : List.of(Opcode.FMUL,Opcode.FADD,Opcode.FSUB,Opcode.LMUL,Opcode.LADD)) report.append('\t').append(count(row.values(),op));
            report.append('\t').append(row.size()).append('\t').append(code.codeLength()).append('\t').append(code.maxLocals())
                    .append('\t').append(count(at.values(),Opcode.FADD)).append('\n');
        }
        Files.writeString(dir.resolve("bytecode.tsv"),report);
    }

    private static long count(Collection<Instruction> code,Opcode op) { return code.stream().filter(i -> i.opcode() == op).count(); }
    private static long calls(Collection<Instruction> code,String name) {
        return code.stream().filter(i -> i instanceof InvokeInstruction call && call.owner().asInternalName().equals("java/lang/foreign/MemorySegment") && call.name().equalsString(name)).count();
    }

    /** Records the exact nominal load sequence and duplicate intervals, including store epochs. */
    @SuppressWarnings("unchecked")
    private static void trace(Path dir,CpuRowMapReductionIr region) throws Exception {
        var method = CpuClassFileKernelGenerator.class.getDeclaredMethod("regionScalarTrace",CpuRowMapReductionIr.class);
        method.setAccessible(true);
        var trace = (List<List<Long>>)method.invoke(null,region);
        var first = new LinkedHashMap<List<Long>,Integer>(); var last = new HashMap<List<Long>,Integer>(); var counts = new HashMap<List<Long>,Integer>();
        for (int i = 0; i < trace.size(); i++) { first.putIfAbsent(trace.get(i),i); last.put(trace.get(i),i); counts.merge(trace.get(i),1,Integer::sum); }
        var text = new StringBuilder("Key=root,fixedC,fixedB,rowC,rowB,storeEpoch; interval indices are external requests, not hardware registers.\n");
        int peak = 0;
        for (int i = 0; i < trace.size(); i++) {
            int live = 0;
            for (var k : first.keySet()) if (counts.get(k) > 1 && first.get(k) <= i && last.get(k) >= i) live++;
            peak = Math.max(peak,live); text.append(i).append('\t').append(trace.get(i)).append("\tlive=").append(live).append('\n');
        }
        text.append("requests=").append(trace.size()).append(" distinct=").append(first.size()).append(" peak_duplicate_intervals=").append(peak).append('\n');
        var budgetMethod = CpuClassFileKernelGenerator.class.getDeclaredMethod("regionScalarBudget",CpuRowMapReductionIr.class);
        budgetMethod.setAccessible(true);
        int remainingSlots = (int)budgetMethod.invoke(null,region), typedWidth = region.arithmetic().values().getFirst().dataType() == DataType.FLOAT64 ? 2 : 1;
        int baseSlots = 128-remainingSlots, offsetSlots = remainingSlots < 2 ? 0 : 2;
        int capacity = Math.max(0,remainingSlots-2)/typedWidth;
        text.append("cold_inline_base_slots=").append(baseSlots).append(" recycled_LONG_slots=").append(offsetSlots)
                .append(" typed_pool_capacity=").append(capacity).append(" nominal_peak_cache_slots=").append(peak*typedWidth)
                .append(" inline_total_bound=").append(baseSlots+offsetSlots+Math.min(peak,capacity)*typedWidth).append(" <=128\n");
        text.append("Base includes producer/terminal cells, carrier/base bindings, existing typed temporaries and control. Parallel helper applies its separate stricter bound.\n");
        assertTrue(baseSlots+offsetSlots+Math.min(peak,capacity)*typedWidth <= 128);
        for (var k : first.keySet()) if (counts.get(k) > 1) text.append(k).append("\tfirst=").append(first.get(k)).append(" last=").append(last.get(k)).append(" uses=").append(counts.get(k)).append('\n');
        text.append("Scan cache excludes evolving carries/scratch and SUM alternatives; this nominal interval trace is an upper bound, not an assertion of actual cache hits.\n");
        Files.writeString(dir.resolve("load-lifetimes.txt"),text);
        if (dir.getFileName().toString().startsWith("ordinary")) { assertEquals(32,trace.size()); assertEquals(20,first.size()); assertEquals(4,peak); }
    }

    static String sha(byte[] bytes) throws Exception { return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)); }

    /** Captures the already used test runtime, without changing a build file or launching a JVM. */
    private static void captureClasspath(Path root) throws Exception {
        var classpath = new LinkedHashSet<String>();
        // Gradle's modular test launch can use the JDK application loader, not URLClassLoader.
        // Capture its actual resolved module file locations as well as ordinary classpath.
        for (String key : List.of("java.class.path","jdk.module.path"))
            for (String entry : System.getProperty(key,"").split(java.util.regex.Pattern.quote(java.io.File.pathSeparator)))
                if (!entry.isBlank()) classpath.add(entry);
        ModuleLayer layer = CpuRowScalarReuseComparison.class.getModule().getLayer();
        if (layer != null) for (var module : layer.configuration().modules()) module.reference().location().ifPresent(uri -> {
            if (uri.getScheme().equals("file")) classpath.add(Path.of(uri).toString());
        });
        for (ClassLoader loader = CpuRowScalarReuseComparison.class.getClassLoader(); loader != null; loader = loader.getParent())
            if (loader instanceof java.net.URLClassLoader urls) for (var url : urls.getURLs())
                if (url.getProtocol().equals("file")) classpath.add(Path.of(url.toURI()).toString());
        if (classpath.isEmpty()) throw new IllegalStateException("cannot capture existing test runtime classpath safely");
        Files.writeString(root.resolve("classpath.txt"),String.join(java.io.File.pathSeparator,classpath));
    }

    /** Owns only test fixtures and distinct test outputs; all production owners remain unchanged. */
    private record BoundCase(AutoCloseable fixture,Arena arena,Runnable[] actions,Runnable check) implements AutoCloseable {
        void validate() { check.run(); }
        @Override public void close() throws Exception { try { arena.close(); } finally { fixture.close(); } }
    }
}
