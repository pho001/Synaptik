package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest;
import io.github.pho001.synaptik.backend.cpu.internal.cache.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.compiler.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import java.nio.file.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/**
 * Source-matched, unmeasured CPU0071 old/new artifact capture. The baseline mode is run only
 * before production edits; comparison always reads frozen old bytes, never regenerates them.
 * Conditional timing fixtures retain actual Compiler siblings and numeric geometry separately
 * from pure generated ClassFiles, so old/new/direct can subsequently share one measured JVM.
 */
public final class CpuMatmulBindingSnapshotTest {
    private final Map<String,String[]> frozen = new HashMap<>();
    /** Actual pre-edit compatibility streams record 128-bit preferred F32 and F64 species. */
    private static final int CAPTURED_VECTOR_BITS = 128;
    private int vectorCoverageSkipped;

    /**
     * Recovers value-based publication/constant witnesses from immutable accepted-before CPU
     * classes. The original eight identity-text diagnostics remain untouched. Old-source whole
     * ClassFiles must first match the sealed pre-edit artifacts; candidate output supplies no
     * expected value. This recovery performs no kernel execution or timing.
     * @throws Exception if old-source isolation, literal provenance or immutable capture fails
     */
    @Test void recoverConditionalBindingValuesFromImmutableBeforeSource() throws Exception {
        String directory = System.getenv("CPU0072_OLD_CLASSES");
        org.junit.jupiter.api.Assumptions.assumeTrue(directory != null);
        Path root = Path.of(System.getenv("CPU0072_ARTIFACT_DIR"));
        Path target = root.resolve("binding-before-values");
        assertFalse(Files.exists(target)); Files.createDirectories(target);
        try (var loader = new java.net.URLClassLoader(new java.net.URL[]{Path.of(directory).toUri().toURL()},getClass().getClassLoader()) {
            /**
             * Isolates CPU before-source classes while retaining shared Compiler/Prepare types.
             * @param name requested binary name
             * @param resolve whether to resolve linkage
             * @return old CPU implementation or shared parent type
             * @throws ClassNotFoundException if the immutable implementation is missing
             */
            @Override protected Class<?> loadClass(String name,boolean resolve) throws ClassNotFoundException {
                synchronized (getClassLoadingLock(name)) {
                    Class<?> result = findLoadedClass(name);
                    if (result == null && name.startsWith("io.github.pho001.synaptik.backend.cpu.")) result = findClass(name);
                    if (result == null) result = super.loadClass(name,false);
                    if (resolve) resolveClass(result); return result;
                }
            }
        }) {
            String prefix = "io.github.pho001.synaptik.backend.cpu.internal.";
            var inputs = (io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs)loader.loadClass(prefix+"prepare.CpuPartitionAnalysisInputs").getField("DEFAULT").get(null);
            var preparer = loader.loadClass(prefix+"prepare.CpuPartitionPreparer");
            var generator = loader.loadClass(prefix+"codegen.emit.CpuClassFileKernelGenerator");
            int count = 0;
            for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean length : List.of(false,true)) {
                var compiled = conditionalFamily(type,length);
                var analysis = (io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis<?>)preparer
                        .getMethod("analyze",io.github.pho001.synaptik.prepare.analysis.PrepareContext.class)
                        .invoke(preparer.getConstructor().newInstance(),GraphPreparation.project(compiled,compiled.partitions().getFirst(),inputs));
                var units = (List<?>)analysis.plan().getClass().getMethod("units").invoke(analysis.plan()); int index = 0;
                for (var unit : units) {
                    Object facts = unit.getClass().getMethod("runtimeFacts").invoke(unit);
                    if (((Optional<?>)facts.getClass().getMethod("generalScan").invoke(facts)).isEmpty()) continue;
                    Object plan = unit.getClass().getMethod("portablePlan").invoke(unit);
                    Object spec = plan.getClass().getMethod("specialization").invoke(plan);
                    Object ir = plan.getClass().getMethod("kernelIr").invoke(plan);
                    String name = "conditional-"+type+"-"+length+"-"+index++;
                    byte[] bytes = (byte[])generator.getMethod("generateClassBytes",spec.getClass(),ir.getClass())
                            .invoke(generator.getConstructor().newInstance(),spec,ir);
                    assertArrayEquals(Files.readAllBytes(root.resolve("baseline").resolve(name+".class")),bytes,name+" before-source provenance");
                    var boundaries = (List<?>)unit.getClass().getMethod("boundaryValues").invoke(unit);
                    Files.writeString(target.resolve(name+".bindings"),conditionalBindingValues(boundaries,compiled)); count++;
                }
            }
            assertEquals(8,count);
        }
    }

    /**
     * Serializes scoped value bindings and positional ties, excluding process-global Tensor IDs.
     * @param boundaries ordered old or current unit ValueIds
     * @param compiled shared actual Compiler forward/reverse artifact
     * @return deterministic publication, target-to-input positions and typed constant values
     */
    private static String conditionalBindingValues(List<?> boundaries,CompileArtifacts compiled) {
        var inputs = compiled.constants().bindableInputBindings();
        return boundaries+"\n"+compiled.publication().forwardBindings().stream().map(b -> b.valueId()).toList()
                +"\n"+compiled.publication().gradientBindings().stream().map(b -> List.of(b.valueId(),b.derivativeOrder(),b.targetIndex(),
                    java.util.stream.IntStream.range(0,inputs.size()).filter(i -> inputs.get(i).tensorId().equals(b.target())).findFirst().orElse(-1))).toList()
                +"\n"+compiled.constants().bindableInputs()+"\n"+compiled.constants().constantSources();
    }

    /** Exercises nonmatching-host qualification without dropping any scalar control. */
    @Test void frozenVectorCorpusRequiresMatchingSpeciesButScalarChecksRemainUnconditional() {
        for (var form : CpuMatmulIr.Realization.values()) for (int bits : List.of(128,256,512)) {
            boolean scalar = form == CpuMatmulIr.Realization.DIRECT_SCALAR || form == CpuMatmulIr.Realization.TILED_SCALAR_2X2;
            assertEquals(scalar || bits == CAPTURED_VECTOR_BITS,qualifiesFrozenSpecies(form,bits));
        }
    }

    /**
     * Qualifies only species-dependent controls against the actual captured 128-bit corpus.
     * No vector evidence is regenerated for a different host; scalar checks always remain live.
     * @param form selected ordinary MATMUL realization
     * @param bits actual preferred species bit size, or zero for scalar generation
     * @return whether this host can compare this realization with the frozen corpus
     */
    private static boolean qualifiesFrozenSpecies(CpuMatmulIr.Realization form,int bits) {
        return form != CpuMatmulIr.Realization.DIRECT_N_VECTOR && form != CpuMatmulIr.Realization.TILED_N_VECTOR_2X2
                || bits == CAPTURED_VECTOR_BITS;
    }

    /**
     * Qualifies the changed ordinary-batch K seam against immutable-before-source generation.
     * An explicit old-classes directory enables cold recovery only; normal CI compares checked-in
     * hashes from that recovery, with no private filesystem dependency or performance launch.
     * @throws Exception if isolated old-source analysis, generation or exact retention fails
     */
    @Test void ordinaryBatchAndTypedControlsRetainBothStreamsAndClassFiles() throws Exception {
        String oldDirectory = System.getenv("CPU0071_OLD_CLASSES");
        Map<String,String[]> expected = new HashMap<>();
        if (oldDirectory == null) try (var stream = getClass().getResourceAsStream("matmul-binding-ordinary-before.tsv")) {
            assertNotNull(stream);
            new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8).lines()
                    .filter(l -> !l.startsWith("#") && !l.isBlank()).map(l -> l.split("\t"))
                    .forEach(c -> expected.put(c[0],c));
        }
        var report = new StringBuilder(); int matmuls = 0, typed = 0;
        try (var loader = new java.net.URLClassLoader(oldDirectory == null ? new java.net.URL[0]
                : new java.net.URL[]{Path.of(oldDirectory).toUri().toURL()},getClass().getClassLoader()) {
            /**
             * Isolates recovered CPU implementation only; shared query and Prepare types agree.
             * @param name requested class name
             * @param resolve whether to resolve linkage
             * @return recovered old CPU class or shared parent class
             * @throws ClassNotFoundException if the requested implementation is unavailable
             */
            @Override protected Class<?> loadClass(String name,boolean resolve) throws ClassNotFoundException {
                synchronized (getClassLoadingLock(name)) {
                    Class<?> result = findLoadedClass(name);
                    if (result == null && oldDirectory != null && name.startsWith("io.github.pho001.synaptik.backend.cpu."))
                        result = findClass(name);
                    if (result == null) result = super.loadClass(name,false);
                    if (resolve) resolveClass(result); return result;
                }
            }
        }) {
            String prefix = "io.github.pho001.synaptik.backend.cpu.internal.";
            for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
                var b = new DynamicDimension("B");
                Tensor a = CpuSymbolicLengthScanFixture.tensor(type,true,Shape.ofDimensions(b,new StaticDimension(3)));
                Tensor w = CpuGeneralScanLoweringTest.tensor(type,true,type == DataType.FLOAT32 ? new long[]{3,2} : new long[]{3});
                Tensor value = type == DataType.FLOAT32 ? a.matmul(w) : a.add(w);
                var roots = type == DataType.FLOAT32 ? List.of(value) : List.of(value.sum(0,false),value.mean());
                var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                        roots.stream().map(r -> (FunctionalGradientRequest.OutputReference)new FunctionalGradientRequest.ForwardTensorReference(r)).toList(),
                        roots.stream().map(r -> Optional.of(CpuSymbolicLengthScanFixture.tensor(type,false,r.descriptor().shape()))).toList(),
                        List.of(a,w),false,FunctionalGradientRequest.DisconnectedPolicy.ZERO)));
                var compiled = CpuGeneralScanCapabilityTest.compile(roots,Optional.of(request),false,new CpuCapabilityProvider());
                var current = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT));
                List<?> oldUnits = null;
                if (oldDirectory != null) {
                    var inputs = (io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs)loader.loadClass(prefix+"prepare.CpuPartitionAnalysisInputs").getField("DEFAULT").get(null);
                    var oldPreparer = loader.loadClass(prefix+"prepare.CpuPartitionPreparer");
                    var analysis = (io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis<?>)oldPreparer
                            .getMethod("analyze",io.github.pho001.synaptik.prepare.analysis.PrepareContext.class)
                            .invoke(oldPreparer.getConstructor().newInstance(),GraphPreparation.project(compiled,compiled.partitions().getFirst(),inputs));
                    oldUnits = (List<?>)analysis.plan().getClass().getMethod("units").invoke(analysis.plan());
                    assertEquals(oldUnits.size(),current.plan().units().size());
                }
                for (int i = 0; i < current.plan().units().size(); i++) {
                    var plan = current.plan().units().get(i).portablePlan(); var spec = plan.specialization();
                    assertTrue(spec.ordinaryBatch());
                    if (spec.matmulIr().isPresent()) matmuls++;
                    if (type == DataType.FLOAT64) typed++;
                    String name = type+"-"+i;
                    byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(spec,plan.kernelIr());
                    byte[] compatibility = spec.compatibilityBytes(), identity = spec.classIdentityBytes();
                    for (var payload : Map.of("class",bytes,"compatibility",compatibility,"identity",identity,
                            "bindings",(current.plan().units().get(i).boundaryValues()+"\n"+spec.entryType()+"\n"
                                    +plan.batchBindings()+"\n"+plan.batchMatmul()+"\n"+plan.batchAggregate()).getBytes(java.nio.charset.StandardCharsets.UTF_8)).entrySet())
                        CpuPointwiseSumCompositionSnapshotTest.assertFrozenPayload("batch-"+name,payload.getKey(),payload.getValue(),0);
                    String compositionRoot = System.getenv("CPU0072_ARTIFACT_DIR");
                    if (compositionRoot != null) {
                        boolean before = "baseline".equals(System.getenv("CPU0072_CAPTURE"));
                        Path phase = Path.of(compositionRoot).resolve(before ? "baseline" : "candidate");
                        Files.createDirectories(phase);
                        for (var payload : Map.of("class",bytes,"compatibility",compatibility,"identity",identity,
                                "bindings",(current.plan().units().get(i).boundaryValues()+"\n"+spec.entryType()+"\n"
                                        +plan.batchBindings()+"\n"+plan.batchMatmul()+"\n"+plan.batchAggregate())
                                        .getBytes(java.nio.charset.StandardCharsets.UTF_8)).entrySet()) {
                            Path file = phase.resolve("batch-"+name+"."+payload.getKey());
                            if (before) assertFalse(Files.exists(file),"immutable CPU0072 baseline");
                            else assertArrayEquals(Files.readAllBytes(Path.of(compositionRoot).resolve("baseline").resolve(file.getFileName())),payload.getValue(),file.toString());
                            Files.write(file,payload.getValue());
                        }
                    }
                    if (oldUnits != null) {
                        Object unit = oldUnits.get(i), oldPlan = unit.getClass().getMethod("portablePlan").invoke(unit);
                        Object oldSpec = oldPlan.getClass().getMethod("specialization").invoke(oldPlan);
                        Object oldIr = oldPlan.getClass().getMethod("kernelIr").invoke(oldPlan);
                        var generator = loader.loadClass(prefix+"codegen.emit.CpuClassFileKernelGenerator");
                        byte[] oldBytes = (byte[])generator.getMethod("generateClassBytes",oldSpec.getClass(),oldIr.getClass())
                                .invoke(generator.getConstructor().newInstance(),oldSpec,oldIr);
                        assertArrayEquals(oldBytes,bytes,name+" old-source ClassFile");
                        assertArrayEquals((byte[])oldSpec.getClass().getMethod("compatibilityBytes").invoke(oldSpec),compatibility,name);
                        assertArrayEquals((byte[])oldSpec.getClass().getMethod("classIdentityBytes").invoke(oldSpec),identity,name);
                        Path output = Path.of(System.getenv("CPU0071_ARTIFACT_DIR")).resolve("ordinary-recovered"); Files.createDirectories(output);
                        Files.write(output.resolve(name+".class"),oldBytes);
                        Files.write(output.resolve(name+".compatibility"),compatibility); Files.write(output.resolve(name+".identity"),identity);
                    } else {
                        assertNotNull(expected.get(name),name);
                        assertEquals(expected.get(name)[1],CpuSymbolicScanGeneratedKernelTest.sha(bytes),name);
                        assertEquals(expected.get(name)[2],CpuSymbolicScanGeneratedKernelTest.sha(compatibility),name);
                        assertEquals(expected.get(name)[3],CpuSymbolicScanGeneratedKernelTest.sha(identity),name);
                    }
                    report.append(name).append('\t').append(CpuSymbolicScanGeneratedKernelTest.sha(bytes)).append('\t')
                            .append(CpuSymbolicScanGeneratedKernelTest.sha(compatibility)).append('\t').append(CpuSymbolicScanGeneratedKernelTest.sha(identity)).append('\n');
                }
            }
        }
        assertEquals(3,matmuls,"ordinary-batch-v1 forward/dX/dW full-K artifacts"); assertTrue(typed > 0);
        if (oldDirectory != null) Files.writeString(Path.of(System.getenv("CPU0071_ARTIFACT_DIR")).resolve("ordinary-recovered/manifest.tsv"),report);
    }
    /**
     * Captures the affected/control matrix without invoking a performance selector.
     * @throws Exception if capture, byte verification or frozen-baseline comparison fails
     */
    @Test
    void captureOrCompareFrozenArtifacts() throws Exception {
        String directory = System.getenv("CPU0072_ARTIFACT_DIR");
        boolean composition = directory != null;
        if (!composition) directory = System.getenv("CPU0071_ARTIFACT_DIR");
        Path root = directory == null ? null : Path.of(directory);
        boolean baseline = "baseline".equals(System.getenv(composition ? "CPU0072_CAPTURE" : "CPU0071_CAPTURE"));
        Path target = root == null ? null : root.resolve(baseline ? "baseline" : "candidate");
        if (baseline) { assertNotNull(target); assertFalse(Files.exists(target.resolve("manifest.tsv")), "never overwrite a baseline"); }
        if (target != null) Files.createDirectories(target);
        if (root == null) try (var stream = getClass().getResourceAsStream("matmul-binding-before.tsv")) {
            assertNotNull(stream);
            for (String line : new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8).lines().toList())
                if (!line.startsWith("#") && !line.isBlank()) {
                    var columns = line.split("\t"); frozen.put(columns[0],columns);
                }
        }
        var report = new StringBuilder();
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            for (boolean reverse : List.of(false,true)) for (int time : List.of(0,4)) {
                for (int family = 0; family < 3; family++) {
                    var q = CpuGeneralScanLoweringTest.denseReadFixture(type,family,time,reverse);
                    captureScan(root,target,baseline,"static-"+type+"-"+reverse+"-"+time+"-"+family,
                            CpuGeneralScanLowering.analyze(q),report);
                    for (int batch : List.of(0,1,7)) for (int features : List.of(0,2))
                        captureScan(root,target,baseline,"length-"+type+"-"+reverse+"-"+time+"-"+family+"-"+batch+"-"+features,
                                CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.lengthFixture(type,family,time,batch,features,reverse,true)),report);
                }
            }
            ordinaryControls(root,target,baseline,type,report);
        }
        if (target != null) Files.writeString(target.resolve("manifest.tsv"),report);
        System.out.println("CPU0071 "+(baseline ? "BASELINE" : "COMPARISON")+" cases="+report.toString().lines().count()
                +" vectorCasesNotQualified="+vectorCoverageSkipped+" capturedVectorBits="+CAPTURED_VECTOR_BITS+" directory="+target);
    }

    /**
     * Compiles the exact T4/B7/F3/H2 conditional fixture, joint nonuniform seed inputs and
     * original-forward/actual-reverse traversal; it performs no timing or physical allocation.
     * @param type homogeneous admitted floating type
     * @param length whether the recurrence is length-aware TANH, otherwise h+x@W
     * @return actual combined compile artifact with two independently qualified Scan occurrences
     */
    public static CompileArtifacts conditionalFamily(DataType type,boolean length) {
        Tensor h = CpuGeneralScanLoweringTest.tensor(type,true,7,2), x = CpuGeneralScanLoweringTest.tensor(type,true,7,3);
        Tensor w = CpuGeneralScanLoweringTest.tensor(type,true,3,2), u = CpuGeneralScanLoweringTest.tensor(type,true,2,2);
        Tensor bias = CpuGeneralScanLoweringTest.tensor(type,true,2);
        Tensor next = length ? x.matmul(w).add(h.matmul(u)).add(bias).tanh() : h.add(x.matmul(w));
        Tensor initial = CpuGeneralScanLoweringTest.tensor(type,true,7,2), seq = CpuGeneralScanLoweringTest.tensor(type,true,4,7,3);
        Tensor wx = CpuGeneralScanLoweringTest.tensor(type,true,3,2), wh = CpuGeneralScanLoweringTest.tensor(type,true,2,2), offset = CpuGeneralScanLoweringTest.tensor(type,true,2);
        var actual = length ? List.of(wx,wh,offset) : List.of(wx);
        var formal = length ? List.of(w,u,bias) : List.of(w);
        var body = new ScanBody(List.of(h),List.of(x),formal,List.of(next),List.of(next));
        var scan = length ? Scan.apply(List.of(initial),List.of(seq),actual,CpuGeneralScanLoweringTest.tensor(DataType.INT64,false,7),
                1,List.of(OptionalInt.of(0)),body,false) : Scan.apply(List.of(initial),List.of(seq),actual,body,false);
        var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
        var wrt = new ArrayList<>(List.of(initial,seq)); wrt.addAll(actual);
        return CpuGeneralScanCapabilityTest.compile(roots,Optional.of(CpuGeneralScanCapabilityTest.request(roots,wrt)),false,new CpuCapabilityProvider());
    }

    /**
     * Saves immutable identity streams, ClassFile, and invocation geometry for one occurrence.
     * @param root evidence root holding an immutable baseline directory
     * @param target fresh baseline or candidate artifact directory
     * @param baseline whether this is the sole pre-edit capture
     * @param name stable test-local occurrence identifier
     * @param facts complete admitted lowering facts, never sampled symbolic geometry
     * @param report append-only in-memory manifest
     * @throws Exception if persistence, verification or compatibility assertions fail
     */
    private void captureScan(Path root,Path target,boolean baseline,String name,CpuGeneralScanLowering.Facts facts,StringBuilder report) throws Exception {
        var spec = CpuGeneralScanGeneratedKernelTest.specialization(facts);
        capture(root,target,baseline,name,spec,facts.ir().encodedKernelIr(),report);
        String geometry = geometryDescription(facts);
        boolean regional = facts.ir().stages().flatMap(p -> p.rowMapReductionIr()).isPresent();
        if (!regional) CpuPointwiseSumCompositionSnapshotTest.assertFrozenPayload(name,"geometry",geometry.getBytes(java.nio.charset.StandardCharsets.UTF_8),0);
        else {
            CpuPointwiseSumCompositionSnapshotTest.assertValueWitness(geometry);
            var stages = facts.ir().stages().orElseThrow();
            if (facts.ir().symbolic()) assertEquals(stages.workspaceSize(facts.ir().dataType().byteWidth()),facts.geometry().workspaceSize(),"selected symbolic resource coefficients match the cold proved cuts");
            else assertEquals(stages.scratchBytes(),facts.geometry().scratchBytes(),"selected fixed resource declaration matches its cold proved cuts");
            for (int retained : stages.rowMapReductionIr().orElseThrow().retainedStores())
                assertTrue(stages.cuts().stream().anyMatch(cut -> cut.value() == retained),"selected region retains every saved/public store");
        }
        if (target != null) Files.writeString(target.resolve(name+".geometry"),geometry);
        if (!baseline && root != null) assertEquals(Files.readString(root.resolve("baseline").resolve(name+".geometry")),geometry,name+" geometry/resources retained");
        if (!baseline && root == null && !regional) assertFrozenGeometry(name,facts);
    }

    /**
     * Authenticates current fixture storage/packing against the actual pre-edit geometry capture.
     * @param name exact frozen scan occurrence identifier
     * @param facts current facts for the same source query
     * @throws Exception if the resource is absent or geometry/resource/ABI meaning changed
     */
    static void assertFrozenGeometry(String name,CpuGeneralScanLowering.Facts facts) throws Exception {
        try (var stream = CpuMatmulBindingSnapshotTest.class.getResourceAsStream("matmul-binding-geometry-before.tsv")) {
            assertNotNull(stream);
            var line = new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8).lines()
                    .filter(l -> l.startsWith(name+"\t")).findFirst().orElseThrow();
            assertEquals(line.split("\t")[1],CpuSymbolicScanGeneratedKernelTest.sha(geometryDescription(facts).getBytes(java.nio.charset.StandardCharsets.UTF_8)),name+" frozen resource/packing ABI");
        }
    }

    /**
     * Serializes immutable metadata only; no storage addresses or sampled layout inference.
     * @param facts admitted fixed or symbolic lowering facts
     * @return deterministic geometry/resource/access/pack witness for the baseline B=7 invocation
     */
    private static String geometryDescription(CpuGeneralScanLowering.Facts facts) {
        return facts.geometry()+"\n"+facts.spans()+"\n"+facts.accessBindings()+"\n"+Arrays.toString(facts.pack(7));
    }

    /**
     * Compares both identity streams and bytes, permitting only the approved static namespaces.
     * @param root evidence root containing frozen old artifacts
     * @param target output directory for the selected capture phase
     * @param baseline whether this is pre-edit generation
     * @param name stable occurrence name
     * @param spec exact generated specialization
     * @param ir canonical typed IR
     * @param report manifest accumulator
     * @throws Exception if capture or retained/changed-namespace checks fail
     */
    private void capture(Path root,Path target,boolean baseline,String name,CpuKernelSpecialization spec,CpuKernelIr ir,StringBuilder report) throws Exception {
        if (System.getenv("CPU0072_ARTIFACT_DIR") != null && spec.vectorSpeciesBitSize() != 0) {
            Path species = target.resolve(name+".species");
            if (!baseline) {
                int bits = Integer.parseInt(Files.readString(root.resolve("baseline").resolve(species.getFileName())));
                if (!CpuPointwiseSumCompositionSnapshotTest.qualifiesSpecies(spec.vectorSpeciesBitSize(),bits)) {
                    vectorCoverageSkipped++;
                    System.out.println("CPU0072 UNQUALIFIED vector row "+name+" captured="+bits+" host="+spec.vectorSpeciesBitSize());
                    report.append(name).append("\tUNQUALIFIED_SPECIES\t-\tvector-coverage-limitation\n");
                    return;
                }
            }
            if (baseline) assertFalse(Files.exists(species),"immutable captured species");
            Files.writeString(species,Integer.toString(spec.vectorSpeciesBitSize()));
        }
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(spec,ir);
        assertTrue(java.lang.classfile.ClassFile.of().verify(bytes).isEmpty(),name);
        var payloads = Map.of("class",bytes,"compatibility",spec.compatibilityBytes(),"identity",spec.classIdentityBytes());
        boolean regional = spec.generalScanIr().flatMap(s -> s.stages()).flatMap(s -> s.rowMapReductionIr()).isPresent();
        for (var payload : payloads.entrySet()) {
            if (regional) CpuPointwiseSumCompositionSnapshotTest.assertRetiredPayload(name,payload.getKey(),payload.getValue());
            else CpuPointwiseSumCompositionSnapshotTest.assertFrozenPayload(name,payload.getKey(),payload.getValue(),spec.vectorSpeciesBitSize());
        }
        boolean changed = false;
        for (var entry : payloads.entrySet()) {
            if (target != null) Files.write(target.resolve(name+"."+entry.getKey()),entry.getValue());
            if (!baseline) {
                byte[] old = root == null ? null : Files.readAllBytes(root.resolve("baseline").resolve(name+"."+entry.getKey()));
                int column = entry.getKey().equals("class") ? 1 : entry.getKey().equals("compatibility") ? 2 : 3;
                String oldHash = old == null ? frozen.get(name)[column] : CpuSymbolicScanGeneratedKernelTest.sha(old);
                String currentHash = CpuSymbolicScanGeneratedKernelTest.sha(entry.getValue());
                var scan = spec.generalScanIr();
                boolean allowed = regional || System.getenv("CPU0072_ARTIFACT_DIR") == null && scan.isPresent() && !scan.get().symbolic() && scan.get().stages().isPresent()
                        && scan.get().stages().orElseThrow().order().stream().anyMatch(s -> s.right() >= 0);
                if (allowed) {
                    assertNotEquals(oldHash,currentHash,name+" "+entry.getKey()+" must advance");
                    changed = true;
                    if (old != null && entry.getKey().equals("class")) assertThrows(IllegalArgumentException.class,
                            () -> new CpuClassFileKernelGenerator().defineClassBytes(spec, ir, old),name+" stale bytes");
                } else assertEquals(oldHash,currentHash,name+" "+entry.getKey()+" retained control");
            }
        }
        report.append(name).append('\t').append(spec.structuralKey()).append('\t')
                .append(CpuSymbolicScanGeneratedKernelTest.sha(bytes)).append('\t').append(changed ? "changed" : "retained").append('\n');
    }

    /**
     * Captures all ordinary floating realizations, TANH, arrays and segments as exact controls.
     * @param root immutable baseline root
     * @param target current phase output
     * @param baseline whether capturing old source
     * @param type represented floating type
     * @param report manifest accumulator
     * @throws Exception if generation or retained-byte assertions fail
     */
    private void ordinaryControls(Path root,Path target,boolean baseline,DataType type,StringBuilder report) throws Exception {
        var read = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,CpuAccessPlan.Regime.DENSE_LINEAR,2,List.of(CpuAccessPlan.AxisRole.CONTIGUOUS,CpuAccessPlan.AxisRole.CONTIGUOUS),2);
        var write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,CpuAccessPlan.Regime.DENSE_LINEAR,2,List.of(CpuAccessPlan.AxisRole.CONTIGUOUS,CpuAccessPlan.AxisRole.CONTIGUOUS),2);
        for (var form : CpuMatmulIr.Realization.values()) for (boolean tanh : List.of(false,true)) for (boolean segment : List.of(false,true)) {
            boolean vector = form == CpuMatmulIr.Realization.DIRECT_N_VECTOR || form == CpuMatmulIr.Realization.TILED_N_VECTOR_2X2;
            int bits = vector ? (type == DataType.FLOAT32 ? jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize() : jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.vectorBitSize()) : 0;
            if (!baseline && System.getenv("CPU0072_ARTIFACT_DIR") == null && !qualifiesFrozenSpecies(form,bits)) {
                String name = "ordinary-"+type+"-"+form+"-"+tanh+"-"+segment;
                vectorCoverageSkipped++;
                System.out.println("CPU0071 VECTOR COVERAGE LIMITATION: "+name+" captured="+CAPTURED_VECTOR_BITS
                        +" bits, host="+bits+" bits; no frozen vector-byte claim on this host; all scalar controls remain checked");
                report.append(name).append("\tNOT_QUALIFIED_SPECIES\t-\tvector-coverage-limitation\n");
                continue;
            }
            var ir = new CpuMatmulIr(type,type,type,form,tanh ? new CpuMatmulIr.Epilogue(CpuMatmulIr.Epilogue.AddInputOrder.NONE,CpuMatmulIr.Epilogue.Terminal.TANH,null) : CpuMatmulIr.Epilogue.none(),bits,CpuMatmulIr.NumericalForm.SEQUENTIAL,List.of(read,read),write);
            var carrier = segment ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT : type == DataType.FLOAT32 ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY : CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
            var spec = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),vector ? CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                    List.of(type,type,type),Collections.nCopies(3,carrier),bits,-1,List.of(),false,54,Optional.of(ir));
            capture(root,target,baseline,"ordinary-"+type+"-"+form+"-"+tanh+"-"+segment,spec,ir.encodedKernelIr(),report);
        }
    }
}
