package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuTypedEmptyScanConsumerTest.Consumer;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Actual Class-File loop execution, not a Java scan fallback. */
public class CpuGeneralScanGeneratedKernelTest {
    /**
     * Rejects actual obsolete MATMUL-bearing and length-Scan ClassFiles before execution.
     * Isolated baseline loading keeps shared Model/query types identical to the current consumer.
     * @throws Exception if the explicitly supplied old-source corpus cannot be loaded or a stale
     *     executable is accepted
     */
    @Test @org.junit.jupiter.api.condition.EnabledIfSystemProperty(named="synaptik.cpu.generalScan.evidenceRoot",matches=".+")
    void obsoleteSourceMatmulAndLengthScanClassFilesRejectBeforeExecution() throws Exception {
        var root = java.nio.file.Path.of(System.getProperty("synaptik.cpu.generalScan.evidenceRoot"));
        var directory = root.resolve("baseline-classes");
        assertTrue(java.nio.file.Files.isDirectory(directory),"an explicit evidence root requires a complete baseline, never a skipped comparison");
        try (var baseline = new java.net.URLClassLoader(new java.net.URL[]{directory.toUri().toURL()},getClass().getClassLoader()) {
            /**
             * Isolates only CPU production classes; shared query/Model types retain parent identity.
             * @param name requested binary name
             * @param resolve whether JVM linkage is requested
             * @return the baseline CPU class or an unchanged shared parent class
             * @throws ClassNotFoundException if neither source supplies the name
             */
            @Override protected Class<?> loadClass(String name,boolean resolve) throws ClassNotFoundException {
                synchronized (getClassLoadingLock(name)) {
                    Class<?> result = findLoadedClass(name);
                    if (result == null && name.startsWith("io.github.pho001.synaptik.backend.cpu.")) {
                        try { result = findClass(name); } catch (ClassNotFoundException absent) { /* shared tests use parent */ }
                    }
                    if (result == null) result = super.loadClass(name,false);
                    if (resolve) resolveClass(result); return result;
                }
            }
        }) {
            String prefix = "io.github.pho001.synaptik.backend.cpu.internal.";
            var oldLowering = baseline.loadClass(prefix+"lowering.CpuGeneralScanLowering");
            var oldFingerprint = baseline.loadClass(prefix+"cache.CpuLoweringFingerprint");
            var oldStrategy = baseline.loadClass(prefix+"prepare.CpuPartitionPreparationPlan$ExecutionStrategy").getField("SCALAR").get(null);
            var oldCarrier = baseline.loadClass(prefix+"cache.CpuKernelSpecialization$CarrierAccess").getField("MEMORY_SEGMENT").get(null);
            var oldSpecClass = baseline.loadClass(prefix+"cache.CpuKernelSpecialization");
            var constructor = Arrays.stream(oldSpecClass.getConstructors()).filter(c -> c.getParameterCount() == 11).findFirst().orElseThrow();
            var generatorClass = baseline.loadClass(prefix+"codegen.emit.CpuClassFileKernelGenerator"); var generator = generatorClass.getConstructor().newInstance();
            var cases = new ArrayList<OperationCapabilityQuery>();
            for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (int family = 0; family < 3; family++) {
                cases.add(CpuGeneralScanLoweringTest.denseReadFixture(type,family,3,false));
            }
            var report = new StringBuilder(); int index = 0;
            for (var q : cases) {
                var current = CpuGeneralScanLowering.analyze(q); var currentSpec = specialization(current);
                Object oldFacts = oldLowering.getMethod("analyze",OperationCapabilityQuery.class).invoke(null,q);
                Object oldIr = oldFacts.getClass().getMethod("ir").invoke(oldFacts);
                Object encoded = oldIr.getClass().getMethod("encodedKernelIr").invoke(oldIr);
                String key = (String)oldIr.getClass().getMethod("structuralKey").invoke(oldIr);
                int boundaries = ((List<?>)oldIr.getClass().getMethod("boundaryAccesses").invoke(oldIr)).size();
                boolean scratch = (int)oldIr.getClass().getMethod("scratchBanks").invoke(oldIr) > 0;
                Object fp = oldFingerprint.getMethod("fromHex",String.class).invoke(null,key);
                Object oldSpec = constructor.newInstance(fp,oldStrategy,Collections.nCopies(boundaries,current.ir().dataType()),Collections.nCopies(boundaries,oldCarrier),
                        0,-1,List.of(),scratch,71,Optional.empty(),Optional.of(oldIr));
                byte[] bytes = (byte[])generatorClass.getMethod("generateClassBytes",oldSpecClass,encoded.getClass()).invoke(generator,oldSpec,encoded);
                if (current.ir().stages().map(s -> s.order().stream().anyMatch(stage -> stage.right() >= 0)).orElse(false)) {
                    assertThrows(IllegalArgumentException.class,() -> new CpuClassFileKernelGenerator()
                            .defineClassBytes(currentSpec,current.ir().encodedKernelIr(),bytes));
                }
                report.append(index++).append(' ').append(current.ir().dataType())
                        .append(" rejected-class=").append(sha256(bytes)).append('\n');
            }
            java.nio.file.Files.writeString(root.resolve("retained-snapshot-compatibility.txt"),report);
            // Actual old executable bytes, not just rewritten envelope text, must reject at v7.
            for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
                var q = CpuGeneralScanLoweringTest.symbolicLengthQuery(type,4,false);
                var current = CpuGeneralScanLowering.analyze(q); var currentSpec = specialization(current);
                Object oldFacts = oldLowering.getMethod("analyze",OperationCapabilityQuery.class).invoke(null,q);
                Object oldIr = oldFacts.getClass().getMethod("ir").invoke(oldFacts);
                Object encoded = oldIr.getClass().getMethod("encodedKernelIr").invoke(oldIr);
                Object fp = oldFingerprint.getMethod("fromHex",String.class).invoke(null,current.ir().structuralKey());
                int boundaries = current.ir().boundaryAccesses().size();
                Object oldSpec = constructor.newInstance(fp,oldStrategy,current.ir().boundaryTypes(),Collections.nCopies(boundaries,oldCarrier),
                        0,-1,List.of(),current.ir().scratchBanks()>0,71,Optional.empty(),Optional.of(oldIr));
                byte[] bytes = (byte[])generatorClass.getMethod("generateClassBytes",oldSpecClass,encoded.getClass()).invoke(generator,oldSpec,encoded);
                assertThrows(IllegalArgumentException.class,() -> new CpuClassFileKernelGenerator().defineClassBytes(currentSpec, current.ir().encodedKernelIr(), bytes));
                java.nio.file.Files.write(root.resolve("retained-v6-"+type+".class"),bytes);
            }
        }
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthOnlyIdentityBindsRolesTerminalControlAndBakedGeometryInBothStreams(DataType type) {
        var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.lengthFixture(type,1,5,3,2,false,true));
        var ir = facts.ir(); var s = ir.stages().orElseThrow(); var original = specialization(facts);
        assertTrue(ir.encodedKernelIr().familyIdentity().contains(":length-aware-v1"));
        assertTrue(new String(original.compatibilityBytes(),java.nio.charset.StandardCharsets.US_ASCII).contains("|scan-length-aware-v1="));
        var store = new CpuGeneratedKernelArtifactStore(); var artifact = store.loadOrGenerate(original,ir.encodedKernelIr());
        for (int mutation = 0; mutation < 4; mutation++) {
            var rows = new ArrayList<>(s.rowAxes()); var terminal = new ArrayList<>(s.terminal());
            if (mutation == 0) rows.set(rows.size()-1,-1);
            if (mutation == 1) terminal.set(terminal.size()-1,true);
            var staged = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Stages(s.views(),s.order(),s.cuts(),s.carryOffsets(),s.carryPorts(),
                    s.scratchBytes(),s.carryOffsetFactors(),s.scratchBytesPerBatch(),false,s.sequenceStrides(),rows,terminal,s.rowCarryCount(),s.lengthRead(),s.batch()+(mutation == 2 ? 1 : 0));
            var actuals = new ArrayList<>(ir.actualReads()); if (mutation == 3) Collections.swap(actuals,0,1);
            var changed = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr(ir.arithmetic(),ir.formalOrdinals(),actuals,ir.scalarReads(),ir.nextOrdinals(),
                    ir.portOrdinals(),ir.demandedPorts(),ir.carries(),ir.boundaryAccesses(),ir.carryCount(),ir.sequenceCount(),ir.reverse(),ir.scratchBanks(),Optional.of(staged));
            var changedFacts = new CpuGeneralScanLowering.Facts(changed,facts.geometry(),facts.spans(),facts.accessBindings());
            var spec = specialization(changedFacts);
            assertFalse(Arrays.equals(original.compatibilityBytes(),spec.compatibilityBytes()));
            assertFalse(Arrays.equals(original.classIdentityBytes(),spec.classIdentityBytes()));
            if (mutation == 2) assertEquals(ir.structuralKey(),changed.structuralKey(),"canonical identity remains geometry-free");
            assertNotEquals(original.structuralKey(),spec.structuralKey());
        }
        var legacy = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.denseReadFixture(type,1,5,false));
        assertThrows(IllegalArgumentException.class,() -> store.loadOrGenerate(original,legacy.ir().encodedKernelIr()));
        assertSame(artifact,store.loadOrGenerate(original,ir.encodedKernelIr()));
        var otherTime = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.lengthFixture(type,1,257,3,2,false,true));
        assertEquals(original,specialization(otherTime)); assertArrayEquals(generated(facts).classBytes(),generated(otherTime).classBytes(),"no T-fold expansion");
        for (int family = 1; family < 3; family++) {
            var old = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.denseReadFixture(type,family,5,false));
            var p = old.ir().stages().orElseThrow();
            String oldProjection = "Stages[views="+p.views()+", order="+p.order()+", cuts="+p.cuts()+", carryOffsets="+p.carryOffsets()+", carryPorts="+p.carryPorts()+", scratchBytes="+p.scratchBytes()+"]";
            assertEquals(oldProjection,p.toString(),"appended length fields do not leak into legacy static v4 payload");
            assertEquals((family == 1 ? "|scan-shaped-v4=" : "|scan-matmul-v5=")+oldProjection,old.ir().bakedSpecializationIdentity());
            var copy = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Stages(p.views(),p.order(),p.cuts(),p.carryOffsets(),p.carryPorts(),p.scratchBytes());
            assertEquals(p,copy); assertEquals(p.toString(),copy.toString());
        }
    }

    /**
     * Captures actual first-order siblings for a fixed F8 broadcast recurrence and joint seeds.
     * @param type homogeneous arithmetic dtype
     * @return independently qualified forward and actual Compiler reverse queries
     */
    public static List<OperationCapabilityQuery> lengthAutomaticQueries(DataType type) {
        Tensor h = CpuGeneralScanLoweringTest.tensor(type,true,3,8), x = CpuGeneralScanLoweringTest.tensor(type,true,3,8);
        Tensor w = CpuGeneralScanLoweringTest.tensor(type,true,8), next = h.add(x.mul(w));
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next));
        Tensor initial = CpuGeneralScanLoweringTest.tensor(type,true,3,8), sequence = CpuGeneralScanLoweringTest.tensor(type,true,5,3,8);
        Tensor weight = CpuGeneralScanLoweringTest.tensor(type,true,8), lengths = CpuGeneralScanLoweringTest.tensor(DataType.INT64,false,3);
        var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight),lengths,1,List.of(OptionalInt.of(0)),body,false);
        var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
        var artifact = CpuGeneralScanCapabilityTest.compile(roots,Optional.of(CpuGeneralScanCapabilityTest.request(roots,List.of(initial,sequence,weight))),false,new CpuCapabilityProvider());
        return CpuGeneralScanLoweringTest.queries(artifact);
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void lengthActualSiblingsMatchCleanJavaForInactiveAndActivePrefixes(DataType type) throws Throwable {
        Tensor h = CpuGeneralScanLoweringTest.tensor(type,true,3,8), x = CpuGeneralScanLoweringTest.tensor(type,true,3,8);
        Tensor w = CpuGeneralScanLoweringTest.tensor(type,true,8), next = h.add(x.mul(w));
        Tensor initial = CpuGeneralScanLoweringTest.tensor(type,true,3,8), sequence = CpuGeneralScanLoweringTest.tensor(type,true,5,3,8);
        Tensor weight = CpuGeneralScanLoweringTest.tensor(type,true,8), lengths = CpuGeneralScanLoweringTest.tensor(DataType.INT64,false,3);
        Tensor finalSeed = CpuGeneralScanLoweringTest.tensor(type,false,3,8), emissionSeed = CpuGeneralScanLoweringTest.tensor(type,false,5,3,8);
        var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight),lengths,1,List.of(OptionalInt.of(0)),
                new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next)),false);
        var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
        var request = new io.github.pho001.synaptik.compiler.FunctionalGradientRequest(List.of(
                new io.github.pho001.synaptik.compiler.FunctionalGradientRequest.Stage(roots.stream().map(r ->
                        (io.github.pho001.synaptik.compiler.FunctionalGradientRequest.OutputReference)
                                new io.github.pho001.synaptik.compiler.FunctionalGradientRequest.ForwardTensorReference(r)).toList(),
                        List.of(Optional.of(finalSeed),Optional.of(emissionSeed)),List.of(initial,sequence,weight),false,
                        io.github.pho001.synaptik.compiler.FunctionalGradientRequest.DisconnectedPolicy.ZERO)));
        var artifact = CpuGeneralScanCapabilityTest.compile(roots,Optional.of(request),false,new CpuCapabilityProvider());
        try (var consumer = new Consumer(artifact,OptionalLong.empty()); var arena = Arena.ofConfined()) {
            var tensors = List.of(initial,sequence,weight,finalSeed,emissionSeed);
            var inputs = new ArrayList<MemorySegment>();
            for (Tensor tensor : tensors) {
                int cells = Math.toIntExact(tensor.descriptor().shape().dimensions().stream().mapToLong(d -> d.staticSize().orElseThrow()).reduce(1,Math::multiplyExact));
                var segment = arena.allocate((long)cells*type.byteWidth(),8); inputs.add(segment);
                for (int i = 0; i < cells; i++) writeCell(type,segment,i,.125+(i%7)*.015625);
            }
            var ls = arena.allocate(24,8);
            var expected = new ArrayList<MemorySegment>();
            for (int cells : new int[]{24,120,24,8,120}) expected.add(arena.allocate((long)cells*type.byteWidth(),8));
            consumer.bind();
            for (long[] activity : List.of(new long[]{0,0,0},new long[]{5,3,0})) {
                for (int t = 0; t < 5; t++) for (int b = 0; b < 3; b++) for (int j = 0; j < 8; j++) {
                    int cell = t*24+b*8+j;
                    writeCell(type,inputs.get(1),cell,t >= activity[b] ? Double.NaN : .125+(cell%7)*.015625);
                    writeCell(type,inputs.get(4),cell,t >= activity[b] ? Double.NaN : .125+(cell%7)*.015625);
                }
                for (int p = 0; p < tensors.size(); p++) {
                    double[] values = new double[Math.toIntExact(inputs.get(p).byteSize()/type.byteWidth())];
                    for (int i = 0; i < values.length; i++) values[i] = readCell(type,inputs.get(p),i);
                    consumer.numbers(tensors.get(p),values);
                }
                consumer.raw(lengths,activity); ls.copyFrom(MemorySegment.ofArray(activity)); consumer.execute();
                lengthOracle(type,false).invokeWithArguments(inputs.get(0),inputs.get(1),inputs.get(2),ls,
                        expected.get(0),expected.get(1),new long[]{5,24},0L,24L);
                var zero = arena.allocate(8L*type.byteWidth(),8);
                lengthOracle(type,true).invokeWithArguments(inputs.get(3),zero,inputs.get(1),inputs.get(4),inputs.get(2),ls,
                        expected.get(2),expected.get(3),expected.get(4),new long[]{5,24},0L,24L);
                var actual = List.of(consumer.forward(0),consumer.forward(1),consumer.gradient(0),consumer.gradient(2),consumer.gradient(1));
                for (int p = 0; p < actual.size(); p++) for (int i = 0; i < actual.get(p).length; i++)
                    assertEquals(readCell(type,expected.get(p),i),actual.get(p)[i],"clean-Java exact output="+p+" cell="+i);
            }
        }
    }

    /**
     * Resolves the specialized optimal reference once outside any timed or semantic hot loop.
     * @param type FLOAT32 or FLOAT64 represented type
     * @param reverse whether the actual first-order sibling is requested
     * @return direct static handle with exactly the corresponding generated segment/geometry ABI
     * @throws ReflectiveOperationException if the fixed test reference is missing
     */
    public static java.lang.invoke.MethodHandle lengthOracle(DataType type,boolean reverse) throws ReflectiveOperationException {
        var parameters = new ArrayList<Class<?>>(Collections.nCopies(reverse ? 9 : 6,MemorySegment.class));
        parameters.add(long[].class); parameters.add(long.class); parameters.add(long.class);
        return java.lang.invoke.MethodHandles.lookup().findStatic(CpuGeneralScanCleanJavaOracle.class,
                "length"+(reverse ? "Reverse" : "Forward")+(type == DataType.FLOAT64 ? "Float64" : "Float32")+"B3F8",
                java.lang.invoke.MethodType.methodType(void.class,parameters));
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void denseActualReadTwinsReuseCanonicalAndBothBakedIdentitiesAndExactClassBytes(DataType type) {
        for (int family = 0; family < 3; family++) {
            var queries = new ArrayList<OperationCapabilityQuery>();
            queries.add(CpuGeneralScanLoweringTest.denseReadFixture(type,family,3,false));
            for (var q : queries) {
                var canonical = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.withActualReadViews(q,false));
                var view = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.withActualReadViews(q,true));
                assertEquals(canonical,view,"complete proof including scalar missing seeds, geometry and exact access ranges");
                assertEquals(canonical.ir().encodedKernelIr(),view.ir().encodedKernelIr());
                assertEquals(canonical.ir().structuralKey(),view.ir().structuralKey());
                assertEquals(canonical.ir().bakedSpecializationIdentity(),view.ir().bakedSpecializationIdentity());
                var a = specialization(canonical); var b = specialization(view);
                assertEquals(a,b); assertEquals(a.structuralKey(),b.structuralKey());
                assertArrayEquals(a.compatibilityBytes(),b.compatibilityBytes());
                assertArrayEquals(a.classIdentityBytes(),b.classIdentityBytes());
                assertArrayEquals(generated(canonical).classBytes(),generated(view).classBytes());
                assertEquals(71,a.classIdentitySchema()); assertEquals(71,CpuGeneratorSchema.CURRENT_VERSION);
                if (canonical.ir().stages().isPresent()) assertTrue(canonical.ir().bakedSpecializationIdentity().contains(
                        canonical.ir().stages().orElseThrow().order().stream().anyMatch(s -> s.right() >= 0) ? "|scan-matmul-v5=" : "|scan-shaped-v4="));
                var store = new CpuGeneratedKernelArtifactStore();
                var cached = store.loadOrGenerate(a,canonical.ir().encodedKernelIr());
                assertSame(cached,store.loadOrGenerate(b,view.ir().encodedKernelIr()),"same artifact is reused, not merely equivalent algorithms");
            }
        }
    }

    @Test void specializedB4H8OracleMatchesCompleteRuntimeLongTimeWorkAndRawEmptyPublication() throws Throwable {
        byte[] oracleBytes;
        try (var stream = CpuGeneralScanCleanJavaOracle.class.getResourceAsStream("CpuGeneralScanCleanJavaOracle.class")) {
            assertNotNull(stream); oracleBytes = stream.readAllBytes();
        }
        var oracle = ClassFile.of().parse(oracleBytes).methods().stream()
                .filter(m -> m.methodName().equalsString("broadcastForwardFloat64B4H8")).findFirst().orElseThrow();
        var oracleOps = oracle.code().orElseThrow().elementStream().filter(java.lang.classfile.Instruction.class::isInstance)
                .map(java.lang.classfile.Instruction.class::cast).map(java.lang.classfile.Instruction::opcode).toList();
        assertTrue(oracleOps.contains(java.lang.classfile.Opcode.LALOAD));
        assertFalse(oracleOps.contains(java.lang.classfile.Opcode.L2I));
        assertEquals(1,Collections.frequency(oracleOps,java.lang.classfile.Opcode.DMUL));
        assertEquals(1,Collections.frequency(oracleOps,java.lang.classfile.Opcode.DADD));
        assertEquals(2,Collections.frequency(oracleOps,java.lang.classfile.Opcode.IINC));
        for (int t : List.of(0,1,3,8)) {
            var facts = broadcastFacts(DataType.FLOAT64,t,4,8,false,true); var generated = generated(facts);
            assertEquals(oracle.methodTypeSymbol().descriptorString(),generated.entryPoint().type().descriptorString());
            assertEquals(specialization(broadcastFacts(DataType.FLOAT64,257,4,8,false,true)),generated.specialization());
            recordClass("specialized-B4H8-T"+t,generated.specialization(),generated.classBytes());
            var ops = opcodes(generated.classBytes());
            assertEquals(2,Collections.frequency(ops,java.lang.classfile.Opcode.IINC));
            assertFalse(ops.contains(java.lang.classfile.Opcode.L2I));
            assertTrue(ops.contains(java.lang.classfile.Opcode.LALOAD),"time is read as long at entry");
            assertEquals(1,Collections.frequency(ops,java.lang.classfile.Opcode.DMUL));
            assertEquals(1,Collections.frequency(ops,java.lang.classfile.Opcode.DADD));
            assertEquals(1,Collections.frequency(ops,java.lang.classfile.Opcode.LCONST_1),"only long time increment, no constant-one factor stores");
            var entryInstructions = ClassFile.of().parse(generated.classBytes()).methods().getFirst().code().orElseThrow()
                    .elementStream().filter(java.lang.classfile.Instruction.class::isInstance).map(java.lang.classfile.Instruction.class::cast).toList();
            var loaded = new HashSet<Integer>();
            for (var instruction : entryInstructions) if (instruction instanceof java.lang.classfile.instruction.LoadInstruction load) loaded.add(load.slot());
            for (var instruction : entryInstructions) if (instruction instanceof java.lang.classfile.instruction.StoreInstruction store)
                assertTrue(loaded.contains(store.slot()),"no dead factor/zero/alias store: local="+store.slot());
            long longCopies = 0;
            for (int i = 0; i+1 < entryInstructions.size(); i++)
                if (entryInstructions.get(i) instanceof java.lang.classfile.instruction.LoadInstruction load && load.typeKind() == java.lang.classfile.TypeKind.LONG
                        && entryInstructions.get(i+1) instanceof java.lang.classfile.instruction.StoreInstruction store && store.typeKind() == java.lang.classfile.TypeKind.LONG) longCopies++;
            assertEquals(1,longCopies,"only necessary mutable old-carry base copy; no forward time or frame aliases");
            try (var arena = Arena.ofConfined()) {
                var initial = arena.allocate(256,8); var sequence = arena.allocate(t*256L,8); var weight = arena.allocate(64,8);
                var result = arena.allocate(256,8); var emissions = arena.allocate(t*256L,8);
                var expected = arena.allocate(256,8); var history = arena.allocate(t*256L,8);
                for (int i = 0; i < 32; i++) initial.setAtIndex(ValueLayout.JAVA_LONG,i,t == 0
                        ? new long[] {Long.MIN_VALUE,1,0x7ff8123456789abcL,0xfff8123456789abcL}[i%4]
                        : Double.doubleToRawLongBits((i-7)*.03125+0x1p-40));
                for (int i = 0; i < t*32; i++) sequence.setAtIndex(ValueLayout.JAVA_DOUBLE,i,(i%19-9)*.015625+0x1p-40);
                for (int i = 0; i < 8; i++) weight.setAtIndex(ValueLayout.JAVA_DOUBLE,i,(i+1)*.0625+0x1p-40);
                long[] geometry = {t,32};
                generated.entryPoint().invokeExact(initial,sequence,weight,result,emissions,geometry,0L,32L);
                CpuGeneralScanCleanJavaOracle.broadcastForwardFloat64B4H8(initial,sequence,weight,expected,history,geometry,0L,32L);
                assertEquals(-1L,result.mismatch(expected)); assertEquals(-1L,emissions.mismatch(history));
                assertArrayEquals(new long[] {t,32},geometry,"borrowed geometry is not mutated");
                if (t == 0) assertEquals(-1L,initial.mismatch(result),"all initial payload bits survive raw final copy");
                else {
                    CpuGeneralScanCleanJavaOracle.broadcastForwardFloat64(initial,sequence,weight,expected,history,t,4,8);
                    assertEquals(-1L,result.mismatch(expected)); assertEquals(-1L,emissions.mismatch(history));
                }
            }
        }
        String root = System.getProperty("synaptik.cpu.generalScan.evidenceRoot");
        if (root != null) {
            var path = java.nio.file.Path.of(root,"structure","CpuGeneralScanCleanJavaOracle.class");
            java.nio.file.Files.write(path,oracleBytes);
            var process = new ProcessBuilder(java.nio.file.Path.of(System.getProperty("java.home"),"bin","javap").toString(),"-c","-v","-p",path.toString()).redirectErrorStream(true).start();
            java.nio.file.Files.writeString(path.resolveSibling("CpuGeneralScanCleanJavaOracle.javap.txt"),new String(process.getInputStream().readAllBytes()));
            assertEquals(0,process.waitFor());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void fixedScalarAndScalarToDenseMatmulFactorsRemainSoundAcrossEmptyAndLiveDomains(DataType type) throws Throwable {
        for (boolean reverse : List.of(false,true)) for (int t : List.of(0,1,3))
            for (int extent : List.of(0,1,2)) for (boolean emission : List.of(false,true)) for (boolean carryProduct : List.of(false,true)) {
                Tensor h = CpuGeneralScanLoweringTest.tensor(type,false,extent,extent), x = CpuGeneralScanLoweringTest.tensor(type,false,extent,extent);
                Tensor w = CpuGeneralScanLoweringTest.tensor(type,false,extent,extent), next = carryProduct ? h.matmul(w).add(x) : h.add(x.matmul(w));
                var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),emission ? List.of(next,h,h.sum()) : List.of(h.sum()));
                Tensor initial = CpuGeneralScanLoweringTest.tensor(type,false).expand(Shape.of(extent,extent));
                Tensor sequence = CpuGeneralScanLoweringTest.tensor(type,false).expand(Shape.of(t,extent,extent));
                Tensor weight = CpuGeneralScanLoweringTest.tensor(type,false).expand(Shape.of(extent,extent));
                var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight),body,reverse);
                var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
                var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(roots)); var ir = facts.ir();
                var generated = generated(facts);
                assertTrue(ClassFile.of().verify(generated.classBytes()).isEmpty());
                try (var arena = Arena.ofConfined()) {
                    var args = new ArrayList<MemorySegment>();
                    for (long span : facts.spans()) args.add(arena.allocate(span*type.byteWidth(),type.byteWidth()));
                    if (facts.geometry().scratchBytes() > 0) args.add(arena.allocate(facts.geometry().scratchBytes(),type.byteWidth()));
                    for (int f = 0; f < 3; f++) if (args.get(ir.actualReads().get(f)).byteSize() > 0)
                        writeCell(type,args.get(ir.actualReads().get(f)),0,f == 0 ? .25 : f == 1 ? .125 : .5);
                    double[] old = new double[extent*extent]; Arrays.fill(old,.25);
                    double[] nextHistory = new double[t*old.length], previousHistory = new double[t*old.length], sums = new double[t];
                    for (int step = 0; step < t; step++) {
                        int time = reverse ? t-1-step : step;
                        System.arraycopy(old,0,previousHistory,time*old.length,old.length);
                        double sum = 0; for (double cell : old) sum = represented(type,sum+cell); sums[time] = sum;
                        double[] updated = new double[old.length];
                        for (int row = 0; row < extent; row++) for (int col = 0; col < extent; col++) {
                            double product = 0;
                            for (int k = 0; k < extent; k++) product = represented(type,product+represented(type,(carryProduct ? old[row*extent+k] : .125)*.5));
                            updated[row*extent+col] = represented(type,product+(carryProduct ? .125 : old[row*extent+col]));
                        }
                        old = updated; System.arraycopy(old,0,nextHistory,time*old.length,old.length);
                    }
                    generated.entryPoint().asSpreader(0,MemorySegment[].class,args.size())
                            .invokeExact(args.toArray(MemorySegment[]::new),new long[] {t,extent*extent},0L,(long)extent*extent);
                    for (int p : ir.demandedPorts()) {
                        double[] expected = p == 0 ? old : emission ? p == 1 ? nextHistory : p == 2 ? previousHistory : sums : sums;
                        var actual = args.get(ir.scalarReads().size()+ir.demandedPorts().indexOf(p));
                        assertEquals(expected.length,actual.byteSize()/type.byteWidth());
                        for (int i = 0; i < expected.length; i++) assertEquals(expected[i],readCell(type,actual,i),"port="+p+" T="+t+" reverse="+reverse);
                    }
                }
            }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void coordinateEndpointsUseIntOnlyWhenLastIncrementIsSafeAndWidenBeforeAddressMath(DataType type) throws Exception {
        assertThrows(IllegalArgumentException.class,() -> CpuGeneralScanEmitter.Shaped.coordinateKind(-1));
        for (long extent : List.of(0L,1L,(long)Integer.MAX_VALUE,Integer.MAX_VALUE+1L)) {
            boolean small = extent <= Integer.MAX_VALUE;
            assertEquals(small ? java.lang.classfile.TypeKind.INT : java.lang.classfile.TypeKind.LONG,
                    CpuGeneralScanEmitter.Shaped.coordinateKind(extent));
            if (extent > 0) {
                if (small) assertEquals(extent,Math.addExact(Math.toIntExact(extent-1),1));
                else assertEquals(extent,Math.addExact(extent-1,1L));
            }
            var facts = broadcastFacts(type,3,extent,2,false,true);
            var generated = generated(facts); byte[] bytes = generated.classBytes();
            assertTrue(ClassFile.of().verify(bytes).isEmpty(),"verifier catches mismatched local kinds");
            var ops = opcodes(bytes);
            assertEquals(extent == 0 ? 0 : extent == 1 ? 1 : small ? 2 : 1,
                    Collections.frequency(ops,java.lang.classfile.Opcode.IINC),"one int increment per non-singleton axis");
            if (extent > 0) assertTrue(ops.contains(java.lang.classfile.Opcode.I2L),"coordinates widen before long strides");
            assertFalse(ops.contains(java.lang.classfile.Opcode.IMUL),"no 32-bit stride/address products");
            if (extent == 0) assertFalse(ops.contains(type == DataType.FLOAT64 ? java.lang.classfile.Opcode.DMUL : java.lang.classfile.Opcode.FMUL),"empty constant guard omits body");
            if (!small) assertTrue(ops.contains(java.lang.classfile.Opcode.IFGE),"legal large axis keeps long compare fallback");
            assertTrue(facts.ir().stages().orElseThrow().scratchBytes() == 0);
            recordClass("endpoint-"+type+"-"+extent,specialization(facts),bytes);
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void longCoordinateFallbackVerifiesInReductionBindingsRestoredAxesAndMatmulCells(DataType type) throws Exception {
        long large = Integer.MAX_VALUE+1L;
        Tensor h = CpuGeneralScanLoweringTest.tensor(type,false,1,2), x = CpuGeneralScanLoweringTest.tensor(type,false,large,2);
        Tensor next = h.add(x.neg().sum(0,true));
        var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(next));
        var scan = Scan.apply(List.of(CpuGeneralScanLoweringTest.tensor(type,false,1,2)),
                List.of(CpuGeneralScanLoweringTest.tensor(type,false,3,large,2)),List.of(),body,false);
        var reduction = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst())));
        assertFalse(reduction.ir().stages().orElseThrow().order().getFirst().epilogue().isEmpty(),"restored singleton epilogue axes");
        recordClass("large-reduction-"+type,specialization(reduction),generated(reduction).classBytes());
        for (boolean rows : List.of(false,true)) {
            long m = rows ? large : 2, n = rows ? 2 : large;
            Tensor mh = CpuGeneralScanLoweringTest.tensor(type,false,m,n), mx = CpuGeneralScanLoweringTest.tensor(type,false,m,1);
            Tensor mw = CpuGeneralScanLoweringTest.tensor(type,false,1,n), mn = mh.add(mx.matmul(mw));
            var mb = new ScanBody(List.of(mh),List.of(mx),List.of(mw),List.of(mn),List.of(mn));
            var ms = Scan.apply(List.of(CpuGeneralScanLoweringTest.tensor(type,false,m,n)),
                    List.of(CpuGeneralScanLoweringTest.tensor(type,false,3,m,1)),List.of(CpuGeneralScanLoweringTest.tensor(type,false,1,n)),mb,false);
            var matrix = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(List.of(ms.finalCarry().getFirst(),ms.outputs().getFirst())));
            recordClass("large-matmul-"+type+"-"+rows,specialization(matrix),generated(matrix).classBytes());
        }
        // Symbolic BatchBinding admits int extents; this retained static case deliberately does
        // not, and must keep long K rather than narrowing a legal larger contraction.
        Tensor kh=CpuGeneralScanLoweringTest.tensor(type,false,2,2), kx=CpuGeneralScanLoweringTest.tensor(type,false,2,large);
        Tensor kw=CpuGeneralScanLoweringTest.tensor(type,false,large,2), kn=kh.add(kx.matmul(kw));
        var kb=new ScanBody(List.of(kh),List.of(kx),List.of(kw),List.of(kn),List.of(kn));
        var ks=Scan.apply(List.of(CpuGeneralScanLoweringTest.tensor(type,false,2,2)),List.of(CpuGeneralScanLoweringTest.tensor(type,false,3,2,large)),
                List.of(CpuGeneralScanLoweringTest.tensor(type,false,large,2)),kb,false);
        var contraction=CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(List.of(ks.finalCarry().getFirst(),ks.outputs().getFirst())));
        byte[] bytes=generated(contraction).classBytes(); var ops=opcodes(bytes);
        assertEquals(2,Collections.frequency(ops,java.lang.classfile.Opcode.IINC),"M/N only; unproved/static K remains long");
        assertFalse(ops.contains(java.lang.classfile.Opcode.L2I));
        recordClass("large-contraction-"+type,specialization(contraction),bytes);
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void exactEmissionCarriesRemoveFirstIterationBranchWhileBanksAndForwardedViewsKeepFallback(DataType type) throws Exception {
        for (boolean reverse : List.of(false,true)) for (int mode = 0; mode < 3; mode++) {
            Tensor h = CpuGeneralScanLoweringTest.tensor(type,false,2,2), x = CpuGeneralScanLoweringTest.tensor(type,false,2,2);
            Tensor w = CpuGeneralScanLoweringTest.tensor(type,false,2);
            Tensor next = mode == 2 ? h.permute(1,0) : h.add(x.mul(w));
            var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),mode == 1 ? List.of() : List.of(next));
            var scan = Scan.apply(List.of(CpuGeneralScanLoweringTest.tensor(type,false,2,2)),
                    List.of(CpuGeneralScanLoweringTest.tensor(type,false,3,2,2)),List.of(CpuGeneralScanLoweringTest.tensor(type,false,2)),body,reverse);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(roots));
            byte[] bytes = generated(facts).classBytes();
            assertEquals(mode == 0 ? 1 : 2,Collections.frequency(opcodes(bytes),java.lang.classfile.Opcode.IFEQ),
                    "only T=0 guard on proved emission path; otherwise first-step fallback remains");
            recordClass("carry-"+type+"-"+reverse+"-"+mode,specialization(facts),bytes);
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void tiedEmissionCarriesSavePreviousStateBeforePostStepBindingAndExpandScalarInitializers(DataType type) throws Throwable {
        for (boolean reverse : List.of(false,true)) for (int t : List.of(0,1,3)) {
            Tensor h = CpuGeneralScanLoweringTest.tensor(type,false,2,2), k = CpuGeneralScanLoweringTest.tensor(type,false,2,2);
            Tensor x = CpuGeneralScanLoweringTest.tensor(type,false,2,2), w = CpuGeneralScanLoweringTest.tensor(type,false,2);
            Tensor product = x.mul(w), sum = h.add(product), next = sum.add(k);
            var body = new ScanBody(List.of(h,k),List.of(x),List.of(w),List.of(next,next),List.of(next,next));
            Tensor initial = CpuGeneralScanLoweringTest.tensor(type,false).expand(Shape.of(2,2));
            var scan = Scan.apply(List.of(initial,CpuGeneralScanLoweringTest.tensor(type,false,2,2)),
                    List.of(CpuGeneralScanLoweringTest.tensor(type,false,t,2,2)),List.of(CpuGeneralScanLoweringTest.tensor(type,false,2)),body,reverse);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs()); roots.addAll(scan.residuals());
            var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(roots)); var ir = facts.ir();
            var generated = generated(facts);
            assertEquals(0,facts.geometry().scratchBytes(),"both tied roles reuse the exact producing frame");
            assertEquals(1,Collections.frequency(opcodes(generated.classBytes()),java.lang.classfile.Opcode.IFEQ),"no first-iteration carry branches");
            try (var arena = Arena.ofConfined()) {
                MemorySegment[] segments = new MemorySegment[facts.spans().size()];
                for (int p = 0; p < segments.length; p++) segments[p] = arena.allocate(facts.spans().get(p)*type.byteWidth(),type.byteWidth());
                MemorySegment ih = segments[ir.actualReads().get(0)], ik = segments[ir.actualReads().get(1)];
                MemorySegment xs = segments[ir.actualReads().get(2)], ws = segments[ir.actualReads().get(3)];
                long payload = type == DataType.FLOAT64 ? 0x7ff8123456789abcL : 0x7fc12345L;
                if (type == DataType.FLOAT64) ih.set(ValueLayout.JAVA_LONG,0,t == 0 ? payload : Long.MIN_VALUE);
                else ih.set(ValueLayout.JAVA_INT,0,t == 0 ? (int)payload : Integer.MIN_VALUE);
                double initialH = readCell(type,ih,0);
                double[] oldH = new double[4], oldK = {.25,-0d,type == DataType.FLOAT64 ? Double.MIN_NORMAL/2 : Float.MIN_NORMAL/2,16777216};
                Arrays.fill(oldH,initialH);
                for (int i = 0; i < 4; i++) writeCell(type,ik,i,oldK[i]);
                for (int i = 0; i < t*4; i++) writeCell(type,xs,i,(i-5)*.03125);
                writeCell(type,ws,0,-1); writeCell(type,ws,1,.75);
                var histories = new IdentityHashMap<Tensor,double[]>();
                for (var value : List.of(h,k,product,sum,next)) histories.put(value,new double[t*4]);
                for (int step = 0; step < t; step++) {
                    int time = reverse ? t-1-step : step; double[] updated = new double[4];
                    for (int i = 0; i < 4; i++) {
                        int p = time*4+i;
                        histories.get(h)[p] = oldH[i]; histories.get(k)[p] = oldK[i];
                        double mul = represented(type,readCell(type,xs,p)*readCell(type,ws,i%2));
                        double add = represented(type,oldH[i]+mul); updated[i] = represented(type,add+oldK[i]);
                        histories.get(product)[p] = mul; histories.get(sum)[p] = add; histories.get(next)[p] = updated[i];
                    }
                    oldH = updated; oldK = updated;
                }
                generated.entryPoint().asSpreader(0,MemorySegment[].class,segments.length)
                        .invokeExact(segments,new long[] {t,4},0L,4L);
                for (int p : ir.demandedPorts()) {
                    double[] expected = p == 0 ? oldH : p == 1 ? oldK : p < 4 ? histories.get(next)
                            : histories.get(body.retainedValues().get(p-4));
                    var result = segments[ir.scalarReads().size()+ir.demandedPorts().indexOf(p)];
                    assertEquals(expected.length,result.byteSize()/type.byteWidth());
                    for (int i = 0; i < expected.length; i++) {
                        if (Double.isNaN(expected[i])) assertTrue(Double.isNaN(readCell(type,result,i)));
                        else assertEquals(Double.doubleToRawLongBits(expected[i]),Double.doubleToRawLongBits(readCell(type,result,i)),"exact represented saved/previous state, port="+p);
                    }
                }
                if (t == 0) {
                    var output = segments[ir.scalarReads().size()+ir.demandedPorts().indexOf(0)];
                    for (int i = 0; i < 4; i++) assertEquals(payload,type == DataType.FLOAT64 ? output.getAtIndex(ValueLayout.JAVA_LONG,i)
                            : Integer.toUnsignedLong(output.getAtIndex(ValueLayout.JAVA_INT,i)),"T=0 raw scalar payload");
                }
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void actualCompilerGradientsPreserveEmissionAndParameterContributionsIncludingZeroTime(DataType type) throws Exception {
        for (boolean reverse : List.of(false,true)) for (int t : List.of(0,1,3)) {
            Tensor h = CpuGeneralScanLoweringTest.tensor(type,true,2,2), x = CpuGeneralScanLoweringTest.tensor(type,true,2,2);
            Tensor w = CpuGeneralScanLoweringTest.tensor(type,true,2), next = h.add(x);
            var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next.mul(w)));
            Tensor initial = CpuGeneralScanLoweringTest.tensor(type,true,2,2), sequence = CpuGeneralScanLoweringTest.tensor(type,true,t,2,2);
            Tensor weight = CpuGeneralScanLoweringTest.tensor(type,true,2), hs = CpuGeneralScanLoweringTest.tensor(type,false,2,2);
            Tensor ys = CpuGeneralScanLoweringTest.tensor(type,false,t,2,2);
            var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(weight),body,reverse);
            var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
            var request = new io.github.pho001.synaptik.compiler.FunctionalGradientRequest(List.of(
                    new io.github.pho001.synaptik.compiler.FunctionalGradientRequest.Stage(roots.stream().map(r ->
                            (io.github.pho001.synaptik.compiler.FunctionalGradientRequest.OutputReference)new io.github.pho001.synaptik.compiler.FunctionalGradientRequest.ForwardTensorReference(r)).toList(),
                            List.of(Optional.of(hs),Optional.of(ys)),List.of(initial,sequence,weight),false,
                            t == 0 ? io.github.pho001.synaptik.compiler.FunctionalGradientRequest.DisconnectedPolicy.ZERO
                                    : io.github.pho001.synaptik.compiler.FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
            var artifact = CpuGeneralScanCapabilityTest.compile(roots,Optional.of(request),false,new CpuCapabilityProvider());
            try (var consumer = new Consumer(artifact,OptionalLong.empty())) {
                for (Tensor input : List.of(initial,sequence,weight,hs,ys)) {
                    double value = input == initial ? .125 : input == sequence ? .03125 : input == weight ? .5 : input == hs ? .25 : .125;
                    double[] values = new double[Math.toIntExact(input.descriptor().shape().dimensions().stream().mapToLong(d -> d.staticSize().orElseThrow()).reduce(1,Math::multiplyExact))];
                    Arrays.fill(values,value); consumer.numbers(input,values);
                }
                consumer.bind(); consumer.execute();
                var dh = consumer.gradient(0); var dx = consumer.gradient(1); var dw = consumer.gradient(2);
                for (int i = 0; i < 4; i++) assertEquals(.25+t*.0625,dh[i]);
                for (int step = 0; step < t; step++) for (int i = 0; i < 4; i++)
                    assertEquals(.25+(step+1)*.0625,dx[(reverse ? step : t-1-step)*4+i]);
                double parameter = 0; for (int step = 0; step < t; step++) parameter += 2*.125*(.125+(step+1)*.03125);
                for (int i = 0; i < 2; i++) assertEquals(parameter,dw[i]);
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void forwardedFinalCarryRetainsLastPrestepFormalAfterProducedEmissionCarry(DataType type) throws Throwable {
        for (boolean reverse : List.of(false,true)) for (boolean transpose : List.of(false,true)) for (int t : List.of(0,1,3)) {
            Tensor h = CpuGeneralScanLoweringTest.tensor(type,false,2,2), k = CpuGeneralScanLoweringTest.tensor(type,false,2,2);
            Tensor x = CpuGeneralScanLoweringTest.tensor(type,false,2,2), w = CpuGeneralScanLoweringTest.tensor(type,false,2);
            Tensor produced = h.add(x.mul(w)), forwarded = transpose ? h.permute(1,0) : h;
            var body = new ScanBody(List.of(h,k),List.of(x),List.of(w),List.of(produced,forwarded),List.of(produced));
            var scan = Scan.apply(List.of(CpuGeneralScanLoweringTest.tensor(type,false,2,2),CpuGeneralScanLoweringTest.tensor(type,false,2,2)),
                    List.of(CpuGeneralScanLoweringTest.tensor(type,false,t,2,2)),List.of(CpuGeneralScanLoweringTest.tensor(type,false,2)),body,reverse);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(roots)); var ir = facts.ir();
            var generated = generated(facts);
            try (var arena = Arena.ofConfined()) {
                var args = new ArrayList<MemorySegment>();
                for (long span : facts.spans()) args.add(arena.allocate(span*type.byteWidth(),type.byteWidth()));
                if (facts.geometry().scratchBytes() > 0) args.add(arena.allocate(facts.geometry().scratchBytes(),type.byteWidth()));
                double[] a = {1,2,3,4}, b = {-4,-3,-2,-1}; double[] histories = new double[t*4];
                for (int i = 0; i < 4; i++) { writeCell(type,args.get(ir.actualReads().get(0)),i,a[i]); writeCell(type,args.get(ir.actualReads().get(1)),i,b[i]); }
                for (int time = 0; time < t; time++) for (int i = 0; i < 4; i++) writeCell(type,args.get(ir.actualReads().get(2)),time*4+i,2 << time);
                for (int i = 0; i < 2; i++) writeCell(type,args.get(ir.actualReads().get(3)),i,1);
                for (int step = 0; step < t; step++) {
                    int time = reverse ? t-1-step : step;
                    b = transpose ? new double[] {a[0],a[2],a[1],a[3]} : a.clone();
                    for (int i = 0; i < 4; i++) a[i] += 2 << time;
                    System.arraycopy(a,0,histories,time*4,4);
                }
                generated.entryPoint().asSpreader(0,MemorySegment[].class,args.size())
                        .invokeExact(args.toArray(MemorySegment[]::new),new long[] {t,4},0L,4L);
                for (int p : ir.demandedPorts()) {
                    var result = args.get(ir.scalarReads().size()+ir.demandedPorts().indexOf(p));
                    double[] expected = p == 0 ? a : p == 1 ? b : histories;
                    for (int i = 0; i < expected.length; i++) assertEquals(expected[i],readCell(type,result,i),
                            "forwarded final publication needs last pre-step formal: port="+p+" T="+t+" reverse="+reverse+" transpose="+transpose);
                }
            }
        }
    }

    /**
     * Defines and executes mixed empty/dense/scalar carries with a zero-sized first frame.
     * Final sources include sequence views, transposes, produced emission frames and bank-backed
     * views. Both directions and T=0/1/3 exercise the final-source initialization proof without
     * letting the empty entry range suppress nonempty copies or scalar reduction work.
     * @param type represented FLOAT32 or FLOAT64 dtype for every body value and carrier
     * @throws Throwable if generated definition, invocation, structural checks or evidence output fails
     */
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void finalCopiesOfStepViewsAndBanksDominateTopTestAlongsideEmptyAndScalarCarries(DataType type) throws Throwable {
        for (boolean reverse : List.of(false,true)) for (int t : List.of(0,1,3)) for (int mode = 0; mode < 4; mode++) {
            Tensor empty = CpuGeneralScanLoweringTest.tensor(type,false,0,2), h = CpuGeneralScanLoweringTest.tensor(type,false,2,2);
            Tensor scalar = CpuGeneralScanLoweringTest.tensor(type,false), x = CpuGeneralScanLoweringTest.tensor(type,false,2,2);
            Tensor produced = h.add(x), next = mode == 0 ? x : mode == 1 ? x.permute(1,0) : mode == 2 ? produced : produced.permute(1,0);
            var body = new ScanBody(List.of(empty,h,scalar),List.of(x),List.of(),List.of(empty,next,scalar.add(x.sum())),
                    mode == 2 ? List.of(produced) : List.of());
            var scan = Scan.apply(List.of(CpuGeneralScanLoweringTest.tensor(type,false,0,2),CpuGeneralScanLoweringTest.tensor(type,false,2,2),
                    CpuGeneralScanLoweringTest.tensor(type,false)),List.of(CpuGeneralScanLoweringTest.tensor(type,false,t,2,2)),List.of(),body,reverse);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(roots)); var ir = facts.ir();
            var kernel = generated(facts); recordClass("final-views-"+type+"-"+reverse+"-T"+t+"-"+mode,kernel.specialization(),kernel.classBytes());
            try (var arena = Arena.ofConfined()) {
                var args = new ArrayList<MemorySegment>();
                for (long span : facts.spans()) args.add(arena.allocate(span*type.byteWidth(),type.byteWidth()));
                if (facts.geometry().scratchBytes() > 0) args.add(arena.allocate(facts.geometry().scratchBytes(),type.byteWidth()));
                double[] old = {1,2,3,4}; double sum = .5;
                for (int i = 0; i < 4; i++) writeCell(type,args.get(ir.actualReads().get(1)),i,old[i]);
                writeCell(type,args.get(ir.actualReads().get(2)),0,sum);
                for (int i = 0; i < t*4; i++) writeCell(type,args.get(ir.actualReads().get(3)),i,(i+1)*.125);
                for (int step = 0; step < t; step++) {
                    int time = reverse ? t-1-step : step; double[] updated = new double[4]; double reduced = 0;
                    for (int i = 0; i < 4; i++) {
                        double input = readCell(type,args.get(ir.actualReads().get(3)),time*4+i);
                        reduced = represented(type,reduced+input); updated[i] = mode < 2 ? input : represented(type,old[i]+input);
                    }
                    old = mode == 1 || mode == 3 ? new double[] {updated[0],updated[2],updated[1],updated[3]} : updated;
                    sum = represented(type,sum+reduced);
                }
                kernel.entryPoint().asSpreader(0,MemorySegment[].class,args.size())
                        .invokeExact(args.toArray(MemorySegment[]::new),new long[] {t,0},0L,0L);
                assertEquals(0,args.get(ir.scalarReads().size()).byteSize(),"empty carry does not suppress nonempty work");
                for (int i = 0; i < 4; i++) assertEquals(old[i],readCell(type,args.get(ir.scalarReads().size()+1),i));
                assertEquals(sum,readCell(type,args.get(ir.scalarReads().size()+2),0));
            }
        }
    }

    /**
     * Reads one represented cell without granting intermediate arithmetic latitude.
     * @param type homogeneous floating type
     * @param segment borrowed exact carrier
     * @param index nonnegative cell index
     * @return represented value widened only for test comparison
     */
    private static double readCell(DataType type, MemorySegment segment, int index) {
        return type == DataType.FLOAT64 ? segment.getAtIndex(ValueLayout.JAVA_DOUBLE,index) : segment.getAtIndex(ValueLayout.JAVA_FLOAT,index);
    }

    /**
     * Writes one represented test input.
     * @param type homogeneous floating type
     * @param segment writable exact carrier
     * @param index nonnegative cell index
     * @param value test value, rounded once for FLOAT32
     */
    private static void writeCell(DataType type, MemorySegment segment, int index, double value) {
        if (type == DataType.FLOAT64) segment.setAtIndex(ValueLayout.JAVA_DOUBLE,index,value);
        else segment.setAtIndex(ValueLayout.JAVA_FLOAT,index,(float)value);
    }

    /**
     * Retains each named producing site's represented precision in the reference.
     * @param type homogeneous floating type
     * @param value exact Java operation result before represented rounding
     * @return FLOAT64 result unchanged or FLOAT32-rounded result widened for comparison
     */
    private static double represented(DataType type, double value) { return type == DataType.FLOAT64 ? value : (float)value; }

    /**
     * Builds a shaped broadcast occurrence without allocating its potentially huge logical domain.
     * @param type homogeneous represented floating type
     * @param t nonnegative static time
     * @param batch nonnegative row extent, possibly larger than int
     * @param width nonnegative column extent
     * @param reverse original-time traversal direction
     * @param emission whether exact next frames are demanded
     * @return source-qualified generation/resource facts
     */
    public static CpuGeneralScanLowering.Facts broadcastFacts(DataType type, long t, long batch, long width, boolean reverse, boolean emission) {
        Tensor h = CpuGeneralScanLoweringTest.tensor(type,false,batch,width), x = CpuGeneralScanLoweringTest.tensor(type,false,batch,width);
        Tensor w = CpuGeneralScanLoweringTest.tensor(type,false,width), next = h.add(x.mul(w));
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),emission ? List.of(next) : List.of());
        var scan = Scan.apply(List.of(CpuGeneralScanLoweringTest.tensor(type,false,batch,width)),
                List.of(CpuGeneralScanLoweringTest.tensor(type,false,t,batch,width)),List.of(CpuGeneralScanLoweringTest.tensor(type,false,width)),body,reverse);
        var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
        return CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(roots));
    }

    /**
     * Projects actual production entry opcodes for bounded structural assertions.
     * @param bytes complete generated class bytes
     * @return ordered executable opcodes, excluding labels and metadata
     */
    private static List<java.lang.classfile.Opcode> opcodes(byte[] bytes) {
        return ClassFile.of().parse(bytes).methods().stream().flatMap(m -> m.code().stream()).flatMap(c -> c.elementStream())
                .filter(java.lang.classfile.Instruction.class::isInstance).map(java.lang.classfile.Instruction.class::cast)
                .map(java.lang.classfile.Instruction::opcode).toList();
    }

    /**
     * Checks the actual control-flow edges rather than merely counting compare opcodes.
     * Every conditional branch goes forward to an exit/selection join; loop backedges are
     * unconditional and land before the next bound test, including dense-copy and full-K loops.
     * @param code actual generated entry Code attribute, borrowed
     */
    private static void assertTopTested(java.lang.classfile.attribute.CodeAttribute code) {
        int bci = 0;
        for (var element : code) if (element instanceof java.lang.classfile.Instruction instruction) {
            if (instruction instanceof java.lang.classfile.instruction.BranchInstruction branch) {
                int target = code.labelToBci(branch.target());
                if (branch.opcode() != java.lang.classfile.Opcode.GOTO)
                    assertTrue(target > bci,"no conditional bottom test at "+bci);
                else if (target < bci) {
                    int position = 0; boolean test = false;
                    for (var candidate : code) if (candidate instanceof java.lang.classfile.Instruction next) {
                        if (position >= target) {
                            if (next instanceof java.lang.classfile.instruction.BranchInstruction bound) {
                                assertTrue(bound.opcode() == java.lang.classfile.Opcode.IFGE
                                        || bound.opcode() == java.lang.classfile.Opcode.IF_ICMPGE,"backedge targets bound test");
                                assertTrue(code.labelToBci(bound.target()) > bci,"test bypasses the complete loop body");
                                test = true; break;
                            }
                            assertTrue(next instanceof java.lang.classfile.instruction.LoadInstruction
                                    || next instanceof java.lang.classfile.instruction.ConstantInstruction
                                    || next.opcode() == java.lang.classfile.Opcode.LCMP,"only bound operands precede test");
                        }
                        position += next.sizeInBytes();
                    }
                    assertTrue(test,"backedge must reach a top test");
                }
            }
            bci += instruction.sizeInBytes();
        }
        assertEquals(code.codeLength(),bci);
    }

    /**
     * Reconstructs only the accepted staged-v1 identity discriminants for provenance checks.
     * @param bytes current compatibility or class-identity stream
     * @return new ASCII stream with v1 baked discriminants and all other facts unchanged
     */
    public static byte[] v1Identity(byte[] bytes) {
        return new String(bytes,java.nio.charset.StandardCharsets.US_ASCII).replace("|scan-shaped-v4=","|scan-shaped-v1=")
                .replace("|scan-matmul-v5=","|scan-matmul-v1=").replace("|matmul-single-product-v1","").getBytes(java.nio.charset.StandardCharsets.US_ASCII);
    }

    /**
     * Projects the accepted v2 identity only; never modifies historical or current Class-Files.
     * @param bytes borrowed current identity stream
     * @return ASCII identity with only the sole staged payload discriminant changed to v2
     */
    public static byte[] v2Identity(byte[] bytes) {
        return new String(bytes,java.nio.charset.StandardCharsets.US_ASCII).replace("|scan-shaped-v4=","|scan-shaped-v2=")
                .replace("|scan-matmul-v5=","|scan-matmul-v2=").replace("|matmul-single-product-v1","").getBytes(java.nio.charset.StandardCharsets.US_ASCII);
    }

    /**
     * Projects the accepted v3 identity for comparison with real CPU0035 artifacts only.
     * @param bytes borrowed current compatibility or class-identity stream
     * @return new ASCII identity with only the sole staged payload discriminant changed to v3
     */
    public static byte[] v3Identity(byte[] bytes) {
        return new String(bytes,java.nio.charset.StandardCharsets.US_ASCII).replace("|scan-shaped-v4=","|scan-shaped-v3=")
                .replace("|scan-matmul-v5=","|scan-matmul-v3=").replace("|matmul-single-product-v1","").getBytes(java.nio.charset.StandardCharsets.US_ASCII);
    }

    /**
     * Hashes complete immutable evidence bytes.
     * @param bytes borrowed class or identity bytes
     * @return lowercase SHA-256 hexadecimal digest
     * @throws Exception if SHA-256 is unavailable
     */
    public static String sha256(byte[] bytes) throws Exception {
        return HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes));
    }

    /**
     * Verifies actual Class-Files, budgets, absence of allocation and staged top-test
     * edges, then optionally retains class/identity/disassembly evidence, never timing.
     * @param name unique evidence case name
     * @param specialization actual production class identity
     * @param bytes actual production-generated class bytes, never patched
     * @throws Exception if verification, evidence output or disassembly fails
     */
    private static void recordClass(String name, CpuKernelSpecialization specialization, byte[] bytes) throws Exception {
        var model = ClassFile.of().parse(bytes);
        assertTrue(ClassFile.of().verify(bytes).isEmpty());
        assertEquals(0,model.constantPool().bootstrapMethodCount());
        for (var method : model.methods()) for (var code : method.code().stream().toList()) {
            var attribute = (java.lang.classfile.attribute.CodeAttribute)code;
            assertTrue(attribute.maxLocals() < 60_000); assertTrue(attribute.codeLength() < 48_000);
            if (specialization.generalScanIr().orElseThrow().stages().isPresent()) assertTopTested(attribute);
            assertTrue(code.elementStream().noneMatch(e -> e instanceof java.lang.classfile.instruction.NewObjectInstruction
                    || e instanceof java.lang.classfile.instruction.NewPrimitiveArrayInstruction
                    || e instanceof java.lang.classfile.instruction.NewReferenceArrayInstruction));
        }
        String root = System.getProperty("synaptik.cpu.generalScan.evidenceRoot"); if (root == null) return;
        var directory = java.nio.file.Path.of(root).resolve("structure"); java.nio.file.Files.createDirectories(directory);
        var path = directory.resolve(name+".class"); java.nio.file.Files.write(path,bytes);
        var process = new ProcessBuilder(java.nio.file.Path.of(System.getProperty("java.home"),"bin","javap").toString(),"-c","-v","-p",path.toString())
                .redirectErrorStream(true).start();
        java.nio.file.Files.writeString(directory.resolve(name+".javap.txt"),new String(process.getInputStream().readAllBytes())); assertEquals(0,process.waitFor());
        var entry = (java.lang.classfile.attribute.CodeAttribute)model.methods().getFirst().code().orElseThrow();
        java.nio.file.Files.write(directory.resolve(name+".compatibility.bytes"),specialization.compatibilityBytes());
        java.nio.file.Files.write(directory.resolve(name+".class-identity.bytes"),specialization.classIdentityBytes());
        java.nio.file.Files.writeString(directory.resolve(name+".identity.txt"),"class_bytes="+bytes.length+"\ncode_length="+entry.codeLength()
                +"\nmax_locals="+entry.maxLocals()+"\nclass_sha256="+sha256(bytes)+"\nkey="+specialization.structuralKey()
                +"\ncompatibility_sha256="+sha256(specialization.compatibilityBytes())+"\nclass_identity="
                +new String(specialization.classIdentityBytes(),java.nio.charset.StandardCharsets.US_ASCII)+"\n");
    }

    @Test void stagedV4MatmulForwardAndActualReverseAdvanceAcceptedV1AndV2ClassFiles() throws Exception {
        // External CPU0032 R4 artifacts; forward small-case hash also equals the independently
        // captured CPU0033 pre-production baseline below. No historical timing rerun is needed.
        var hashes = List.of(List.of("89f788e4cd4be04597c1ecb31644bf329c9d039ddf6fb30ca3069bcbe8ca9895","76368164cc36c70d89e363f10c2c4ca8bbecd1b39befe079be676488eba3fd8a"),
                List.of("912984f6b28c14ebdd51ef9750c8984b472df6d63beb8d06e616dc8aef425e2c","3db2d4bb4d4841a6bd9ea72244faeec663e1f3beb488ae2453b75dc7b3b5f0ff"));
        // CPU0034 actual production generation, staged v2. No frozen-byte counter patching.
        var v2Hashes = List.of(List.of("55dbf382568dcfa7ae4317776346ca37752604ffacdeb88c1c60f7b0ad702a6a","b390416b33c7057bccf0a4d0690f3bbc5a1cae45da7c4aa9b977e6a1ac7300ae"),
                List.of("967c68366e22629be6de6a6bf780e0c34c96435f1809dc2d5eb5575b58246101","9ea44f25abcd9242723435d03dc07fd7a618058fdc6fb1df2d54d7d7d44dd5d8"));
        var v3Hashes = List.of(List.of("4f1d9e33375de8b941488920e86de18045dafadda2fb53e084e026bd708ccb1d","493bb84610ac137c2415e82263a43ff51f7dbd864a40ab506b435e84c28319a6"),
                List.of("e3a7a2d75edd0998d8df554c08a52e95111c07ce791fc6419d72cabc4d6b5b0a","6194eec6a4f6f35d7589fa665a459b4a07e3eeb44f81a300ab0126ed836073be"));
        int shapeIndex = 0;
        for (int[] d : List.of(new int[] {4,2,3,4},new int[] {4,16,24,32})) {
            int t = d[0], m = d[1], k = d[2], n = d[3];
            Tensor h = CpuGeneralScanLoweringTest.tensor(true,m,n), x = CpuGeneralScanLoweringTest.tensor(true,m,k), w = CpuGeneralScanLoweringTest.tensor(true,k,n);
            Tensor next = h.add(x.matmul(w)); var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next));
            Tensor initial = CpuGeneralScanLoweringTest.tensor(true,m,n), seq = CpuGeneralScanLoweringTest.tensor(true,t,m,k), weight = CpuGeneralScanLoweringTest.tensor(true,k,n);
            var scan = Scan.apply(List.of(initial),List.of(seq),List.of(weight),body,false); var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
            var artifact = CpuGeneralScanCapabilityTest.compile(roots,Optional.of(CpuGeneralScanCapabilityTest.request(roots,List.of(initial,seq,weight))),false,new CpuCapabilityProvider());
            var context = io.github.pho001.synaptik.prepare.GraphPreparation.project(artifact,artifact.partitions().getFirst(),
                    io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.DEFAULT);
            var units = new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer().analyze(context).plan().units().stream().filter(u -> u.runtimeFacts().generalScan().isPresent()).toList();
            assertEquals(2,units.size());
            for (int phase = 0; phase < 2; phase++) {
                var u = units.get(phase); byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(u.portablePlan().specialization(),u.portablePlan().kernelIr());
                assertNotEquals(hashes.get(shapeIndex).get(phase),sha256(bytes),"CPU0034 staged-v2 deliberately invalidates v1 bytes");
                assertNotEquals(v2Hashes.get(shapeIndex).get(phase),sha256(bytes),"historical CPU0034 v2 production Class-File");
                assertNotEquals(v3Hashes.get(shapeIndex).get(phase),sha256(bytes),"historical CPU0035 v3 production Class-File");
                recordClass("matmul-"+shapeIndex+"-"+phase,u.portablePlan().specialization(),bytes);
            }
            shapeIndex++;
        }
    }
    @Test void float64ScalarAttributesKeepRawDoubleBitsInBothKeysAndTimeIsNotBaked() {
        var type = DataType.FLOAT64;
        var specs = new ArrayList<CpuKernelSpecialization>();
        for (double attribute : new double[] {.25,.25+0x1p-40}) {
            Tensor h = CpuGeneralScanLoweringTest.tensor(type,false,3), x = CpuGeneralScanLoweringTest.tensor(type,false,3);
            Tensor next = h.add(x).mul(io.github.pho001.synaptik.model.datatype.ScalarValue.float64(attribute));
            var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(next));
            byte[] previous = null; String key = null;
            for (int t : List.of(0,3,100_003)) {
                var scan = Scan.apply(List.of(CpuGeneralScanLoweringTest.tensor(type,false,3)),List.of(CpuGeneralScanLoweringTest.tensor(type,false,t,3)),List.of(),body,false);
                var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst())));
                var immediate = facts.ir().arithmetic().instructions().getLast().scalarImmediate();
                assertEquals(type,immediate.dataType()); assertEquals(Double.doubleToRawLongBits(attribute),immediate.bits());
                var spec = specialization(facts); byte[] bytes = generated(facts).classBytes();
                if (previous != null) { assertArrayEquals(previous,bytes,"O(body) code independent of T"); assertEquals(key,spec.structuralKey()); }
                previous = bytes; key = spec.structuralKey(); if (t == 3) specs.add(spec);
                assertEquals(0,facts.geometry().scratchBytes()); assertEquals(List.of(0,1),facts.ir().demandedPorts(),"no latent history stores/resources");
            }
        }
        assertFalse(Arrays.equals(specs.get(0).classIdentityBytes(),specs.get(1).classIdentityBytes()));
        assertFalse(Arrays.equals(specs.get(0).compatibilityBytes(),specs.get(1).compatibilityBytes()));
    }
    @Test void float64TypedSpecializationsAreAppendOnlyVerifyDoubleLocalsAndRejectWrongArtifacts() {
        for (var f32 : List.of(CpuGeneralScanLowering.analyze(query(3,7,false,List.of(0,1))),shapedFacts(3,2,3),matmulFacts(3,2,3,4))) {
            // Source-derived typed variants below preserve the same semantic shape, not carriers
            // masquerading as a capability proof.
            var ir = f32.ir();
            var arithmetic = ir.arithmetic();
            var typedArithmetic = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr(arithmetic.values().stream()
                    .map(v -> new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value(v.ordinal(),DataType.FLOAT64,v.kind(),v.accessPlan())).toList(),
                    arithmetic.instructions(),arithmetic.loop(),arithmetic.stores());
            var stages = ir.stages().map(p -> new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Stages(p.views(),p.order(),
                    p.cuts().stream().map(c -> new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Cut(c.value(),c.port(),c.offset()*2,c.bytes()*2,c.birth(),c.death(),c.carry())).toList(),
                    p.carryOffsets().stream().map(o -> o < 0 ? o : o*2).toList(),p.carryPorts(),p.scratchBytes()*2));
            var typed = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr(typedArithmetic,ir.formalOrdinals(),ir.actualReads(),ir.scalarReads(),ir.nextOrdinals(),
                    ir.portOrdinals(),ir.demandedPorts(),ir.carries(),ir.boundaryAccesses(),ir.carryCount(),ir.sequenceCount(),ir.reverse(),ir.scratchBanks(),stages);
            var facts = new CpuGeneralScanLowering.Facts(typed,new CpuGeneralScanLowering.Geometry(f32.geometry().time(),f32.geometry().elements(),f32.geometry().scratchBytes()*2),f32.spans(),f32.accessBindings());
            var spec = specialization(facts); var old = specialization(f32);
            assertEquals(71,spec.classIdentitySchema()); assertNotEquals(old.structuralKey(),spec.structuralKey());
            assertFalse(Arrays.equals(old.compatibilityBytes(),spec.compatibilityBytes()));
            assertDoubleClass(generated(facts).classBytes());
            assertThrows(IllegalArgumentException.class,() -> new CpuClassFileKernelGenerator().defineClassBytes(spec, facts.ir().encodedKernelIr(), generated(f32).classBytes()));
            assertThrows(IllegalArgumentException.class,() -> new CpuKernelSpecialization(spec.loweringFingerprint(),spec.executionStrategy(),old.boundaryDataTypes(),spec.carrierPattern(),
                    0,-1,List.of(),spec.scratchParameter(),71,Optional.empty(),Optional.of(typed)));
        }
    }

    /**
     * Checks the actual generated class, including verifier-enforced two-slot double locals.
     * @param bytes complete generated Class-File bytes, borrowed
     */
    public static void assertDoubleClass(byte[] bytes) {
        assertTrue(ClassFile.of().verify(bytes).isEmpty());
        var model = ClassFile.of().parse(bytes); assertEquals(0,model.constantPool().bootstrapMethodCount());
        var instructions = model.methods().stream().flatMap(m -> m.code().stream()).flatMap(c -> c.elementStream()).toList();
        assertTrue(instructions.stream().filter(java.lang.classfile.Instruction.class::isInstance).map(java.lang.classfile.Instruction.class::cast)
                .anyMatch(i -> i.opcode().name().startsWith("DLOAD")),"actual double loops");
        for (var e : instructions) {
            assertFalse(e instanceof java.lang.classfile.instruction.NewObjectInstruction || e instanceof java.lang.classfile.instruction.NewPrimitiveArrayInstruction
                    || e instanceof java.lang.classfile.instruction.NewReferenceArrayInstruction,"no per-step allocation");
            if (e instanceof java.lang.classfile.Instruction i) assertFalse(i.opcode().name().startsWith("F") || i.opcode() == java.lang.classfile.Opcode.D2F,"no float narrowing/site");
        }
        for (var e : model.constantPool()) if (e instanceof MemberRefEntry m)
            assertFalse(m.owner().asInternalName().startsWith("io/github/pho001/synaptik"),"no hidden arithmetic helper or dispatch");
    }
    @Test void pointwiseRemainsFrozenWhileStagedV4HasExplicitV1AndV2IdentityProvenance() throws Exception {
        // Captured by this test on the accepted dirty base, before CPU0033 production edits.
        var expected = List.of(
                List.of("3aa4e7130a5aceb52545cfc3663884599e55ed7f4d7b730c597d246bd0ad637a", "0510bda3365675a42f682dd447716477e140f6e218be97b89cc380a39d8b9965", "6c1c4d6e533b451b4fdb9a69e966b1f2366f02b3e19b933689cfd3f15079d538"),
                List.of("03e3dc889a43f6b8941557b7570128672d35bf841a27f849eae36a01971115f1", "78ee879b40e60385375d93bfb1fe65a005c372e7090085a0c5f040cd4e27e2fc", "8571e6a50f3498a716d7a2ae9d9715b6d5d2a97a3b2895efe3a451f57a8a0832"),
                List.of("e1de5745d528dec19f42d45358d25bc1854bac85c5f2b794284c243e2cda5b58", "32305906eaf3cd3f5b4aa2cfb8527b12408c34a7f5a472dd762ceeed63b0a1a0", "89f788e4cd4be04597c1ecb31644bf329c9d039ddf6fb30ca3069bcbe8ca9895"));
        // Ordered key / compatibility SHA / class SHA for changed staged-v2 cases only.
        var v2Expected = List.of(
                List.of("656fed56420eda9314be90fed51413c10dca6c4fb4416b5cdcc99a1770c8a71b","17113ab65f185989ba048c3e4aab3111037992c5e9699e8bfe7de3d45bccf8b0","eba07ffe0ea82647072eba3f6e8f0e783e4411bfb93579faac0cf395163474a5"),
                List.of("3855fc1972754b9ddaca278bec16176bad6a97b355eeaafd43d5c173145119c5","8ea1e69f6a0f15a79016583c03c3a9038ca10706b457b6cdd55637114b8b5108","55dbf382568dcfa7ae4317776346ca37752604ffacdeb88c1c60f7b0ad702a6a"));
        var v3Expected = List.of(
                List.of("959b374c1e4141e3fca9ffe443921c7e43b4c3b836b54bacc2b9c82f87fd1b9c","4372f0cb1f441c816718f7a6b42025b676e9a106a09971406410b09159972db1","4b39c191abfe04fa06af33523bc24026324f0e74448893fad2c331a63fee71d9"),
                List.of("c4c9578b22b5ed9442d38fe12f50e5aea560a9a03fe39eb8ce71b9576b083521","91e9cec27d2c202aa129c7ece9e81d2b087df9a2a906e130ccff783953ccf393","4f1d9e33375de8b941488920e86de18045dafadda2fb53e084e026bd708ccb1d"));
        int index = 0;
        for (var facts : List.of(CpuGeneralScanLowering.analyze(query(3, 7, false, List.of(0, 1))),
                shapedFacts(3, 2, 3), matmulFacts(3, 2, 3, 4))) {
            var key = specialization(facts);
            var digest = java.security.MessageDigest.getInstance("SHA-256");
            var hashes = expected.get(index++);
            byte[] bytes = generated(facts).classBytes();
            if (facts.ir().stages().isEmpty()) {
                assertEquals(hashes.get(0),key.structuralKey());
                assertEquals(hashes.get(1),sha256(key.compatibilityBytes()));
                assertEquals(hashes.get(2),sha256(bytes));
            } else {
                // Only the sole baked payload changes version; the accepted v1 identity is
                // independently reproducible without restoring historical generator code.
                assertEquals(hashes.get(0),sha256(v1Identity(key.classIdentityBytes())));
                assertEquals(hashes.get(1),sha256(v1Identity(key.compatibilityBytes())));
                assertNotEquals(hashes.get(0),key.structuralKey());
                assertNotEquals(hashes.get(2),sha256(bytes));
                var current = v2Expected.get(index-2);
                assertEquals(current.get(0),sha256(v2Identity(key.classIdentityBytes())));
                assertEquals(current.get(1),sha256(v2Identity(key.compatibilityBytes())));
                assertNotEquals(current.get(2),sha256(bytes),"historical v2 bytecode is not current");
                var v3 = v3Expected.get(index-2);
                assertEquals(v3.get(0),sha256(v3Identity(key.classIdentityBytes()))); assertEquals(v3.get(1),sha256(v3Identity(key.compatibilityBytes())));
                assertNotEquals(v3.get(2),sha256(bytes),"historical v3 bytecode is not current");
                for (byte[] stream : List.of(key.classIdentityBytes(),key.compatibilityBytes())) {
                    String identity = new String(stream,java.nio.charset.StandardCharsets.US_ASCII);
                    assertEquals(1,identity.split("\\|scan-(shaped-v4|matmul-v5)=",-1).length-1,"one sole current baked payload");
                    assertFalse(identity.contains("|scan-shaped-v2=") || identity.contains("|scan-matmul-v2="));
                }
            }
            assertEquals(key.structuralKey(),HexFormat.of().formatHex(digest.digest(key.classIdentityBytes())));
            recordClass("identity-"+index,key,bytes);
        }
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=DataType.class,names={"FLOAT32","FLOAT64"})
    void invariantMatmulRunsOnceBeforeTimeAndScratchSurvivesEveryStep(DataType type) throws Throwable {
        for (int t : List.of(0,1,3,257)) {
            Tensor h = CpuGeneralScanLoweringTest.tensor(type,false,2,3), x = CpuGeneralScanLoweringTest.tensor(type,false,2,3);
            Tensor a = CpuGeneralScanLoweringTest.tensor(type,false,3,2), b = CpuGeneralScanLoweringTest.tensor(type,false,2,3);
            var body = new ScanBody(List.of(h),List.of(x),List.of(a,b),List.of(h.matmul(a.matmul(b))),List.of());
            var scan = Scan.apply(List.of(CpuGeneralScanLoweringTest.tensor(type,false,2,3)),List.of(CpuGeneralScanLoweringTest.tensor(type,false,t,2,3)),
                    List.of(CpuGeneralScanLoweringTest.tensor(type,false,3,2),CpuGeneralScanLoweringTest.tensor(type,false,2,3)),body,false);
            var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(scan.finalCarry())); var original = generated(facts);
            var counter = java.lang.constant.ClassDesc.of(CpuGeneralScanGeneratedKernelTest.class.getName());
            byte[] bytes = ClassFile.of().transformClass(ClassFile.of().parse(original.classBytes()),
                    java.lang.classfile.ClassTransform.transformingMethodBodies((code,element) -> {
                        code.with(element);
                        if (element instanceof java.lang.classfile.Instruction i && i.opcode() == (type == DataType.FLOAT64 ? java.lang.classfile.Opcode.DMUL : java.lang.classfile.Opcode.FMUL))
                            code.getstatic(counter,"observedStores",java.lang.constant.ConstantDescs.CD_long).loadConstant(1L).ladd()
                                    .putstatic(counter,"observedStores",java.lang.constant.ConstantDescs.CD_long);
                    }));
            var lookup = java.lang.invoke.MethodHandles.lookup().defineHiddenClass(bytes,true);
            var target = lookup.findStatic(lookup.lookupClass(),CpuGeneratorSchema.ENTRY_NAME,original.entryPoint().type());
            try (var arena = Arena.ofConfined()) {
                var args = new ArrayList<MemorySegment>(); for (long span : facts.spans()) args.add(arena.allocate(span*type.byteWidth(),type.byteWidth()));
                if (facts.geometry().scratchBytes() > 0) args.add(arena.allocate(facts.geometry().scratchBytes(),type.byteWidth()));
                observedStores = 0;
                target.asSpreader(0,MemorySegment[].class,args.size()).invokeExact(args.toArray(MemorySegment[]::new),new long[] {t,6},0L,6L);
                assertEquals(t == 0 ? 0 : 18L+18L*t,observedStores,"no repeated invariant contraction");
            }
        }
    }

    @Test void matmulIdentityBindsEveryGeometryAndOrderedOperandWhileTimeRemainsDynamic() {
        var a = matmulFacts(3,2,3,4); var time = matmulFacts(257,2,3,4);
        var generator = new CpuClassFileKernelGenerator();
        assertEquals(a.ir().encodedKernelIr(),time.ir().encodedKernelIr());
        assertEquals(specialization(a),specialization(time));
        assertArrayEquals(generated(a).classBytes(),generated(time).classBytes());
        assertTrue(a.ir().bakedSpecializationIdentity().startsWith("|scan-matmul-v5="));
        for (var b : List.of(matmulFacts(3,5,3,4),matmulFacts(3,2,7,4),matmulFacts(3,2,3,6))) {
            assertEquals(a.ir().encodedKernelIr(),b.ir().encodedKernelIr(),"no baked M/N/K in canonical identity");
            assertFalse(Arrays.equals(specialization(a).classIdentityBytes(),specialization(b).classIdentityBytes()));
            assertFalse(Arrays.equals(specialization(a).compatibilityBytes(),specialization(b).compatibilityBytes()));
            assertThrows(IllegalArgumentException.class,() -> generator.defineClassBytes(specialization(b), b.ir().encodedKernelIr(), generated(a).classBytes()));
        }
        var plan = a.ir().stages().orElseThrow(); var order = new ArrayList<>(plan.order()); var s = order.getFirst();
        assertTrue(s.right() >= 0);
        order.set(0,new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Stage(s.values(),s.output(),s.right(),s.axes(),s.reduction(),s.epilogue(),s.input()));
        var ir = a.ir();
        var swapped = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr(ir.arithmetic(),ir.formalOrdinals(),ir.actualReads(),ir.scalarReads(),
                ir.nextOrdinals(),ir.portOrdinals(),ir.demandedPorts(),ir.carries(),ir.boundaryAccesses(),ir.carryCount(),ir.sequenceCount(),ir.reverse(),ir.scratchBanks(),
                Optional.of(new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Stages(plan.views(),order,plan.cuts(),plan.carryOffsets(),plan.carryPorts(),plan.scratchBytes())));
        assertNotEquals(ir.encodedKernelIr(),swapped.encodedKernelIr(),"operand order is semantic topology");
        assertNotEquals(ir.bakedSpecializationIdentity(),swapped.bakedSpecializationIdentity());
        var model = ClassFile.of().parse(generated(a).classBytes());
        assertEquals(0,model.constantPool().bootstrapMethodCount());
        for (var entry : model.constantPool()) if (entry instanceof MemberRefEntry member)
            assertFalse(member.owner().asInternalName().startsWith("io/github/pho001/synaptik"),member.toString());
        assertTrue(model.methods().stream().flatMap(m -> m.code().stream()).flatMap(c -> c.elementStream()).noneMatch(e ->
                e instanceof java.lang.classfile.instruction.NewObjectInstruction || e instanceof java.lang.classfile.instruction.NewPrimitiveArrayInstruction
                        || e instanceof java.lang.classfile.instruction.NewReferenceArrayInstruction));
    }

    /**
     * Builds source-derived proof facts for {@code next=emit=h+x@w}, without allocating data.
     * Used to compare dynamic time reuse with baked matrix geometry in both identity streams.
     * @param t non-negative static sequence length, in steps
     * @param m non-negative matrix row extent
     * @param k non-negative shared contraction extent
     * @param n non-negative product column extent
     * @return non-null immutable lowering facts with final carry and next-frame emission demanded
     * @throws IllegalArgumentException if an extent is negative or the query fails scan qualification
     * @throws ArithmeticException if exact geometry or resource arithmetic overflows
     */
    public static CpuGeneralScanLowering.Facts matmulFacts(int t, int m, int k, int n) {
        Tensor h = CpuGeneralScanLoweringTest.tensor(false,m,n), x = CpuGeneralScanLoweringTest.tensor(false,m,k), w = CpuGeneralScanLoweringTest.tensor(false,k,n);
        Tensor product = x.matmul(w), next = h.add(product);
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next));
        var scan = Scan.apply(List.of(CpuGeneralScanLoweringTest.tensor(false,m,n)),List.of(CpuGeneralScanLoweringTest.tensor(false,t,m,k)),
                List.of(CpuGeneralScanLoweringTest.tensor(false,k,n)),body,false);
        return CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst())));
    }

    @Test void shapedCellCachesRepeatedFormalOnceWithoutLeakingAcrossCoordinates() throws Throwable {
        Tensor h = CpuGeneralScanLoweringTest.tensor(false,2,2), x = CpuGeneralScanLoweringTest.tensor(false,2,2), w = CpuGeneralScanLoweringTest.tensor(false,2);
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(h.add(x.mul(w)).mul(h)),List.of());
        var scan = Scan.apply(List.of(CpuGeneralScanLoweringTest.tensor(false,2,2)),List.of(CpuGeneralScanLoweringTest.tensor(false,3,2,2)),
                List.of(CpuGeneralScanLoweringTest.tensor(false,2)),body,false);
        var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(scan.finalCarry())); var original = generated(facts);
        var counter = java.lang.constant.ClassDesc.of(CpuGeneralScanGeneratedKernelTest.class.getName());
        byte[] bytes = ClassFile.of().transformClass(ClassFile.of().parse(original.classBytes()),
                java.lang.classfile.ClassTransform.transformingMethodBodies((code,element) -> {
                    code.with(element);
                    if (element instanceof java.lang.classfile.instruction.InvokeInstruction invoke && invoke.name().equalsString("get"))
                        code.getstatic(counter,"observedStores",java.lang.constant.ConstantDescs.CD_long).loadConstant(1L).ladd()
                                .putstatic(counter,"observedStores",java.lang.constant.ConstantDescs.CD_long);
                }));
        var lookup = java.lang.invoke.MethodHandles.lookup().defineHiddenClass(bytes,true);
        var entry = lookup.findStatic(lookup.lookupClass(),CpuGeneratorSchema.ENTRY_NAME,original.entryPoint().type());
        try (var arena = Arena.ofConfined()) {
            var args = new ArrayList<MemorySegment>(); for (long span : facts.spans()) args.add(arena.allocate(span*4,4));
            args.add(arena.allocate(facts.geometry().scratchBytes(),4));
            for (int i = 0; i < 4; i++) args.get(0).setAtIndex(ValueLayout.JAVA_FLOAT,i,.125f*(i+1));
            for (int i = 0; i < 12; i++) args.get(1).setAtIndex(ValueLayout.JAVA_FLOAT,i,.03125f*(i+1));
            args.get(2).setAtIndex(ValueLayout.JAVA_FLOAT,0,.5f); args.get(2).setAtIndex(ValueLayout.JAVA_FLOAT,1,.75f);
            observedStores = 0;
            entry.asSpreader(0,MemorySegment[].class,args.size()).invokeExact(args.toArray(MemorySegment[]::new),new long[] {3,4},0L,4L);
            assertEquals(40,observedStores,"3 unique formal loads per coordinate plus 4 final-copy loads, no duplicated h load");
            float[] expected = {.125f,.25f,.375f,.5f};
            for (int t = 0; t < 3; t++) for (int i = 0; i < 4; i++) {
                float old = expected[i], product = .03125f*(t*4+i+1)*(i%2 == 0 ? .5f : .75f);
                float sum = old+product; expected[i] = sum*old;
            }
            assertArrayEquals(expected,args.get(3).toArray(ValueLayout.JAVA_FLOAT));
        }
    }
    @Test void everyBakedGeometryAndResourceMagnitudeChangesBothSpecializationIdentitiesOnly() {
        var facts = shapedFacts(3,2,3); var ir = facts.ir(); var plan = ir.stages().orElseThrow();
        for (int mutation = 0; mutation < 5; mutation++) {
            var views = new ArrayList<>(plan.views()); var cuts = new ArrayList<>(plan.cuts());
            var offsets = new ArrayList<>(plan.carryOffsets()); long scratch = plan.scratchBytes();
            if (mutation < 2) {
                var view = views.getFirst(); var shape = new ArrayList<>(view.shape()); var strides = new ArrayList<>(view.strides());
                if (mutation == 0) shape.set(0,shape.getFirst()+1); else strides.set(0,strides.getFirst()+1);
                views.set(0,new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.View(view.origin(),shape,strides));
            } else if (mutation < 4) {
                var c = cuts.getFirst(); cuts.set(0,new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Cut(c.value(),c.port(),
                        c.offset()+(mutation == 2 ? 4 : 0),c.bytes()+(mutation == 3 ? 4 : 0),c.birth(),c.death(),c.carry()));
            } else { offsets.set(0,offsets.getFirst()+4); scratch += 4; }
            var changed = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr(ir.arithmetic(),ir.formalOrdinals(),ir.actualReads(),ir.scalarReads(),
                    ir.nextOrdinals(),ir.portOrdinals(),ir.demandedPorts(),ir.carries(),ir.boundaryAccesses(),ir.carryCount(),ir.sequenceCount(),ir.reverse(),ir.scratchBanks(),
                    Optional.of(new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Stages(views,plan.order(),cuts,offsets,plan.carryPorts(),scratch)));
            var original = specialization(facts);
            var modified = new CpuKernelSpecialization(original.loweringFingerprint(),original.executionStrategy(),original.boundaryDataTypes(),original.carrierPattern(),
                    original.vectorSpeciesBitSize(),original.materializedSourcePosition(),original.scalarPowerRealizations(),original.scratchParameter(),71,Optional.empty(),Optional.of(changed));
            assertEquals(ir.encodedKernelIr(),changed.encodedKernelIr(),"no geometry magnitude in any canonical field, mutation="+mutation);
            assertFalse(Arrays.equals(original.compatibilityBytes(),modified.compatibilityBytes()));
            assertFalse(Arrays.equals(original.classIdentityBytes(),modified.classIdentityBytes()));
            assertNotEquals(original.structuralKey(),modified.structuralKey());
        }
    }
    @Test void forwardPointwiseClassesRemainByteIdenticalToAcceptedCpu0030Artifacts() throws Exception {
        for (int[] shape : List.of(new int[]{8,32},new int[]{64,256})) for (int mode : List.of(1,2)) {
            Tensor h = CpuGeneralScanLoweringTest.tensor(true,shape[1]), x = CpuGeneralScanLoweringTest.tensor(true,shape[1]);
            Tensor next = h.add(x).tanh();
            var scan = Scan.apply(List.of(CpuGeneralScanLoweringTest.tensor(true,shape[1])),
                    List.of(CpuGeneralScanLoweringTest.tensor(true,shape[0],shape[1])),List.of(),
                    new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(next)),false);
            var artifact = CpuGeneralScanCapabilityTest.compile(List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst()),
                    Optional.empty(),false,new CpuCapabilityProvider());
            var analysis = new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer().analyze(
                    io.github.pho001.synaptik.prepare.GraphPreparation.project(artifact,artifact.partitions().getFirst(),
                            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.DEFAULT));
            for (var unit : analysis.plan().units()) if (unit.runtimeFacts().generalScan().isPresent()) {
                byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(unit.portablePlan().specialization(),unit.portablePlan().kernelIr());
                assertEquals("6c1c4d6e533b451b4fdb9a69e966b1f2366f02b3e19b933689cfd3f15079d538",sha256(bytes),"CPU0030 forward exact class SHA, mode="+mode);
            }
        }
    }
    @Test void shapedResourcesStoreDirectlyToNextBankAndOmitUnobservedTemporalCarry() throws Throwable {
        for (boolean unused : List.of(false,true)) {
            Tensor h = CpuGeneralScanLoweringTest.tensor(false, unused ? new long[] {2} : new long[] {2,2});
            Tensor x = CpuGeneralScanLoweringTest.tensor(false,2,2), w = CpuGeneralScanLoweringTest.tensor(false,2);
            Tensor next = unused ? x.sum(0,false) : h.add(x.mul(w));
            var body = new ScanBody(List.of(h),List.of(x),unused ? List.of() : List.of(w),List.of(next),unused ? List.of(x) : List.of());
            var scan = Scan.apply(List.of(CpuGeneralScanLoweringTest.tensor(false,h.descriptor().shape().toLongArray())),
                    List.of(CpuGeneralScanLoweringTest.tensor(false,3,2,2)),
                    unused ? List.of() : List.of(CpuGeneralScanLoweringTest.tensor(false,2)),body,false);
            var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(unused ? scan.outputs() : scan.finalCarry()));
            var plan = facts.ir().stages().orElseThrow();
            assertEquals(unused ? 0 : 32, facts.geometry().scratchBytes());
            assertEquals(unused ? 0 : 1,plan.cuts().size());
            if (!unused) assertEquals(0,plan.cuts().getFirst().carry(),"producer writes directly into next bank");
            var original = generated(facts);
            var counter = java.lang.constant.ClassDesc.of(CpuGeneralScanGeneratedKernelTest.class.getName());
            byte[] bytes = ClassFile.of().transformClass(ClassFile.of().parse(original.classBytes()),
                    java.lang.classfile.ClassTransform.transformingMethodBodies((code, element) -> {
                        code.with(element);
                        if (element instanceof java.lang.classfile.instruction.InvokeInstruction invoke && invoke.name().equalsString("set"))
                            code.getstatic(counter,"observedStores",java.lang.constant.ConstantDescs.CD_long).loadConstant(1L).ladd()
                                    .putstatic(counter,"observedStores",java.lang.constant.ConstantDescs.CD_long);
                    }));
            var lookup = java.lang.invoke.MethodHandles.lookup().defineHiddenClass(bytes,true);
            var target = lookup.findStatic(lookup.lookupClass(),CpuGeneratorSchema.ENTRY_NAME,original.entryPoint().type());
            try (var arena = Arena.ofConfined()) {
                var args = new ArrayList<MemorySegment>(); for (long span : facts.spans()) args.add(arena.allocate(span*4,4));
                if (facts.geometry().scratchBytes() > 0) args.add(arena.allocate(facts.geometry().scratchBytes(),4));
                observedStores = 0;
                target.asSpreader(0,MemorySegment[].class,args.size()).invokeExact(args.toArray(MemorySegment[]::new),
                        new long[] {3, unused ? 2 : 4},0L,unused ? 2L : 4L);
                assertEquals(unused ? 12 : 16,observedStores,"actual MemorySegment.set calls, unused="+unused);
            }
        }
    }
    @Test void stagedGeometryIsSpecializationOnlyAndCompatibleTimeReusesIdenticalClass() {
        var a = shapedFacts(3, 2, 3); var b = shapedFacts(257, 4, 7); var same = shapedFacts(257, 2, 3);
        assertTrue(a.ir().stages().isPresent());
        assertEquals(a.ir().encodedKernelIr(), b.ir().encodedKernelIr(), "canonical IR excludes EVERY geometry magnitude");
        assertNotEquals(specialization(a).structuralKey(), specialization(b).structuralKey());
        assertFalse(Arrays.equals(specialization(a).compatibilityBytes(), specialization(b).compatibilityBytes()));
        assertFalse(Arrays.equals(specialization(a).classIdentityBytes(), specialization(b).classIdentityBytes()));
        assertEquals(specialization(a), specialization(same));
        assertArrayEquals(generated(a).classBytes(), generated(same).classBytes());
        var model = ClassFile.of().parse(generated(a).classBytes());
        assertEquals(0, model.constantPool().bootstrapMethodCount());
        for (var entry : model.constantPool()) if (entry instanceof MemberRefEntry member)
            assertFalse(member.owner().asInternalName().startsWith("io/github/pho001/synaptik"), member.toString());
        assertTrue(model.methods().stream().flatMap(m -> m.code().stream()).flatMap(c -> c.elementStream())
                .noneMatch(e -> e instanceof java.lang.classfile.instruction.NewObjectInstruction
                        || e instanceof java.lang.classfile.instruction.NewPrimitiveArrayInstruction
                        || e instanceof java.lang.classfile.instruction.NewReferenceArrayInstruction));
        var pointwise = specialization(CpuGeneralScanLowering.analyze(query(3,2,false,List.of(0,1))));
        String expected = CpuGeneratorSchema.CURRENT_VERSION + "|" + pointwise.classIdentitySchema() + "|"
                + pointwise.loweringFingerprint().hex() + "|" + pointwise.executionStrategy().compute() + "|"
                + pointwise.boundaryDataTypes() + "|" + pointwise.carrierPattern() + "|" + pointwise.vectorSpeciesBitSize()
                + "|materialized=" + pointwise.materializedSourcePosition() + "|power=" + pointwise.scalarPowerRealizations()
                + "|scratch=" + pointwise.scratchParameter();
        assertArrayEquals(expected.getBytes(java.nio.charset.StandardCharsets.US_ASCII), pointwise.classIdentityBytes());
        assertEquals("",pointwise.generalScanIr().orElseThrow().bakedSpecializationIdentity());
    }

    public static CpuGeneralScanLowering.Facts shapedFacts(int t, int batch, int width) {
        Tensor h = CpuGeneralScanLoweringTest.tensor(false,batch,width), x = CpuGeneralScanLoweringTest.tensor(false,batch,width);
        Tensor w = CpuGeneralScanLoweringTest.tensor(false,width), next = h.mul(w).add(x).tanh();
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next.sum(0,false)));
        var scan = Scan.apply(List.of(CpuGeneralScanLoweringTest.tensor(false,batch,width)),
                List.of(CpuGeneralScanLoweringTest.tensor(false,t,batch,width)),List.of(CpuGeneralScanLoweringTest.tensor(false,width)),body,false);
        return CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst())));
    }
    public static long observedStores;
    private static final IdentityHashMap<MemorySegment, Long> storesBySegment = new IdentityHashMap<>();

    /**
     * Records an actual instrumented test-only F32 store destination by receiver identity.
     * @param segment non-null receiver of the completed MemorySegment.set call; never mutated here
     */
    public static void recordStore(MemorySegment segment) {
        storesBySegment.merge(segment, 1L, Long::sum);
    }


    @Test void instrumentedClassFileCountsRealStoresWithoutAddingHistoryOrDuplicateBankWrites() throws Throwable {
        for (int t : List.of(0, 1, 3)) for (int n : List.of(0, 2))
            for (var demand : List.of(List.of(0, 1), List.of(0, 4), List.of(1), List.of(4), List.of(0))) {
                var facts = CpuGeneralScanLowering.analyze(query(t, n, false, demand));
                var original = generated(facts);
                var counter = java.lang.constant.ClassDesc.of(CpuGeneralScanGeneratedKernelTest.class.getName());
                byte[] bytes = ClassFile.of().transformClass(ClassFile.of().parse(original.classBytes()),
                        java.lang.classfile.ClassTransform.transformingMethodBodies((code, element) -> {
                            code.with(element);
                            if (element instanceof java.lang.classfile.instruction.InvokeInstruction invoke
                                    && invoke.name().stringValue().equals("set"))
                                code.getstatic(counter, "observedStores", java.lang.constant.ConstantDescs.CD_long)
                                        .loadConstant(1L).ladd().putstatic(counter, "observedStores", java.lang.constant.ConstantDescs.CD_long);
                        }));
                var lookup = java.lang.invoke.MethodHandles.lookup().defineHiddenClass(bytes, true);
                var target = lookup.findStatic(lookup.lookupClass(), CpuGeneratorSchema.ENTRY_NAME, original.entryPoint().type());
                try (var arena = Arena.ofConfined()) {
                    MemorySegment[] arguments = new MemorySegment[facts.spans().size()];
                    for (int p = 0; p < arguments.length; p++) arguments[p] = arena.allocate(facts.spans().get(p) * 4L, 4);
                    observedStores = 0;
                    target.asSpreader(0,MemorySegment[].class,arguments.length).invokeExact(arguments,new long[] {t,n},0L,(long)n);
                    long expected = t == 0 ? demand.contains(0) ? n : 0 : (long)t*n +
                            (demand.contains(0) && demand.size() > 1 ? n : 0);
                    assertEquals(expected, observedStores, "actual MemorySegment.set execution count, demand="+demand+" T="+t+" N="+n);
                    assertEquals(0, facts.geometry().scratchBytes());
                }
            }
    }
    @Test void bakedDirectionDemandAndBankChoicesChangeIdentityButCompatibleGeometryDoesNot() {
        var forward = CpuGeneralScanLowering.analyze(query(3, 2, false, List.of(0, 1)));
        var reverse = CpuGeneralScanLowering.analyze(query(3, 2, true, List.of(0, 1)));
        var finalOnly = CpuGeneralScanLowering.analyze(query(3, 2, false, List.of(0)));
        var historyOnly = CpuGeneralScanLowering.analyze(query(3, 2, false, List.of(1)));
        assertNotEquals(forward.ir().structuralKey(), reverse.ir().structuralKey());
        assertNotEquals(forward.ir().structuralKey(), finalOnly.ir().structuralKey());
        assertNotEquals(forward.ir().structuralKey(), historyOnly.ir().structuralKey());
        var store = new CpuGeneratedKernelArtifactStore();
        var a = generated(forward);
        var cached = store.loadOrGenerate(a.specialization(), forward.ir().encodedKernelIr());
        assertSame(cached, store.loadOrGenerate(a.specialization(), forward.ir().encodedKernelIr()));
        assertThrows(IllegalArgumentException.class, () -> store.loadOrGenerate(a.specialization(), reverse.ir().encodedKernelIr()));
        assertEquals(71, CpuGeneratorSchema.CURRENT_VERSION);
    }

    @Test void independentBakedArithmeticAliasRoleAccessAndStorageMutationsCannotReuseArtifact() {
        var baseline = mutationFacts(0); var original = generated(baseline);
        var store = new CpuGeneratedKernelArtifactStore();
        store.loadOrGenerate(original.specialization(), baseline.ir().encodedKernelIr());
        var keys = new HashSet<String>(); keys.add(baseline.ir().structuralKey());
        for (int mutation = 1; mutation <= 11; mutation++) {
            var changed = mutationFacts(mutation);
            assertTrue(keys.add(changed.ir().structuralKey()), "distinct baked mutation=" + mutation);
            assertThrows(IllegalArgumentException.class, () -> store.loadOrGenerate(
                    original.specialization(), changed.ir().encodedKernelIr()), "stale envelope rejected");
            assertFalse(Arrays.equals(original.classBytes(), generated(changed).classBytes()), "mutation=" + mutation);
        }
        var frame = baseline.ir();
        assertEquals(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Storage.FRAME,
                frame.carries().getFirst().storage());
        var outputBank = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr(
                frame.arithmetic(), frame.formalOrdinals(), frame.actualReads(), frame.scalarReads(),
                frame.nextOrdinals(), frame.portOrdinals(), frame.demandedPorts(),
                List.of(new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Carry(
                        io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Storage.OUTPUT,
                        frame.scalarReads().size())), frame.boundaryAccesses(), frame.carryCount(), frame.sequenceCount(), false, 0);
        assertNotEquals(frame.structuralKey(), outputBank.structuralKey(), "bank changes independently of demand");
        assertThrows(IllegalArgumentException.class, () -> store.loadOrGenerate(original.specialization(), outputBank.encodedKernelIr()));
        // Only the concrete frame extent changes: baked rank/access/body/demand remain identical.
        assertArrayEquals(generated(CpuGeneralScanLowering.analyze(query(3, 2, false, List.of(0, 1)))).classBytes(),
                generated(CpuGeneralScanLowering.analyze(query(257, 19, false, List.of(0, 1)))).classBytes());
    }

    private static CpuGeneralScanLowering.Facts mutationFacts(int mutation) {
        long[] dimensions = mutation == 6 ? new long[] {1, 2} : new long[] {2};
        Tensor h = CpuGeneralScanLoweringTest.tensor(false, dimensions), x = CpuGeneralScanLoweringTest.tensor(false, dimensions);
        Tensor w = CpuGeneralScanLoweringTest.tensor(false, dimensions);
        Tensor sum = mutation == 1 ? h.sub(x) : mutation == 2 ? x.add(h) : h.add(x);
        if (mutation == 4) w = h; // formal alias changes local dataflow independently of demand
        Tensor next = sum.mul(w).mul(io.github.pho001.synaptik.model.datatype.ScalarValue.float32(mutation == 3 ? .25f : .5f)).tanh();
        if (mutation == 8) next = h.neg().add(x.tanh()).mul(w);
        if (mutation == 9) next = x.tanh().add(h.neg()).mul(w);
        Tensor emit = mutation == 5 ? sum : next;
        if (mutation == 7) next = h; // immutable self-source instead of a next-value frame
        var body = new ScanBody(List.of(h), List.of(x), mutation == 4 ? List.of() : List.of(w), List.of(next), List.of(emit));
        Tensor initial = CpuGeneralScanLoweringTest.tensor(false, dimensions);
        Tensor actualWeight = mutation == 11 ? initial : CpuGeneralScanLoweringTest.tensor(false, dimensions);
        if (mutation == 10) actualWeight = CpuGeneralScanLoweringTest.tensor(false).expand(Shape.of(dimensions));
        long[] stacked = new long[dimensions.length + 1]; stacked[0] = 3;
        System.arraycopy(dimensions, 0, stacked, 1, dimensions.length);
        var scan = Scan.apply(List.of(initial), List.of(CpuGeneralScanLoweringTest.tensor(false, stacked)),
                mutation == 4 ? List.of() : List.of(actualWeight), body, false);
        var roots = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
        if (mutation == 11) {
            var artifact = CpuGeneralScanCapabilityTest.compile(roots, Optional.empty(), false, new CpuCapabilityProvider());
            var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(artifact.partitions().getFirst(),
                    artifact.graph().nodes(), artifact.graph().values(), artifact.memory().requirements(), Map.of(),
                    io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.DEFAULT);
            var facts = new CpuGeneralScanLowering().lower(context).generalScan().orElseThrow();
            assertEquals(List.of(0, 1, 0), facts.ir().actualReads());
            return facts;
        }
        return CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(roots));
    }
    @Test void frameBackedTanhUsesNoScratchAndMatchesRepresentedSites() throws Throwable {
        for (boolean reverse : List.of(false, true)) for (int t : List.of(0, 1, 3, 257)) {
            var facts = CpuGeneralScanLowering.analyze(query(t, 3, reverse, List.of(0, 1, 2, 3, 4)));
            assertEquals(0, facts.geometry().scratchBytes());
            var generated = generated(facts);
            try (var arena = Arena.ofConfined()) {
                MemorySegment[] segments = new MemorySegment[facts.spans().size()];
                for (int p = 0; p < segments.length; p++) segments[p] = arena.allocate(facts.spans().get(p) * 4, 4);
                for (int i = 0; i < 3; i++) segments[0].setAtIndex(ValueLayout.JAVA_FLOAT, i, i * .25f);
                for (int i = 0; i < 3 * t; i++) segments[1].setAtIndex(ValueLayout.JAVA_FLOAT, i, (i % 11 - 5) * .03125f);
                float[] state = {0, .25f, .5f};
                float[] history = new float[t * 3], old = new float[t * 3], sum = new float[t * 3];
                for (int k = 0; k < t; k++) {
                    int time = reverse ? t - 1 - k : k;
                    for (int i = 0; i < 3; i++) {
                        old[time * 3 + i] = state[i];
                        float next = state[i] + segments[1].getAtIndex(ValueLayout.JAVA_FLOAT, time * 3L + i);
                        sum[time * 3 + i] = next;
                        state[i] = (float) Math.tanh(next); history[time * 3 + i] = state[i];
                    }
                }
                var call = generated.entryPoint().asSpreader(0, MemorySegment[].class, segments.length);
                call.invokeExact(segments, new long[] {t, 3}, 0L, 3L);
                assertArrayEquals(state, segments[2].toArray(ValueLayout.JAVA_FLOAT));
                assertArrayEquals(history, segments[3].toArray(ValueLayout.JAVA_FLOAT));
                assertArrayEquals(old, segments[4].toArray(ValueLayout.JAVA_FLOAT));
                assertArrayEquals(sum, segments[5].toArray(ValueLayout.JAVA_FLOAT));
                assertArrayEquals(history, segments[6].toArray(ValueLayout.JAVA_FLOAT));
            }
        }
    }

    @Test void generatedCodeHasNoAllocationDispatchOrSynaptikArithmeticHelpers() {
        var generated = generated(CpuGeneralScanLowering.analyze(query(3, 2, false, List.of(0, 1))));
        var model = ClassFile.of().parse(generated.classBytes());
        assertEquals(0, model.constantPool().bootstrapMethodCount());
        for (var entry : model.constantPool()) if (entry instanceof MemberRefEntry member)
            assertFalse(member.owner().asInternalName().startsWith("io/github/pho001/synaptik"), member.toString());
        assertTrue(model.methods().stream().flatMap(m -> m.code().stream()).flatMap(c -> c.elementStream())
                .noneMatch(java.lang.classfile.instruction.NewObjectInstruction.class::isInstance));
        assertArrayEquals(generated.classBytes(), generated(CpuGeneralScanLowering.analyze(
                query(257, 17, false, List.of(0, 1)))).classBytes());
    }

    public static CpuGeneratedKernel generated(CpuGeneralScanLowering.Facts facts) {
        var specialization = specialization(facts);
        var generator = new CpuClassFileKernelGenerator();
        return generator.defineClassBytes(specialization, facts.ir().encodedKernelIr(), generator.generateClassBytes(specialization, facts.ir().encodedKernelIr()));
    }

    public static CpuKernelSpecialization specialization(CpuGeneralScanLowering.Facts facts) {
        var ir = facts.ir();
        return new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                ir.boundaryTypes(),
                Collections.nCopies(ir.boundaryAccesses().size(), CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                0, -1, List.of(), ir.scratchBanks() > 0, 71, Optional.empty(), Optional.of(ir));
    }

    public static OperationCapabilityQuery query(long t, long n, boolean reverse, List<Integer> demand) {
        var h = new ValueId(0); var x = new ValueId(1); var sum = new ValueId(2); var next = new ValueId(3);
        var s = Shape.of(n); var stacked = Shape.of(t, n);
        var values = List.of(new GraphValue(h, descriptor(s)), new GraphValue(x, descriptor(s)),
                new GraphValue(sum, descriptor(s)), new GraphValue(next, descriptor(s)));
        var add = new CompiledNode(new NodeId(0), new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE), List.of(h, x), List.of(sum));
        var tanh = new CompiledNode(new NodeId(1), new Operation(UnaryElementwiseKind.TANH, NoOperationAttrs.INSTANCE), List.of(sum), List.of(next));
        var graph = new CompiledGraphModel(values, List.of(add, tanh), List.of(h, x), List.of(h, sum, next),
                Map.of(add.id(), GraphPhase.FORWARD, tanh.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(next), List.of(next), List.of(h, sum, next), demand);
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, new ScanAttrs(t, 1, 1, 0, 1, 3, reverse)), List.of(descriptor(s), descriptor(stacked)), List.of(descriptor(s), descriptor(stacked), descriptor(stacked), descriptor(stacked), descriptor(stacked)), Optional.of(body), java.util.Optional.empty());
    }

    private static TensorDescriptor descriptor(Shape s) {
        return new TensorDescriptor(DataType.FLOAT32, s, Optional.of(LayoutDescriptor.contiguous(s)), false);
    }
}
