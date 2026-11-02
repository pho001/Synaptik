package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Literal accepted-current CPU0072 controls; capture is explicit, immutable and unmeasured. */
class CpuPointwiseSumCompositionSnapshotTest {
    private static final Map<String,String[]> FROZEN = frozen();

    /**
     * Reads only immutable pre-production capture digests, never candidate-generated expectations.
     * @return captured rows indexed by stable fixture name
     */
    private static Map<String,String[]> frozen() {
        try (var stream = CpuPointwiseSumCompositionSnapshotTest.class.getResourceAsStream("pointwise-sum-composition-before.tsv")) {
            if (stream == null) throw new IllegalStateException("missing CPU0072 frozen evidence");
            var rows = new HashMap<String,String[]>();
            new String(stream.readAllBytes(),StandardCharsets.UTF_8).lines().filter(l -> !l.startsWith("#") && !l.isBlank())
                    .map(l -> l.split("\t")).forEach(row -> rows.put(row[0],row));
            assertEquals(278,rows.size()); return Map.copyOf(rows);
        } catch (java.io.IOException failure) { throw new java.io.UncheckedIOException(failure); }
    }

    /**
     * Checks one literal captured payload, qualifying only vector rows by their captured species.
     * @param name stable pre-production fixture name
     * @param extension class, compatibility, identity, geometry or bindings
     * @param payload current complete payload without binary-name normalization
     * @param bits host species bits or zero for a scalar control
     * @throws Exception if digest computation or retained payload equality fails
     */
    static void assertFrozenPayload(String name,String extension,byte[] payload,int bits) throws Exception {
        if (extension.equals("bindings") || extension.equals("geometry")) assertValueWitness(new String(payload,StandardCharsets.UTF_8));
        var row = FROZEN.get(name); assertNotNull(row,name);
        int captured = Integer.parseInt(row[6]);
        if (!qualifiesSpecies(bits,captured)) {
            System.out.println("CPU0072 UNQUALIFIED vector row "+name+" captured="+captured+" host="+bits);
            return;
        }
        int column = List.of("class","compatibility","identity","geometry","bindings").indexOf(extension)+1;
        assertTrue(column > 0); assertNotEquals("-",row[column],name+" missing capture");
        assertEquals(row[column],CpuSymbolicScanGeneratedKernelTest.sha(payload),name+" literal "+extension);
    }

    /**
     * Proves selective retirement against the immutable original digest, without regenerating
     * an expected value from candidate code. The caller must establish the selected new route.
     * @param name original fixture name
     * @param extension class, compatibility or identity
     * @param payload current selected-route payload
     * @throws Exception if the old row is absent or the supposedly retired payload is unchanged
     */
    static void assertRetiredPayload(String name,String extension,byte[] payload) throws Exception {
        var row = FROZEN.get(name);assertNotNull(row,name);
        int column = List.of("class","compatibility","identity").indexOf(extension)+1;
        assertTrue(column > 0);assertNotEquals("-",row[column]);
        assertNotEquals(row[column],CpuSymbolicScanGeneratedKernelTest.sha(payload),name+" selected row region retires literal "+extension);
    }
    /** Species qualification cannot suppress scalar controls on a different vector host. */
    @Test void scalarControlsRemainQualifiedAcrossVectorSpecies() {
        for (int captured : List.of(128,256,512)) for (int host : List.of(128,256,512)) {
            assertTrue(qualifiesSpecies(0,captured));
            assertEquals(captured == host,qualifiesSpecies(host,captured));
        }
    }

    /**
     * Checks literal species compatibility independently of scalar artifact qualification.
     * @param host selected species width, or zero for a scalar specialization
     * @param captured immutable captured width
     * @return whether literal byte evidence is comparable on this host
     */
    static boolean qualifiesSpecies(int host,int captured) { return host == 0 || host == captured; }

    /** Value witnesses must reject Java object/array identity diagnostics at any nesting depth. */
    @Test void witnessSerializationRejectsObjectAndArrayIdentityLeaks() {
        for (String bad : List.of("PublicationPlan@cb03411","CompileConstantPlan@4c59e45e","selectedAxes=[I@19868320","Layout[strides=[J@e041f0c]"))
            assertThrows(AssertionError.class,() -> assertValueWitness(bad));
        assertValueWitness("selectedAxes=[0, 1], strides=[3, 1], [ValueId[value=0]]");
    }

    /**
     * Rejects non-value diagnostics before any frozen geometry/binding hash comparison.
     * @param value complete deterministic witness text
     */
    static void assertValueWitness(String value) {
        assertFalse(java.util.regex.Pattern.compile("@[0-9a-fA-F]+\\b").matcher(value).find(),value);
    }
    /**
     * Freezes scalar/vector composition and exact versus sequential SUM without executing timing.
     * @throws Exception if generation, immutable capture or literal comparison fails
     */
    @Test void retainedPointwiseAndSumArtifacts() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var read = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,CpuAccessPlan.Regime.DENSE_LINEAR,1,
                    List.of(CpuAccessPlan.AxisRole.CONTIGUOUS),1);
            var write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,CpuAccessPlan.Regime.DENSE_LINEAR,1,
                    List.of(CpuAccessPlan.AxisRole.CONTIGUOUS),1);
            var ir = new CpuKernelIr(List.of(new CpuKernelIr.Value(0,type,CpuKernelIr.Value.Kind.INPUT,read),
                    new CpuKernelIr.Value(1,type,CpuKernelIr.Value.Kind.INPUT,read),
                    new CpuKernelIr.Value(2,type,CpuKernelIr.Value.Kind.VIRTUAL,read),
                    new CpuKernelIr.Value(3,type,CpuKernelIr.Value.Kind.OUTPUT,write)),
                    List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.ADD,List.of(0,1),2),
                            new CpuKernelIr.Instruction(CpuPointwiseOpcode.TANH,List.of(2),3)),
                    new CpuKernelIr.Loop("start","end"),List.of(new CpuKernelIr.Store(3,0)));
            for (boolean vector : List.of(false,true)) for (boolean segment : List.of(false,true)) {
                int bits = !vector ? 0 : type == DataType.FLOAT32
                        ? jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize()
                        : jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.vectorBitSize();
                var carrier = segment ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT : type == DataType.FLOAT32
                        ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY : CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
                var spec = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                        vector ? CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                        Collections.nCopies(3,type),Collections.nCopies(3,carrier),bits,-1,List.of(),false,52);
                capture("pointwise-"+type+"-"+vector+"-"+segment,spec,ir,"species="+bits);
            }
            for (int form = 0; form < 5; form++) for (var kind : List.of(AggregateReductionKind.SUM,AggregateReductionKind.MEAN)) {
                Shape input = Shape.of(2,3), output = form == 0 ? Shape.scalar() : form == 1 ? Shape.of(2)
                        : form == 2 ? Shape.of(1,1) : form == 3 ? Shape.of(2,1) : input;
                if (kind == AggregateReductionKind.MEAN && form >= 3) continue;
                Object attrs = form == 0 ? NoOperationAttrs.INSTANCE : form == 1 ? new AxisReductionAttrs(1,false)
                        : form == 2 ? new MultiAxisReductionAttrs(List.of(0,1),true) : new SumToShapeAttrs(output);
                var context = CpuAggregateLoweringTest.context(kind,type,input,
                        (io.github.pho001.synaptik.model.operation.OperationAttrs)attrs,output);
                var analysis = new CpuPartitionPreparer().analyze(context).plan();
                var route = analysis.units().getFirst().portablePlan();
                capture("exact-"+type+"-"+kind+"-"+form,route.specialization(),route.kernelIr(),
                        Arrays.toString(analysis.aggregateGeometry().orElseThrow().pack(new long[2]))+"\n"
                                +analysis.aggregateGeometry().orElseThrow().workspaceBytes(1));
            }
            for (boolean symbolic : List.of(false,true)) for (boolean length : List.of(false,true)) {
                if (symbolic && !length && type == DataType.FLOAT64) continue;
                Dimension b = symbolic ? new DynamicDimension("B") : new StaticDimension(3);
                Shape row = Shape.ofDimensions(b,new StaticDimension(3));
                Tensor h = CpuSymbolicLengthScanFixture.tensor(type,false,row), x = CpuSymbolicLengthScanFixture.tensor(type,false,row);
                Tensor bias = CpuGeneralScanLoweringTest.tensor(type,false,3);
                Tensor tail = h.neg().sumToShape(Shape.of(1,3)).add(bias);
                var body = new ScanBody(List.of(h),List.of(x),List.of(bias),List.of(h),List.of(tail));
                Tensor initial = CpuSymbolicLengthScanFixture.tensor(type,false,row);
                Tensor sequence = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(new StaticDimension(1),b,new StaticDimension(3)));
                Tensor actualBias = CpuGeneralScanLoweringTest.tensor(type,false,3);
                Tensor old = CpuGeneralScanLoweringTest.tensor(type,false,1,3);
                var scan = length ? Scan.apply(List.of(initial,CpuGeneralScanLoweringTest.tensor(type,false,1,3)),List.of(sequence),List.of(actualBias),
                        CpuSymbolicLengthScanFixture.tensor(DataType.INT64,false,Shape.ofDimensions(b)),1,List.of(OptionalInt.of(0)),
                        new ScanBody(List.of(h,old),List.of(x),List.of(bias),List.of(h,old.add(tail)),List.of(h)),false)
                        : Scan.apply(List.of(initial),List.of(sequence),List.of(actualBias),body,false);
                var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
                var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(roots));
                capture("producer-sum-tail-"+type+"-"+symbolic+"-"+length,
                        CpuGeneralScanGeneratedKernelTest.specialization(facts),facts.ir().encodedKernelIr(),
                        facts.geometry()+"\n"+facts.spans()+"\n"+facts.accessBindings()+"\n"+Arrays.toString(facts.pack(7)));
            }
        }
    }

    /**
     * Stores whole artifacts only into a fresh baseline, or compares with frozen expected bytes.
     * @param name stable fixture name
     * @param spec exact specialization, including species
     * @param ir canonical generated body
     * @param binding deterministic resource/binding witness
     * @throws Exception if capture or literal retention fails
     */
    private void capture(String name,CpuKernelSpecialization spec,CpuKernelIr ir,String binding) throws Exception {
        String directory = System.getenv("CPU0072_ARTIFACT_DIR");
        var payloads = Map.of("class",new CpuClassFileKernelGenerator().generateClassBytes(spec,ir),
                "compatibility",spec.compatibilityBytes(),"identity",spec.classIdentityBytes(),
                "bindings",(spec.entryType()+"\n"+binding).getBytes(StandardCharsets.UTF_8));
        for (var payload : payloads.entrySet()) assertFrozenPayload(name,payload.getKey(),payload.getValue(),spec.vectorSpeciesBitSize());
        if (directory == null) return;
        boolean before = "baseline".equals(System.getenv("CPU0072_CAPTURE"));
        Path root = Path.of(directory), target = root.resolve(before ? "baseline" : "candidate");
        Files.createDirectories(target);
        if (!before && spec.vectorSpeciesBitSize() != 0) {
            String captured = Files.readString(root.resolve("baseline").resolve(name+".bindings"));
            int bits = Integer.parseInt(captured.substring(captured.lastIndexOf("species=")+8));
            if (!qualifiesSpecies(spec.vectorSpeciesBitSize(),bits)) {
                System.out.println("CPU0072 UNQUALIFIED vector row "+name+": captured "+captured
                        +", host "+binding+"; scalar controls remain unconditional");
                return;
            }
        }
        for (var payload : payloads.entrySet()) {
            Path file = target.resolve(name+"."+payload.getKey());
            if (before && Files.exists(file)) {
                assertArrayEquals(Files.readAllBytes(file),payload.getValue(),"partial pre-edit capture remains immutable");
                continue;
            }
            if (!before) assertArrayEquals(Files.readAllBytes(root.resolve("baseline").resolve(file.getFileName())),payload.getValue(),name+" "+payload.getKey());
            Files.write(file,payload.getValue());
        }
    }
}
