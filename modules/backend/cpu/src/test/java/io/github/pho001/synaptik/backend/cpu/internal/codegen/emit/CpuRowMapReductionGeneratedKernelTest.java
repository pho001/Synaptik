package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest;
import io.github.pho001.synaptik.backend.cpu.internal.cache.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.compiler.*;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Unmeasured CPU0073 artifact provenance and generated-body qualification. */
public final class CpuRowMapReductionGeneratedKernelTest {
    /** Exclusive projection, unsupported dtype and mismatched recipe guards at cold factories. */
    @Test void ordinarySpecializationAndRouteFactoriesRequireExclusiveMatchingRegionFacts() {
        var ir = CpuRowMapReductionLoweringTest.candidate(DataType.FLOAT32,false);
        var equivalent = CpuRowMapReductionLoweringTest.candidate(DataType.FLOAT32,false);
        assertNotSame(ir,equivalent);assertEquals(ir,equivalent);assertEquals(ir.hashCode(),equivalent.hashCode());
        for (int mutation = 0; mutation < 9; mutation++) {
            int bad = mutation;
            assertThrows(IllegalArgumentException.class,() -> {
                var candidate = bad == 7 ? CpuRowMapReductionLoweringTest.candidate(DataType.BFLOAT16,false)
                        : bad == 8 ? CpuRowMapReductionLoweringTest.candidate(DataType.FLOAT64,false) : ir;
                var type = candidate.arithmetic().values().getFirst().dataType();
                new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(bad == 4 ? "0".repeat(64) : candidate.structuralKey()),
                        CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,Collections.nCopies(4,type),
                        Collections.nCopies(bad == 6 ? 3 : 4,CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),0,bad == 3 ? 0 : -1,List.of(),bad == 2,
                        bad == 0 ? 52 : 72,Optional.empty(),Optional.empty(),bad == 8,bad == 5,bad == 1 ? Optional.empty() : Optional.of(candidate));
            },"mutation="+bad);
        }
    }

    /** Explicit view provenance, including broadcast and feature-only rank edits, at several R. */
    @Test void ordinaryInternalViewsPreserveOneSharedMapAndBothTerminalOrders() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (int rows : List.of(0,1,5)) {
            Tensor x = CpuGeneralScanLoweringTest.tensor(type,false,rows,2),z = CpuGeneralScanLoweringTest.tensor(type,false,rows,3);
            Tensor w = CpuGeneralScanLoweringTest.tensor(type,false,2,2),bias = CpuGeneralScanLoweringTest.tensor(type,false,2);
            Tensor a = x.add(x).expandDims(1).squeeze(1).add(bias.expand(rows,2));
            Tensor d = a.mul(a),row = d.matmul(w),first = x.permute(1,0).matmul(d),second = z.permute(1,0).matmul(d);
            var compiled = CpuGeneralScanCapabilityTest.compile(List.of(row,first,second),Optional.empty(),false,new CpuCapabilityProvider());
            var analysis = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,compiled.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT));
            assertEquals(1,analysis.plan().units().size());
            var region = analysis.plan().units().getFirst().portablePlan().specialization().rowMapReductionIr().orElseThrow();
            assertEquals(2,region.terminalOutputs().size());
            assertTrue(region.viewOrigins().stream().distinct().count() < region.viewOrigins().size());
            try (var fixture = new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(compiled,CpuPartitionAnalysisInputs.DEFAULT,analysis,1,true,true,Map.of())) {
                for (var binding : compiled.constants().bindableInputBindings()) {
                    var segment = fixture.storage.get(binding.valueId());
                    for (int cell = 0; cell < segment.byteSize()/type.byteWidth(); cell++) putTyped(type,segment,cell,
                            binding.tensorId().equals(bias.id()) ? -.125*(cell+1) : (cell%7-3)*.25);
                }
                fixture.executable.bind(fixture.state).execute();
                double[][] expected = {new double[rows*2],new double[4],new double[6]};
                for (int r = 0; r < rows; r++) {
                    double[] delta = new double[2];
                    for (int j = 0; j < 2; j++) {
                        double xv = ((2*r+j)%7-3)*.25;
                        double av = type == DataType.FLOAT32 ? (float)((float)(xv+xv)+(float)(-.125*(j+1))) : (xv+xv)-.125*(j+1);
                        delta[j] = type == DataType.FLOAT32 ? (float)(av*av) : av*av;
                    }
                    for (int j = 0; j < 2; j++) {
                        double sum = 0;
                        for (int k = 0; k < 2; k++) { double product = delta[k]*((k*2+j)%7-3)*.25; sum = type == DataType.FLOAT32 ? (float)((float)sum+(float)product) : sum+product; }
                        expected[0][r*2+j] = sum;
                        for (int k = 0; k < 2; k++) { double product = ((2*r+k)%7-3)*.25*delta[j]; int at=k*2+j; expected[1][at]=type == DataType.FLOAT32 ? (float)((float)expected[1][at]+(float)product) : expected[1][at]+product; }
                        for (int k = 0; k < 3; k++) { double product = ((3*r+k)%7-3)*.25*delta[j]; int at=k*2+j; expected[2][at]=type == DataType.FLOAT32 ? (float)((float)expected[2][at]+(float)product) : expected[2][at]+product; }
                    }
                }
                for (int port = 0; port < 3; port++) {
                    var segment = fixture.storage.get(compiled.publication().forwardBindings().get(port).valueId());
                    for (int cell = 0; cell < expected[port].length; cell++) assertTyped(type,expected[port][cell],type == DataType.FLOAT64
                            ? segment.getAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,cell) : segment.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,cell),"view port="+port+" R="+rows);
                }
            }
        }
    }
    /** Full-body work-one ranges reject before any output store, even at zero rows. */
    @Test void ordinaryRangeEmptyAndStructuralContract() throws Throwable {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (int rows : List.of(0,1,2,7)) {
            var ir = CpuRowMapReductionLoweringTest.candidate(type,false,rows);
            var boundaries = ir.arithmetic().values().stream().filter(v -> v.kind() != CpuKernelIr.Value.Kind.VIRTUAL).toList();
            var spec = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                    CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,boundaries.stream().map(CpuKernelIr.Value::dataType).toList(),
                    Collections.nCopies(boundaries.size(),CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),0,-1,List.of(),false,
                    72,Optional.empty(),Optional.empty(),false,false,Optional.of(ir));
            var route = new io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan(ir,spec);
            assertEquals(ir.structuralKey(),route.kernelIr().structuralKey());
            var generator = new CpuClassFileKernelGenerator(); byte[] bytes = generator.generateClassBytes(spec,route.kernelIr());
            var body = (java.lang.classfile.attribute.CodeAttribute)java.lang.classfile.ClassFile.of().parse(bytes).methods().getFirst().code().orElseThrow();
            var instructions = body.elementStream().filter(java.lang.classfile.Instruction.class::isInstance)
                    .map(java.lang.classfile.Instruction.class::cast).toList();
            var branches = instructions.stream().filter(i -> i.opcode() == java.lang.classfile.Opcode.GOTO)
                    .map(java.lang.classfile.instruction.BranchInstruction.class::cast).toList();
            // One row-loop back edge; the other GOTO-free guards cannot become per-element work.
            assertEquals(1,branches.size());
            int begin = body.labelToBci(branches.getFirst().target());
            int bci = 0, stores = 0;
            for (var instruction : instructions) {
                if (bci >= begin) {
                    assertNotEquals(java.lang.classfile.Opcode.LALOAD,instruction.opcode(),"geometry must be invocation-bound");
                    assertNotEquals(java.lang.classfile.Opcode.NEW,instruction.opcode());
                    assertNotEquals(java.lang.classfile.Opcode.NEWARRAY,instruction.opcode());
                    assertNotEquals(java.lang.classfile.Opcode.ANEWARRAY,instruction.opcode());
                    if (instruction instanceof java.lang.classfile.instruction.InvokeInstruction call) {
                        assertEquals("java/lang/foreign/MemorySegment",call.owner().asInternalName(),"no hidden arithmetic/dispatch helper");
                        if (call.name().stringValue().startsWith("set")) stores++;
                    }
                }
                bci += instruction.sizeInBytes();
            }
            assertEquals(6,stores,"only row projection(2) and completed terminal(4); no private A/B row stores");
            assertTrue(body.codeLength() <= 48000);
            System.out.println("CPU0073 structural ordinary "+type+" R="+rows+" bytecode="+body.codeLength()+" maxLocals="+body.maxLocals()+" stores="+stores);
            var kernel = generator.defineClassBytes(spec, ir.encodedKernelIr(), bytes);
            try (var arena = java.lang.foreign.Arena.ofConfined()) {
                var x = arena.allocate(2L*rows*type.byteWidth(),type.byteWidth());
                var w = arena.allocate(4L*type.byteWidth(),type.byteWidth());
                var out = arena.allocate(x.byteSize(),type.byteWidth()); var terminal = arena.allocate(w.byteSize(),type.byteWidth());
                for (int cell = 0; cell < 2*rows; cell++) putTyped(type,x,cell,cell+1);
                for (int cell = 0; cell < 4; cell++) putTyped(type,w,cell,cell%3==0 ? 1 : 0);
                var entry = kernel.entryPoint().asSpreader(0,java.lang.foreign.MemorySegment[].class,4);
                var args = new java.lang.foreign.MemorySegment[]{x,w,out,terminal}; long[] geometry = {0,0,0,0,0};
                out.fill((byte)0x6b);terminal.fill((byte)0x6b);
                byte[] savedOut = out.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE),savedTerminal = terminal.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
                entry.invokeExact(args,geometry,0L,0L);
                assertArrayEquals(savedOut,out.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));assertArrayEquals(savedTerminal,terminal.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
                for (long[] range : List.of(new long[]{-1,1},new long[]{0,2},new long[]{1,1},new long[]{1,2})) {
                    assertThrows(IllegalArgumentException.class,() -> { entry.invokeExact(args,geometry,range[0],range[1]); });
                    assertArrayEquals(savedOut,out.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));assertArrayEquals(savedTerminal,terminal.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
                }
                entry.invokeExact(args,geometry,0L,1L);
                if (rows == 0) assertArrayEquals(new byte[4*type.byteWidth()],terminal.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE),"empty R publishes nonempty shared zeros");
            }
            var store = new CpuGeneratedKernelArtifactStore();
            assertSame(store.loadOrGenerate(spec,route.kernelIr()),store.loadOrGenerate(spec,route.kernelIr()));
        }
    }
    /** Typed sites, cancellation and specials, sharing one finalized owner across mixed B. */
    @Test void typedDenseOracleAndConcurrentMixedBatchReuse() throws Exception {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean symbolic : List.of(false,true)) {
            if (symbolic && type == DataType.FLOAT64) continue;
            Dimension r = symbolic ? new DynamicDimension("B") : new StaticDimension(5);
            Tensor x = CpuSymbolicLengthScanFixture.tensor(type,true,Shape.ofDimensions(r,new StaticDimension(2)));
            Tensor z = CpuSymbolicLengthScanFixture.tensor(type,true,Shape.ofDimensions(r,new StaticDimension(1)));
            Tensor w = CpuGeneralScanLoweringTest.tensor(type,true,2,1),u = CpuGeneralScanLoweringTest.tensor(type,true,1,1);
            Tensor bias = CpuGeneralScanLoweringTest.tensor(type,true,1),seed = CpuGeneralScanLoweringTest.tensor(type,false);
            Tensor y = x.matmul(w).add(z.matmul(u)).add(bias),loss = y.mul(y).sum();
            var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                    List.of(new FunctionalGradientRequest.ForwardTensorReference(loss)),List.of(Optional.of(seed)),
                    List.of(x,z,w,u,bias),false,FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
            var compiled = CpuGeneralScanCapabilityTest.compile(List.of(loss,y),Optional.of(request),false,new CpuCapabilityProvider());
            var analysisInputs = CpuPartitionAnalysisInputs.DEFAULT;
            var analysis = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,
                    compiled.partitions().getFirst(),analysisInputs));
            assertTrue(analysis.plan().units().stream().anyMatch(unit -> unit.portablePlan().specialization().rowMapReductionIr().isPresent()));
            var inputs = List.of(x.id(),z.id(),w.id(),u.id(),bias.id(),seed.id());
            try (var owner = new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(
                    compiled,analysisInputs,analysis,1,true,true,Map.of(),symbolic ? OptionalLong.of(64) : OptionalLong.empty())) {
                for (int batch : symbolic ? List.of(64,40,1,64,0) : List.of(5)) for (int scenario = 0; scenario < 4; scenario++)
                    verifyTypedDense(compiled,analysisInputs,analysis,inputs,owner,type,symbolic,batch,scenario);
                if (symbolic) try (var pool = java.util.concurrent.Executors.newFixedThreadPool(4)) {
                    var tasks = new ArrayList<java.util.concurrent.Callable<Void>>();
                    for (int batch : List.of(1,40,64,0,64,1,40,64)) tasks.add(() -> {
                        verifyTypedDense(compiled,analysisInputs,analysis,inputs,owner,type,true,batch,0); return null;
                    });
                    for (var result : pool.invokeAll(tasks)) result.get();
                }
            }
        }
    }

    /**
     * Executes typed dense equations using fresh resources but the exact same finalized owner.
     * @param compiled actual Compiler AD artifact
     * @param analysisInputs exact inputs used to select the immutable plan
     * @param analysis immutable selected plan
     * @param inputs x/z/w/u/bias/seed tensor identities
     * @param owner retained prepared executable and exact memory-plan owner
     * @param type homogeneous working dtype
     * @param symbolic whether invocation B resolves symbolic geometry
     * @param batch invocation row count
     * @param scenario finite cancellation, signed zero, infinity or NaN seed/input case
     */
    private static void verifyTypedDense(CompileArtifacts compiled,CpuPartitionAnalysisInputs analysisInputs,
            io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            List<TensorId> inputs,io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture owner,
            DataType type,boolean symbolic,int batch,int scenario) {
        double huge = type == DataType.FLOAT32 ? 0x1p25 : 0x1p54;
        double[] xa = new double[2*batch],za = new double[batch],ya = new double[batch];
        double[] wa = {1,Math.nextUp(type == DataType.FLOAT32 ? (double)1.0000001192092896f : 1d)},ua = {-.75};
        double seed = scenario == 1 ? -0d : type == DataType.FLOAT32 ? (double)Math.nextUp(1f) : Math.nextUp(1d);
        for (int row = 0; row < batch; row++) {
            xa[2*row] = switch (row%3) { case 0 -> huge; case 1 -> .375; default -> -huge; };
            xa[2*row+1] = (row%2 == 0 ? 1 : -1)*Math.nextUp(type == DataType.FLOAT32 ? (double)0.10000000149011612f : .1);
            za[row] = row%2 == 0 ? .3 : -.7;
        }
        if (batch > 0 && scenario >= 2) xa[0] = scenario == 2 ? Double.POSITIVE_INFINITY : Double.NaN;
        for (int row = 0; row < batch; row++) {
            if (type == DataType.FLOAT32) {
                float a = 0f; a += (float)xa[2*row]*(float)wa[0]; a += (float)xa[2*row+1]*(float)wa[1];
                float b = 0f; b += (float)za[row]*(float)ua[0]; ya[row] = (a+b)+.25f;
            } else {
                double a = 0d; a += xa[2*row]*wa[0]; a += xa[2*row+1]*wa[1];
                double b = 0d; b += za[row]*ua[0]; ya[row] = (a+b)+.25;
            }
        }
        var source = List.of(xa,za,wa,ua,new double[]{.25},new double[]{seed});
        try (var fixture = new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(
                compiled,analysisInputs,analysis,1,true,true,Map.of(),symbolic ? OptionalLong.of(batch) : OptionalLong.empty(),owner)) {
            assertSame(owner.executable,fixture.executable); assertSame(owner.state.memoryPlan(),fixture.state.memoryPlan());
            for (var binding : compiled.constants().bindableInputBindings()) {
                int position = inputs.indexOf(binding.tensorId()); assertTrue(position >= 0);
                var segment = fixture.storage.get(binding.valueId()); double[] values = source.get(position);
                for (int cell = 0; cell < values.length; cell++) putTyped(type,segment,cell,values[cell]);
            }
            if (batch > 0 && scenario == 0) {
                var xValue = compiled.constants().bindableInputBindings().stream().filter(binding -> binding.tensorId().equals(inputs.getFirst())).findFirst().orElseThrow().valueId();
                var dxValue = compiled.publication().gradientBindings().stream().filter(binding -> binding.targetIndex() == 0).findFirst().orElseThrow().valueId();
                var shared = fixture.storage.get(xValue); byte[] before = shared.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
                try (var aliased = new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(
                        compiled,analysisInputs,analysis,1,true,true,Map.of(xValue,shared,dxValue,shared),symbolic ? OptionalLong.of(batch) : OptionalLong.empty(),owner)) {
                    assertThrows(IllegalArgumentException.class,() -> aliased.executable.bind(aliased.state),"all regional outputs, not only the final terminal, must be disjoint");
                    assertArrayEquals(before,shared.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE),"alias rejection precedes writes");
                }
            }
            fixture.executable.bind(fixture.state).execute();
            double[] d = new double[batch],dx = new double[2*batch],dz = new double[batch],dw = new double[2],du = new double[1];
            if (type == DataType.FLOAT64) CpuRowMapReductionJavaOracle.denseFloat64(xa,za,wa,ua,ya,seed,d,dx,dz,dw,du);
            else {
                float[] fd = new float[batch],fx = new float[2*batch],fz = new float[batch],fw = new float[2],fu = new float[1];
                CpuRowMapReductionJavaOracle.denseFloat32(floats(xa),floats(za),floats(wa),floats(ua),floats(ya),(float)seed,fd,fx,fz,fw,fu);
                for (int i = 0; i < d.length; i++) d[i] = fd[i]; for (int i = 0; i < dx.length; i++) dx[i] = fx[i];
                for (int i = 0; i < dz.length; i++) dz[i] = fz[i]; for (int i = 0; i < dw.length; i++) dw[i] = fw[i]; du[0] = fu[0];
            }
            var expected = List.of(dx,dz,dw,du,new double[]{exactSum(d,type)});
            for (var gradient : compiled.publication().gradientBindings()) {
                var segment = fixture.storage.get(gradient.valueId()); var values = expected.get(gradient.targetIndex());
                for (int cell = 0; cell < values.length; cell++) assertTyped(type,values[cell],type == DataType.FLOAT64
                        ? segment.getAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,cell)
                        : segment.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,cell),"B="+batch+" scenario="+scenario+" gradient="+gradient.targetIndex()+" cell="+cell);
            }
            var saved = fixture.storage.get(compiled.publication().forwardBindings().get(1).valueId());
            for (int cell = 0; cell < ya.length; cell++) assertTyped(type,ya[cell],type == DataType.FLOAT64
                    ? saved.getAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,cell) : saved.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,cell),"saved Y");
        }
    }

    /** @param values represented doubles to narrow @return independently allocated float cells */
    private static float[] floats(double[] values) { float[] result = new float[values.length]; for (int i=0;i<values.length;i++) result[i]=(float)values[i]; return result; }
    /** @param type represented dtype @param target writable cells @param cell index @param value represented source */
    private static void putTyped(DataType type,java.lang.foreign.MemorySegment target,long cell,double value) {
        if (type == DataType.FLOAT64) target.setAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,cell,value);
        else target.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,cell,(float)value);
    }
    /** @param type dtype @param expected reference site @param actual generated site @param message failure context */
    private static void assertTyped(DataType type,double expected,double actual,String message) {
        if (Double.isNaN(expected)) assertTrue(Double.isNaN(actual),message);
        else if (type == DataType.FLOAT64) assertEquals(Double.doubleToRawLongBits(expected),Double.doubleToRawLongBits(actual),message);
        else assertEquals(Float.floatToRawIntBits((float)expected),Float.floatToRawIntBits((float)actual),message);
    }
    /**
     * Independent exact mathematical sum, not the backend limb algorithm. All finite inputs
     * are represented exactly in BigDecimal and rounded only once at conversion to the dtype.
     * @param values represented input cells
     * @param type result dtype
     * @return exact SUM result with singleton raw identity and all-negative-zero preservation
     */
    private static double exactSum(double[] values,DataType type) {
        if (values.length == 1) return values[0];
        var sum = java.math.BigDecimal.ZERO; boolean positiveInfinity=false,negativeInfinity=false,negativeZero=values.length>0;
        for (double value : values) {
            if (Double.isNaN(value)) return Double.NaN;
            if (value == Double.POSITIVE_INFINITY) positiveInfinity=true;
            else if (value == Double.NEGATIVE_INFINITY) negativeInfinity=true;
            else sum = sum.add(new java.math.BigDecimal(value));
            negativeZero &= Double.doubleToRawLongBits(value)==Long.MIN_VALUE;
        }
        if (positiveInfinity && negativeInfinity) return Double.NaN;
        if (positiveInfinity) return Double.POSITIVE_INFINITY; if (negativeInfinity) return Double.NEGATIVE_INFINITY;
        return negativeZero ? -0d : type == DataType.FLOAT64 ? sum.doubleValue() : sum.floatValue();
    }
    @Test void actualCombinedScanReverseMatchesIndependentNumericalControl() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true)) {
            var family = new CpuSymbolicLengthScanFixture.Family(type,reverse);
            for (int b : List.of(0,1,7,40,64))
                try (var fixture = new CpuSymbolicLengthScanFixture(family,b)) { fixture.verify(); }
        }
    }

    /** Activity, raw inactive identity and typed cancellation against independent saved primals. */
    @Test void scanOracleCoversDirectionsSeedsMasksAndEmptyTime() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true))
                for (int time : List.of(0,4)) {
            var family = new CpuSymbolicLengthScanFixture.Family(type,reverse,time);
            for (int batch : List.of(0,1,7)) for (int mode = 0; mode < 4; mode++) for (int seeds = 0; seeds < 3; seeds++) {
                try (var f = new CpuSymbolicLengthScanFixture(family,batch)) {
                    for (int r = 0; r < batch; r++) {
                        long length = mode == 0 ? time : mode == 1 ? 0 : mode == 2 ? (r == 0 ? time : 0) : Math.min(r%5,time);
                        f.lengths.setAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG,r,length);
                        for (int j = 0; j < 2; j++) {
                            if (length == 0) f.put(f.initial,2*r+j,-0d);
                            double magnitude = type == DataType.FLOAT32 ? 0x1p25 : 0x1p54;
                            f.put(f.fs,2*r+j,seeds == 1 ? -0d : length == 0 ? -0d : (r%3 == 0 ? magnitude : r%3 == 1 ? .375 : -magnitude));
                        }
                        for (int t = 0; t < time; t++) {
                            for (int j = 0; j < 3; j++) if (t >= length) f.put(f.x,3*(t*batch+r)+j,Double.NaN);
                            for (int j = 0; j < 2; j++) f.put(f.es,2*(t*batch+r)+j,t >= length ? Double.NaN
                                    : seeds == 0 ? 0 : (t+j)%2 == 0 ? .10000000000000003 : -.30000000000000004);
                        }
                    }
                    f.generated();
                    f.oracleForward();
                    if (type == DataType.FLOAT32) CpuRowMapReductionJavaOracle.scanFloat32(f.x,f.w,f.u,f.lengths,
                            f.fs,f.es,f.oraclePrevious,f.oracleSaved,f.expected.get(2),f.expected.get(3),f.expected.get(4),
                            f.expected.get(5),f.expected.get(6),time,batch,reverse);
                    else CpuRowMapReductionJavaOracle.scanFloat64(f.x,f.w,f.u,f.lengths,
                            f.fs,f.es,f.oraclePrevious,f.oracleSaved,f.expected.get(2),f.expected.get(3),f.expected.get(4),
                            f.expected.get(5),f.expected.get(6),time,batch,reverse);
                    for (int port = 0; port < 7; port++) for (long cell = 0; cell < f.actual.get(port).byteSize()/type.byteWidth(); cell++)
                        assertTyped(type,f.get(f.expected.get(port),cell),f.get(f.actual.get(port),cell),
                                "Scan "+type+" reverse="+reverse+" T="+time+" B="+batch+" mode="+mode+" seeds="+seeds+" port="+port+" cell="+cell);
                }
            }
        }
    }
    @Test void lowMatmulContributorsRetainLiteralDecodedFloatWorkingInstructions() {
        for (var type : List.of(DataType.BFLOAT16,DataType.FLOAT16)) {
            var owner = java.lang.constant.ClassDesc.of("LowMatmulWorkingSite");
            var signature = java.lang.constant.MethodTypeDesc.of(java.lang.constant.ConstantDescs.CD_float,
                    java.lang.constant.ConstantDescs.CD_float,java.lang.constant.ConstantDescs.CD_float,
                    java.lang.constant.ConstantDescs.CD_float);
            byte[] observed = java.lang.classfile.ClassFile.of().build(owner,c -> c.withMethodBody("sum",signature,
                    java.lang.classfile.ClassFile.ACC_STATIC,b -> {
                        CpuMatmulEmitter.accumulate(b,type,0,1,2); b.fload(0).freturn();
                    }));
            // Literal pre-CPU0073 working-site instructions, not derived from the new seams.
            byte[] expected = java.lang.classfile.ClassFile.of().build(owner,c -> c.withMethodBody("sum",signature,
                    java.lang.classfile.ClassFile.ACC_STATIC,b -> b.fload(0).fload(1).fload(2).fmul().fadd().fstore(0).fload(0).freturn()));
            assertArrayEquals(expected,observed,type+" decoded low values still use FLOAT locals");
            assertTrue(java.lang.classfile.ClassFile.of().verify(observed).isEmpty());
        }
    }

    @Test void regionRejectsIncompleteAndUnbackedHandleTablesBeforeCallbacks() {
        for (int mutation = 0; mutation < 4; mutation++) {
            int corrupt = mutation;
            assertThrows(IllegalArgumentException.class,() -> java.lang.classfile.ClassFile.of().build(
                    java.lang.constant.ClassDesc.of("BadRegionHandles"),c -> c.withMethodBody("rows",
                            java.lang.constant.MethodTypeDesc.of(java.lang.constant.ConstantDescs.CD_void,java.lang.constant.ConstantDescs.CD_int),
                            java.lang.classfile.ClassFile.ACC_STATIC,b -> {
                var ir = CpuRowMapReductionLoweringTest.candidate(DataType.FLOAT32,false);
                int row = b.allocateLocal(java.lang.classfile.TypeKind.INT);
                int[][] values = {{-1,-1},{-1,-1,-1,-1},new int[2],new int[2],new int[2],{-1,-1},{-1,-1,-1,-1}};
                for (int v : List.of(2,3,4)) for (int j = 0; j < 2; j++) values[v][j] = b.allocateLocal(java.lang.classfile.TypeKind.FLOAT);
                int[][] terminal = {new int[4]};
                for (int j = 0; j < 4; j++) terminal[0][j] = b.allocateLocal(java.lang.classfile.TypeKind.FLOAT);
                if (corrupt == 0) values[2] = new int[0];
                if (corrupt == 1) terminal[0] = new int[3];
                if (corrupt == 2) values[2][0] = -1;
                if (corrupt == 3) terminal[0][3] = terminal[0][2];
                CpuRowMapReductionEmitter.emitBound(b,ir,row,0,values,terminal,
                        () -> fail("invalid handles reached callbacks"),Runnable::run,v -> {},v -> {},() -> {});
                b.return_();
            })));
        }
    }
    /** Exercises selection using real non-recurrent Compiler AD, not a handwritten derivative. */
    @Test void actualOrdinaryDenseSelectsTwoTerminalRegionAndKeepsExactBiasSum() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean symbolic : List.of(false,true)) {
            if (symbolic && type == DataType.FLOAT64) continue;
            var r = symbolic ? new DynamicDimension("B") : new StaticDimension(2);
            Tensor x = CpuSymbolicLengthScanFixture.tensor(type,true,Shape.ofDimensions(r,new StaticDimension(2)));
            Tensor z = CpuSymbolicLengthScanFixture.tensor(type,true,Shape.ofDimensions(r,new StaticDimension(1)));
            Tensor w = CpuGeneralScanLoweringTest.tensor(type,true,2,1);
            Tensor u = CpuGeneralScanLoweringTest.tensor(type,true,1,1);
            Tensor bias = CpuGeneralScanLoweringTest.tensor(type,true,1);
            Tensor y = x.matmul(w).add(z.matmul(u)).add(bias), loss = y.mul(y).sum();
            var compiled = CpuGeneralScanCapabilityTest.compile(List.of(loss),Optional.of(
                    CpuGeneralScanCapabilityTest.request(List.of(loss),List.of(x,z,w,u,bias))),false,new CpuCapabilityProvider());
            var context = GraphPreparation.project(compiled,compiled.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT);
            var analysis = new CpuPartitionPreparer().analyze(context);
            System.out.println(type+" CPU0073 units="+analysis.plan().units().stream().map(unit ->
                    unit.memberNodeOrdinals()+":"+unit.portablePlan().kernelIr().familyIdentity()).toList());
            var regions = analysis.plan().units().stream().filter(unit -> unit.portablePlan().specialization().rowMapReductionIr().isPresent()).toList();
            assertFalse(regions.isEmpty(),type+" actual AD selected region");
            assertTrue(regions.stream().anyMatch(unit -> unit.portablePlan().specialization().rowMapReductionIr().orElseThrow().terminalOutputs().size() == 2));
            for (var unit : regions) {
                assertEquals(1,unit.elementCount()); assertEquals(1,unit.selectedRangeCount());
                byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(unit.portablePlan().specialization(),unit.portablePlan().kernelIr());
                assertTrue(java.lang.classfile.ClassFile.of().verify(bytes).isEmpty());
                if (unit.portablePlan().specialization().rowMapReductionIr().orElseThrow().terminalOutputs().size() == 2)
                    assertOrdinaryDenseStructure(type,bytes);
            }
            assertTrue(analysis.plan().units().stream().anyMatch(unit -> unit.runtimeFacts().aggregateGeometry().isPresent()
                    || unit.portablePlan().batchAggregate().isPresent()),"exact SUM remains separate");
            try (var owner = new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(
                    compiled,CpuPartitionAnalysisInputs.DEFAULT,analysis,1,true,true,Map.of(),symbolic ? OptionalLong.of(64) : OptionalLong.empty())) {
            for (int batch : symbolic ? List.of(64,40,1,64,0,2) : List.of(2)) {
                try (var fixture = new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(
                        compiled,CpuPartitionAnalysisInputs.DEFAULT,analysis,1,true,true,Map.of(),symbolic ? OptionalLong.of(batch) : OptionalLong.empty(),owner)) {
                    assertSame(owner.executable,fixture.executable);
                    assertSame(owner.state.memoryPlan(),fixture.state.memoryPlan());
                    for (var binding : compiled.constants().bindableInputBindings()) {
                        var segment = fixture.storage.get(binding.valueId());
                        for (int cell = 0; cell < segment.byteSize()/type.byteWidth(); cell++) {
                            double value = binding.tensorId().equals(x.id()) ? cell+1
                                    : binding.tensorId().equals(z.id()) ? cell+1
                                    : binding.tensorId().equals(w.id()) ? cell+1 : 1;
                            if (type == DataType.FLOAT64) segment.setAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,cell,value);
                            else segment.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,cell,(float)value);
                        }
                    }
                    fixture.executable.bind(fixture.state).execute();
                    double[] dx = new double[2*batch], dz = new double[batch];
                    double dw0 = 0, dw1 = 0, du = 0, db = 0;
                    for (int row = 0; row < batch; row++) {
                        double d = 14*(row+1); dx[2*row] = d; dx[2*row+1] = 2*d; dz[row] = d;
                        dw0 += (2*row+1)*d; dw1 += (2*row+2)*d; du += (row+1)*d; db += d;
                    }
                    var expected = List.of(dx,dz,new double[]{dw0,dw1},new double[]{du},new double[]{db});
                    for (var gradient : compiled.publication().gradientBindings()) {
                        var segment = fixture.storage.get(gradient.valueId()); var target = expected.get(gradient.targetIndex());
                        assertEquals(target.length,segment.byteSize()/type.byteWidth());
                        for (int cell = 0; cell < target.length; cell++) assertEquals(target[cell],type == DataType.FLOAT64
                                ? segment.getAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,cell)
                                : segment.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,cell),0,
                                type+" symbolic="+symbolic+" B="+batch+" target="+gradient.targetIndex()+" cell="+cell);
                    }
                }
            }
            }
        }
    }

    /**
     * Compares actual dense AD emission to the independently written typed Java equation body.
     * Fixed F2/J1/H1 has eight ordered MULs, seven ADDs and seven producing stores, with one
     * row backedge and no geometry-array read after entering that traversal.
     * @param type represented F32/F64 working dtype
     * @param bytes actual selected projection-72 ClassFile
     */
    private static void assertOrdinaryDenseStructure(DataType type,byte[] bytes) {
        try (var input = CpuRowMapReductionJavaOracle.class.getResourceAsStream("CpuRowMapReductionJavaOracle.class")) {
            assertNotNull(input);
            var oracle = java.lang.classfile.ClassFile.of().parse(input.readAllBytes()).methods().stream()
                    .filter(m -> m.methodName().equalsString(type == DataType.FLOAT64 ? "denseFloat64" : "denseFloat32")).findFirst().orElseThrow();
            var reference = oracle.code().orElseThrow().elementStream().filter(java.lang.classfile.Instruction.class::isInstance)
                    .map(java.lang.classfile.Instruction.class::cast).toList();
            var body = (java.lang.classfile.attribute.CodeAttribute)java.lang.classfile.ClassFile.of().parse(bytes).methods().getFirst().code().orElseThrow();
            var generated = body.elementStream().filter(java.lang.classfile.Instruction.class::isInstance)
                    .map(java.lang.classfile.Instruction.class::cast).toList();
            var mul = type == DataType.FLOAT64 ? java.lang.classfile.Opcode.DMUL : java.lang.classfile.Opcode.FMUL;
            var add = type == DataType.FLOAT64 ? java.lang.classfile.Opcode.DADD : java.lang.classfile.Opcode.FADD;
            assertEquals(8,reference.stream().filter(i -> i.opcode() == mul).count());
            assertEquals(7,reference.stream().filter(i -> i.opcode() == add).count());
            for (var opcode : List.of(mul,add)) assertEquals(reference.stream().filter(i -> i.opcode() == opcode).count(),
                    generated.stream().filter(i -> i.opcode() == opcode).count(),"same separate represented producing sites as optimal typed Java");
            assertEquals(7,generated.stream().filter(i -> i instanceof java.lang.classfile.instruction.InvokeInstruction call
                    && call.owner().asInternalName().equals("java/lang/foreign/MemorySegment") && call.name().stringValue().startsWith("set")).count(),
                    "D for exact db + dX/dZ + completed dW/dU, no other intermediates");
            var backs = generated.stream().filter(i -> i.opcode() == java.lang.classfile.Opcode.GOTO)
                    .map(java.lang.classfile.instruction.BranchInstruction.class::cast).toList();
            assertEquals(1,backs.size());int start = body.labelToBci(backs.getFirst().target()),bci = 0;
            for (var instruction : generated) {
                if (bci >= start) {
                    assertFalse(Set.of(java.lang.classfile.Opcode.NEW,java.lang.classfile.Opcode.NEWARRAY,java.lang.classfile.Opcode.ANEWARRAY,
                            java.lang.classfile.Opcode.INVOKEDYNAMIC,java.lang.classfile.Opcode.LALOAD).contains(instruction.opcode()));
                    if (instruction instanceof java.lang.classfile.instruction.InvokeInstruction call)
                        assertEquals("java/lang/foreign/MemorySegment",call.owner().asInternalName(),"no hidden arithmetic/boxing/dispatch helper");
                }
                bci += instruction.sizeInBytes();
            }
            System.out.println("CPU0073 actual dense "+type+" generated=optimalJava MUL8/ADD7/stores7/rowLoops1 exact-db-boundary retained");
        } catch (java.io.IOException failure) { throw new java.io.UncheckedIOException(failure); }
    }
    /** Literal ordinary and raw-Scan controls remain independent of fresh derivative metadata. */
    @Test void frozenOrdinaryAndRawScanControlsRemainUnchanged() throws Exception {
        var frozen = new TreeMap<String,String>();
        try (var stream = getClass().getResourceAsStream("row-map-reduction-before.tsv")) {
            assertNotNull(stream);
            new String(stream.readAllBytes(),StandardCharsets.UTF_8).lines()
                    .filter(line -> !line.isBlank() && !line.startsWith("#"))
                    .map(line -> line.split("\t")).forEach(row -> frozen.put(row[0],row[1]));
        }
        for (var item : baselineCorpus().entrySet()) {
            String actual = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(item.getValue()));
            assertEquals(frozen.get(item.getKey()),actual,"literal ordinary/raw recipe drift: "+item.getKey());
        }
    }

    /**
     * Constructs genuine ordinary and raw Scan controls without invoking an obsolete baseline
     * writer. Fresh Compiler Scan derivatives are numerical production consumers instead of
     * historical identity fixtures; every ordinary MATMUL realization remains represented.
     * @return ordered independently generated identity, ClassFile, and resource payloads
     */
    public static Map<String, byte[]> baselineCorpus() {
        var payloads = new TreeMap<String, byte[]>();
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            var b = new DynamicDimension("B");
            Tensor a = CpuSymbolicLengthScanFixture.tensor(type, true,
                    Shape.ofDimensions(b, new StaticDimension(3)));
            Tensor w = CpuGeneralScanLoweringTest.tensor(type, true,
                    type == DataType.FLOAT32 ? new long[]{3, 2} : new long[]{3});
            Tensor value = type == DataType.FLOAT32 ? a.matmul(w) : a.add(w);
            var roots = type == DataType.FLOAT32 ? List.of(value)
                    : List.of(value.sum(0, false), value.mean());
            var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                    roots.stream().map(r -> (FunctionalGradientRequest.OutputReference)
                            new FunctionalGradientRequest.ForwardTensorReference(r)).toList(),
                    roots.stream().map(r -> Optional.of(CpuSymbolicLengthScanFixture.tensor(type, false,
                            r.descriptor().shape()))).toList(), List.of(a, w), false,
                    FunctionalGradientRequest.DisconnectedPolicy.ZERO)));
            captureOrdinary(payloads, "batch-" + type, CpuGeneralScanCapabilityTest.compile(
                    roots, Optional.of(request), false, new CpuCapabilityProvider()));
            Tensor pointwise = a.mul(a).add(a);
            captureOrdinary(payloads, "batch-pointwise-sum-" + type,
                    CpuGeneralScanCapabilityTest.compile(List.of(pointwise, pointwise.sum(0, false)),
                            Optional.empty(), false, new CpuCapabilityProvider()));
            for (boolean reverse : List.of(false, true)) for (int family = 0; family < 3; family++) {
                var facts = CpuGeneralScanLowering.analyze(
                        CpuGeneralScanLoweringTest.denseReadFixture(type, family, 4, reverse));
                captureScan(payloads, "static-" + type + "-" + reverse + "-" + family, facts);
                var lengths = CpuGeneralScanLowering.analyze(
                        CpuGeneralScanLoweringTest.lengthFixture(type, family, 4, 7, 2, reverse, true));
                captureScan(payloads, "length-" + type + "-" + reverse + "-" + family, lengths);
            }
            var read = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ, CpuAccessPlan.Regime.DENSE_LINEAR,
                    2, Collections.nCopies(2, CpuAccessPlan.AxisRole.CONTIGUOUS), 2);
            var write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE, CpuAccessPlan.Regime.DENSE_LINEAR,
                    2, Collections.nCopies(2, CpuAccessPlan.AxisRole.CONTIGUOUS), 2);
            for (var form : CpuMatmulIr.Realization.values()) for (boolean tanh : List.of(false, true))
                    for (boolean segment : List.of(false, true)) {
                boolean vector = form == CpuMatmulIr.Realization.DIRECT_N_VECTOR
                        || form == CpuMatmulIr.Realization.TILED_N_VECTOR_2X2;
                int bits = vector ? (type == DataType.FLOAT32
                        ? jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize()
                        : jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.vectorBitSize()) : 0;
                var ir = new CpuMatmulIr(type, type, type, form, tanh ? new CpuMatmulIr.Epilogue(
                        CpuMatmulIr.Epilogue.AddInputOrder.NONE, CpuMatmulIr.Epilogue.Terminal.TANH, null)
                        : CpuMatmulIr.Epilogue.none(), bits, CpuMatmulIr.NumericalForm.SEQUENTIAL,
                        List.of(read, read), write);
                var carrier = segment ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                        : type == DataType.FLOAT32 ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                        : CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
                var spec = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                        vector ? CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR
                                : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                        List.of(type, type, type), Collections.nCopies(3, carrier), bits, -1, List.of(),
                        false, 54, Optional.of(ir));
                capture(payloads, "ordinary-" + type + "-" + form + "-" + tanh + "-" + segment,
                        spec, ir.encodedKernelIr());
            }
        }
        return payloads;
    }

    private static void captureOrdinary(Map<String, byte[]> payloads, String name, CompileArtifacts compiled) {
        var analysis = new CpuPartitionPreparer().analyze(GraphPreparation.project(compiled,
                compiled.partitions().getFirst(), CpuPartitionAnalysisInputs.DEFAULT));
        int index = 0;
        for (var unit : analysis.plan().units()) {
            var route = unit.portablePlan(); String key = name + "-" + index++;
            capture(payloads, key, route.specialization(), route.kernelIr());
            payloads.put(key + ".resources", (unit.boundaryValues() + "\n" + route.batchBindings()
                    + "\n" + route.batchMatmul() + "\n" + route.batchAggregate() + "\n"
                    + analysis.plan().bufferDeclarations() + "\n" + unit.runtimeFacts().workspaceDeclaration()
                    + "\n" + unit.runtimeFacts().workspaceUse())
                    .getBytes(StandardCharsets.UTF_8));
        }
    }

    private static void captureScan(Map<String, byte[]> payloads, String name, CpuGeneralScanLowering.Facts facts) {
        capture(payloads, name, CpuGeneralScanGeneratedKernelTest.specialization(facts), facts.ir().encodedKernelIr());
        payloads.put(name + ".resources", (facts.geometry() + "\n" + facts.spans() + "\n"
                + facts.accessBindings() + "\n" + Arrays.toString(facts.pack(7))).getBytes(StandardCharsets.UTF_8));
    }

    private static void capture(Map<String, byte[]> payloads, String name, CpuKernelSpecialization spec, CpuKernelIr ir) {
        payloads.put(name + ".compatibility", spec.compatibilityBytes());
        payloads.put(name + ".classidentity", spec.classIdentityBytes());
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(spec, ir);
        assertTrue(java.lang.classfile.ClassFile.of().verify(bytes).isEmpty(), name);
        payloads.put(name + ".class", bytes);
    }
}
