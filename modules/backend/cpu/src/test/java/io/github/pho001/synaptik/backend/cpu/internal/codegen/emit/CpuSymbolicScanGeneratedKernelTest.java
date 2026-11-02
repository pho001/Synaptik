package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.lang.classfile.*;
import java.lang.classfile.instruction.*;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.security.MessageDigest;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Numeric recipes, exact static compatibility and direct symbolic Class-File execution. */
public class CpuSymbolicScanGeneratedKernelTest {

    /** ERROR rejects empty-time targets without a selected final-carry identity route. */
    @Test void emptyTimeDisconnectedTargetsRejectErrorPolicy() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean lengthMode : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                Tensor h = CpuSymbolicLengthScanFixture.tensor(type, true, Shape.of(2, 2));
                Tensor x = CpuSymbolicLengthScanFixture.tensor(type, true, Shape.of(2, 2));
                Tensor w = CpuSymbolicLengthScanFixture.tensor(type, true, Shape.of(2, 2));
                Tensor next = x.matmul(w).add(h).tanh();
                Tensor initial = CpuSymbolicLengthScanFixture.tensor(type, true, Shape.of(2, 2));
                Tensor sequence = CpuSymbolicLengthScanFixture.tensor(type, true, Shape.of(0, 2, 2));
                Tensor weight = CpuSymbolicLengthScanFixture.tensor(type, true, Shape.of(2, 2));
                var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of(next));
                var scan = lengthMode ? Scan.apply(List.of(initial), List.of(sequence), List.of(weight),
                        CpuSymbolicLengthScanFixture.tensor(DataType.INT64, false, Shape.of(2)), 1,
                        List.of(OptionalInt.of(0)), body, reverse)
                        : Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, reverse);
                var roots = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
                var selected = roots.stream().map(r -> (FunctionalGradientRequest.OutputReference)
                        new FunctionalGradientRequest.ForwardTensorReference(r)).toList();
                var seeds = roots.stream().map(r -> Optional.of(
                        CpuSymbolicLengthScanFixture.tensor(type, false, r.descriptor().shape()))).toList();
                var targets = List.of(initial, sequence, weight);
                var strict = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                        selected, seeds, targets, false, FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
                assertThrows(IllegalArgumentException.class, () -> CpuGeneralScanCapabilityTest.compile(
                        roots, Optional.of(strict), false, new CpuCapabilityProvider()));
                var emissionOnly = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                        List.of(selected.get(1)), List.of(seeds.get(1)), List.of(initial), false,
                        FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
                assertThrows(IllegalArgumentException.class, () -> CpuGeneralScanCapabilityTest.compile(
                        roots, Optional.of(emissionOnly), false, new CpuCapabilityProvider()),
                        "T=0 emission does not inherit the final-carry identity route");
            }
    }


    /** Empty production-bound invocations preserve raw public state and seed identities. */
    @Test void cpu0060EmptyMaximumUsesInitialPublicationForBanksAndOtherFinalPorts() {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true))
            for (int time : new int[]{0,4}) {
                var family = new CpuSymbolicLengthScanFixture.Family(type,reverse,time);
                for (int batch : new int[]{0,1,2,3,7}) for (long bits : type == DataType.FLOAT32
                        ? new long[]{0x80000000L,0x7fc12345L} : new long[]{0x8000000000000000L,0x7ff8123456789abcL})
                    try (var f = new CpuSymbolicLengthScanFixture(family,batch)) {
                        f.lengths.fill((byte)0);
                        f.consumer.poisonProducedPublications((byte)0x5a);
                        for (var input : List.of(f.initial,f.fs))
                            for (long i = 0; i < input.byteSize()/type.byteWidth(); i++) putRaw(type,input,i,bits);
                        f.execute();
                        for (int p = 0; p < f.actual.size(); p++) for (long i = 0; i < f.actual.get(p).byteSize()/type.byteWidth(); i++)
                            assertEquals(p == 0 || p == 2 ? bits : 0L,raw(type,f.actual.get(p),i),"bound empty publication port="+p);
                    }
            }
    }

    /**
     * A true B=1 reduction is raw movement, including NaN payload and negative zero. B>1
     * with one active row still adds to typed zero. The separate shared-carry ADD retains its
     * negative-zero initializer, making raw-singleton versus SUM observable at publication.
     * @throws Throwable if a generated entry cannot execute
     */
    @Test void cpu0060RawSingletonAndOneActiveAmongManyRemainDistinct() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true)) {
            var b = new DynamicDimension("singletonRows"); var row = Shape.ofDimensions(b,new StaticDimension(3));
            Tensor h = CpuSymbolicLengthScanFixture.tensor(type,false,row), x = CpuSymbolicLengthScanFixture.tensor(type,false,row);
            Tensor old = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(3));
            var sum = h.sumToShape(Shape.of(3));
            var scan = Scan.apply(List.of(CpuSymbolicLengthScanFixture.tensor(type,false,row),CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(3))),
                    List.of(CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(new StaticDimension(1),b,new StaticDimension(3)))),
                    List.of(),CpuSymbolicLengthScanFixture.tensor(DataType.INT64,false,Shape.ofDimensions(b)),1,List.of(OptionalInt.of(0)),
                    new ScanBody(List.of(h,old),List.of(x),List.of(),List.of(h,old.add(sum)),List.of(h,h)),reverse);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(roots)); var ir = facts.ir();
            var generated = CpuGeneralScanGeneratedKernelTest.generated(facts); inspect(generated.classBytes());
            var entry = generated.entryPoint().asSpreader(0,MemorySegment[].class,ir.boundaryAccesses().size()+1);
            for (int batch : new int[]{1,3,2,1,0}) try (var arena = Arena.ofConfined()) {
                var args = repairArguments(facts,batch,arena);
                long[] bits = type == DataType.FLOAT32 ? new long[]{0x80000000L,0x7fc12345L,1L}
                        : new long[]{0x8000000000000000L,0x7ff8123456789abcL,1L};
                for (int col = 0; col < 3; col++) putRaw(type,args[ir.actualReads().get(1)],col,bits[0]);
                for (int r = 0; r < batch; r++) {
                    args[ir.lengthRead()].setAtIndex(ValueLayout.JAVA_LONG,r,r == 0 ? 1 : 0);
                    for (int col = 0; col < 3; col++) putRaw(type,args[ir.actualReads().getFirst()],r*3L+col,bits[col]);
                }
                long[] pack = facts.pack(batch); entry.invokeExact(args,pack,0L,pack[1]);
                int out = ir.scalarReads().size();
                assertEquals(-1,args[out+2].mismatch(args[out+3]),"distinct demanded wrappers");
                for (int col = 0; col < 3; col++) {
                    if (col == 1 && batch > 0) assertTrue(Double.isNaN(scalarGet(type,args[out+1],col)));
                    else if (batch == 1) assertEquals(bits[col],raw(type,args[out+1],col),"raw singleton then separate carry ADD");
                    else assertEquals(batch == 0 ? bits[0] : col == 0 ? 0 : bits[col],raw(type,args[out+1],col));
                }
            }
        }
    }

    /**
     * Makes raw singleton movement observable on an actually selected computed-map region.
     * TANH(-0) is reused by row carry/emissions and represented SUM; the shared carry starts
     * at -0, so a mistaken positive-zero-start SUM would change the final published sign.
     * @throws Throwable if admission, ClassFile linkage or represented results differ
     */
    @Test void selectedComputedRowMapPreservesRawSingletonBeforeDistinctCarryAdd() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true)) {
            var b = new DynamicDimension("computedSingletonRows"); var row = Shape.ofDimensions(b,new StaticDimension(3));
            Tensor h = CpuSymbolicLengthScanFixture.tensor(type,false,row),x = CpuSymbolicLengthScanFixture.tensor(type,false,row);
            Tensor old = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(3)),mapped = h.tanh();
            var sum = mapped.sumToShape(Shape.of(3));
            var scan = Scan.apply(List.of(CpuSymbolicLengthScanFixture.tensor(type,false,row),CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(3))),
                    List.of(CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(new StaticDimension(1),b,new StaticDimension(3)))),
                    List.of(),CpuSymbolicLengthScanFixture.tensor(DataType.INT64,false,Shape.ofDimensions(b)),1,List.of(OptionalInt.of(0)),
                    new ScanBody(List.of(h,old),List.of(x),List.of(),List.of(mapped,old.add(sum)),List.of(mapped,mapped)),reverse);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(roots));var ir = facts.ir();
            var stages = ir.stages().orElseThrow();var region = stages.rowMapReductionIr().orElseThrow();
            assertEquals(1,region.sums().size());assertTrue(region.sums().getFirst().sumToShape());
            assertEquals(List.of(0),region.sums().getFirst().axes());
            assertTrue(region.arithmetic().instructions().stream().anyMatch(i -> i.opcode() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.TANH));
            assertEquals(1,stages.order().stream().mapToInt(s -> s.epilogue().size()).sum(),"separate shared-carry finish ADD");
            var generated = CpuGeneralScanGeneratedKernelTest.generated(facts);inspect(generated.classBytes());
            var entry = generated.entryPoint().asSpreader(0,MemorySegment[].class,generated.entryPoint().type().parameterCount()-3);
            long negativeZero = type == DataType.FLOAT32 ? 0x80000000L : Long.MIN_VALUE;
            for (int batch : new int[]{1,3,0,1}) try (var arena = Arena.ofConfined()) {
                var args = Arrays.copyOf(repairArguments(facts,batch,arena),generated.entryPoint().type().parameterCount()-3);
                for (int col = 0; col < 3; col++) putRaw(type,args[ir.actualReads().get(1)],col,negativeZero);
                for (int r = 0; r < batch; r++) {
                    args[ir.lengthRead()].setAtIndex(ValueLayout.JAVA_LONG,r,r == 0 ? 1 : 0);
                    for (int col = 0; col < 3; col++) scalarPut(type,args[ir.actualReads().getFirst()],r*3L+col,r == 0 ? -0d : Double.NaN);
                }
                long[] pack = facts.pack(batch);entry.invokeExact(args,pack,0L,pack[1]);
                int out = ir.scalarReads().size();
                for (int col = 0; col < 3; col++) {
                    assertEquals(batch == 0 || batch == 1 ? negativeZero : 0,raw(type,args[out+1],col),
                            "selected SUM raw geometry, then -0 carry ADD: B="+batch);
                    if (batch > 0) assertEquals(negativeZero,raw(type,args[out],col),"computed TANH(-0) row publication");
                }
                assertEquals(-1,args[out+2].mismatch(args[out+3]),"both demanded computed histories are retained");
            }
        }
    }

    /**
     * Reads represented storage bits without NaN canonicalization.
     * @param type homogeneous floating storage dtype
     * @param segment live checked storage
     * @param index element index
     * @return unsigned F32 bits or the complete F64 bit pattern
     */
    private static long raw(DataType type,MemorySegment segment,long index) {
        return type == DataType.FLOAT32 ? Integer.toUnsignedLong(segment.getAtIndex(ValueLayout.JAVA_INT,index))
                : segment.getAtIndex(ValueLayout.JAVA_LONG,index);
    }

    /**
     * Writes represented bits without arithmetic or conversion.
     * @param type homogeneous floating storage dtype
     * @param segment live writable storage
     * @param index element index
     * @param bits unsigned F32 bits or complete F64 pattern
     */
    private static void putRaw(DataType type,MemorySegment segment,long index,long bits) {
        if (type == DataType.FLOAT32) segment.setAtIndex(ValueLayout.JAVA_INT,index,(int)bits);
        else segment.setAtIndex(ValueLayout.JAVA_LONG,index,bits);
    }

    /**
     * Executes ROW(1) pointwise staging followed by a ROW(0) alias with changing carry/time
     * bindings. Batch/feature collisions cannot establish address or role equality; duplicate
     * emission wrappers retain the same producing bits.
     * @throws Throwable if the generated entry cannot execute
     */
    @Test void cpu0060PointwisePrefixesKeepTransposedAxesAndChangedBindingLifetimes() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true))
            for (int features : new int[]{1,2,3}) {
                var b = new DynamicDimension("prefixRows"); var row = Shape.ofDimensions(b,new StaticDimension(features));
                Tensor h = CpuSymbolicLengthScanFixture.tensor(type,false,row), x = CpuSymbolicLengthScanFixture.tensor(type,false,row);
                var next = h.permute(1,0).neg().permute(1,0).add(x);
                var scan = Scan.apply(List.of(CpuSymbolicLengthScanFixture.tensor(type,false,row)),
                        List.of(CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(new StaticDimension(3),b,new StaticDimension(features)))),
                        List.of(),CpuSymbolicLengthScanFixture.tensor(DataType.INT64,false,Shape.ofDimensions(b)),1,List.of(OptionalInt.of(0)),
                        new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(next,next)),reverse);
                var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
                var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(roots)); var ir = facts.ir(); var plan = ir.stages().orElseThrow();
                assertTrue(plan.order().stream().anyMatch(s -> s.output() < 0 && plan.rowAxes().get(s.values().getFirst()) == 1),"real pointwise stage must bind ROW(1), not only row zero");
                var generated = CpuGeneralScanGeneratedKernelTest.generated(facts); inspect(generated.classBytes());
                var entry = generated.entryPoint().asSpreader(0,MemorySegment[].class,ir.boundaryAccesses().size()+1);
                for (int batch : new int[]{3,1,2,0,7,3}) try (var arena = Arena.ofConfined()) {
                    var args = repairArguments(facts,batch,arena); long[] pack = facts.pack(batch);
                    for (int version = 0; version < 2; version++) {
                        double[] carry = new double[batch*features];
                        double[] history = new double[3*batch*features];
                        for (int r = 0; r < batch; r++) {
                            args[ir.lengthRead()].setAtIndex(ValueLayout.JAVA_LONG,r,Math.min((r+version)%4,3));
                            for (int col = 0; col < features; col++) {
                                carry[r*features+col] = .125*(r+col+1+version);
                                scalarPut(type,args[0],r*features+col,carry[r*features+col]);
                                for (int t = 0; t < 3; t++) scalarPut(type,args[1],(t*batch+r)*features+col,
                                        t < args[ir.lengthRead()].getAtIndex(ValueLayout.JAVA_LONG,r) ? .25*(1+t+r+col+version) : Double.NaN);
                            }
                        }
                        for (int visit = 0; visit < 3; visit++) {
                            int t = reverse ? 2-visit : visit;
                            for (int r = 0; r < batch; r++) if (t < args[ir.lengthRead()].getAtIndex(ValueLayout.JAVA_LONG,r))
                                for (int col = 0; col < features; col++) {
                                    int cell = r*features+col;
                                    carry[cell] = rounded(type,-carry[cell]+scalarGet(type,args[1],t*batch*features+cell));
                                    history[t*batch*features+cell] = carry[cell];
                                }
                        }
                        entry.invokeExact(args,pack,0L,pack[1]); int out = ir.scalarReads().size();
                        for (int i = 0; i < carry.length; i++) assertEquals(carry[i],scalarGet(type,args[out],i));
                        for (int i = 0; i < history.length; i++) assertEquals(history[i],scalarGet(type,args[out+1],i));
                        assertEquals(-1,args[out+1].mismatch(args[out+2]),"saved duplicate uses exact producing local");
                    }
                }
            }
    }


    /** Executes unchanged independent typed Java oracles at empty/mixed/full activity and reuse. */
    @Test void cpu0058TypedActualReversePreservesEmptyAndActiveDomains() {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true))
            for (int time : new int[]{0,4}) {
                var family = new CpuSymbolicLengthScanFixture.Family(type,reverse,time);
                for (int batch : new int[]{0,1,7,32}) for (int pattern = 0; pattern < 3; pattern++)
                    try (var f = new CpuSymbolicLengthScanFixture(family,batch)) {
                        for (int row = 0; row < batch; row++) f.lengths.setAtIndex(ValueLayout.JAVA_LONG,row,
                                pattern == 0 ? 0 : pattern == 1 ? row%(time+1) : time);
                        if (time == 0) {
                            f.execute();
                            assertArrayEquals(f.initial.toArray(ValueLayout.JAVA_BYTE),f.actual.get(0).toArray(ValueLayout.JAVA_BYTE));
                            assertArrayEquals(f.fs.toArray(ValueLayout.JAVA_BYTE),f.actual.get(2).toArray(ValueLayout.JAVA_BYTE));
                            for (int p = 3; p < 7; p++) for (byte value : f.actual.get(p).toArray(ValueLayout.JAVA_BYTE)) assertEquals(0,value);
                            continue; // The retained literal oracle is deliberately T=4, never rewritten.
                        }
                        f.verify();
                        for (int t = 0; t < time; t++) for (int row = 0; row < batch; row++)
                            if (t >= f.lengths.getAtIndex(ValueLayout.JAVA_LONG,row)) {
                                for (int k = 0; k < 3; k++) f.put(f.x,(t*batch+row)*3L+k,Double.NaN);
                                for (int k = 0; k < 2; k++) f.put(f.es,(t*batch+row)*2L+k,Double.NaN);
                            }
                        f.verify();
                    }
            }
    }

    /**
     * Qualifies the prebound joint production action against independently differentiated public seeds.
     * Current derivative constants are intrinsic, not injectable boundary formals.
     */
    @Test void cpu0045PreboundJointActionRetainsPublicSeeds() {
        var family = new CpuSymbolicLengthScanFixture.Family(DataType.FLOAT32,false);
        for (int batch : new int[]{7,32}) try (var f = new CpuSymbolicLengthScanFixture(family,batch)) {
            var action = f.preboundExecution();
            for (double scalar : new double[]{1,2.5,-.75}) {
                f.fill(f.fs,scalar); action.run(); f.oracleForward(); f.oracleReverse(); assertEquivalentOutputsCpu0045(f);
            }
        }
    }

    @Test void cpu0045RawStableCarryProofRejectsNoncombinedRecipes() {
        var noncombined = CpuGeneralScanGeneratedKernelTest.matmulFacts(3,2,3,4).ir();
        inspectStableBindingCpu0045(noncombined);
        var b = new DynamicDimension("allValidRows");
        Tensor h = symbolic(false,b,new StaticDimension(2)), x = symbolic(false,b,new StaticDimension(2));
        var allValid = Scan.apply(List.of(symbolic(false,b,new StaticDimension(2))),
                List.of(symbolic(false,new StaticDimension(3),b,new StaticDimension(2))),List.of(),
                new ScanBody(List.of(h),List.of(x),List.of(),List.of(h.add(x)),List.of()),false);
        var allValidIr = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(allValid.finalCarry())).ir();
        assertTrue(allValidIr.symbolic()); assertFalse(allValidIr.lengthAware());
        assertEquals(0,allValidIr.stages().orElseThrow().carryPorts().getFirst());
        assertTrue(allValidIr.stages().orElseThrow().carryOffsets().getFirst() < 0);
        inspectStableBindingCpu0045(allValidIr);
    }


    /**
     * Checks that genuine raw noncombined recipes do not acquire stable-carry bindings.
     * @param ir complete raw proof IR
     * @throws AssertionError if the negative binding proof changes existing strides
     */
    private static void inspectStableBindingCpu0045(CpuGeneralScanIr ir) {
        var spec = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,ir.boundaryTypes(),
                Collections.nCopies(ir.boundaryAccesses().size(),CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                0,-1,List.of(),ir.scratchBanks() > 0,71,Optional.empty(),Optional.of(ir));
        byte[] bytes = ClassFile.of().build(java.lang.constant.ClassDesc.of("io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.StableCarryProof"),builder -> {
            builder.withVersion(ClassFile.JAVA_26_VERSION,0);
            builder.withMethod("invoke",java.lang.constant.MethodTypeDesc.ofDescriptor(spec.entryType().descriptorString()),
                    java.lang.reflect.AccessFlag.STATIC.mask(),method -> method.withCode(code -> {
                        CpuCarrierEmitter.prepareSegmentLayouts(code,spec.boundaryDataTypes(),spec.carrierPattern(),true,ir.symbolic() && ir.lengthAware());
                        var state = new CpuGeneralScanEmitter.Shaped(code,spec,ir);
                        if (ir.symbolic()) state.bindSymbolicGeometry();
                        var before = ir.symbolic() ? Arrays.stream(state.batchStrides).map(int[]::clone).toArray(int[][]::new) : null;
                        assertFalse(state.bindStableLengthCarry(0)); assertTrue(state.zeroBases.isEmpty());
                        if (before != null) for (int v = 0; v < before.length; v++) assertArrayEquals(before[v],state.batchStrides[v]);
                        code.return_();
                    }));
        });
        assertTrue(ClassFile.of().verify(bytes).isEmpty());
    }

    /**
     * Executes a real shared read alias: a scalar-splat actual initializes an evolving shared
     * carry and independently supplies an invariant row operand. Occurrence lowering deduplicates
     * the read slot, but rebinding only the carry must not change the invariant's mapping/value.
     * @throws Throwable if the generated entry or any typed/raw result assertion fails
     */
    @Test void cpu0045CopiedCarryReadAliasLeavesInvariantAtItsOriginalActual() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true))
            for (int time : new int[]{0,3}) {
                var b = new DynamicDimension("aliasRows"); var row = Shape.ofDimensions(b,new StaticDimension(2));
                Tensor h = CpuSymbolicLengthScanFixture.tensor(type,false,row), x = CpuSymbolicLengthScanFixture.tensor(type,false,row);
                Tensor s = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2)), invariant = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.of(2));
                Tensor initial = CpuSymbolicLengthScanFixture.tensor(type,false,row);
                Tensor sequence = CpuSymbolicLengthScanFixture.tensor(type,false,Shape.ofDimensions(new StaticDimension(time),b,new StaticDimension(2)));
                Tensor sharedActual = TensorFactory.create(new TensorDescriptor(type,Shape.of(2),
                        Optional.of(LayoutDescriptor.of(Shape.of(2),new long[]{0},0,true)),false));
                Tensor lengths = CpuSymbolicLengthScanFixture.tensor(DataType.INT64,false,Shape.ofDimensions(b));
                var scan = Scan.apply(List.of(initial,sharedActual),List.of(sequence),List.of(sharedActual),lengths,1,List.of(OptionalInt.of(0)),
                        new ScanBody(List.of(h,s),List.of(x),List.of(invariant),List.of(h.add(x).add(invariant),s.add(x.sumToShape(Shape.of(2)))),List.of()),reverse);
                var compiled = io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest.compile(scan.finalCarry(),Optional.empty(),false,new CpuCapabilityProvider());
                var analysis = new CpuPartitionPreparer().analyze(io.github.pho001.synaptik.prepare.GraphPreparation.project(
                        compiled,compiled.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT));
                var unit = analysis.plan().units().stream().filter(u -> u.runtimeFacts().generalScan().isPresent()).findFirst().orElseThrow();
                var facts = unit.runtimeFacts().generalScan().orElseThrow(); var ir = facts.ir();
                assertEquals(ir.actualReads().get(1),ir.actualReads().get(3),"carry and invariant genuinely share the SAME actual read slot");
                assertEquals(1,ir.stages().orElseThrow().carryPorts().get(1)); assertTrue(ir.stages().orElseThrow().carryOffsets().get(1) < 0);
                var generated = CpuGeneralScanGeneratedKernelTest.generated(facts);
                var ids = new HashMap<TensorId,io.github.pho001.synaptik.model.graph.ValueId>();
                compiled.constants().bindableInputBindings().forEach(i -> ids.put(i.tensorId(),i.valueId()));
                for (int batch : new int[]{0,1,7}) try (var arena = Arena.ofConfined()) {
                    var storage = new HashMap<io.github.pho001.synaptik.model.graph.ValueId,MemorySegment>();
                    for (var requirement : analysis.plan().bufferDeclarations()) storage.put(requirement.valueId(),
                            arena.allocate(requirement.size().byteSize(batch),requirement.byteAlignment()));
                    var ih = storage.get(ids.get(initial.id())); var xs = storage.get(ids.get(sequence.id()));
                    var splat = storage.get(ids.get(sharedActual.id())); var ls = storage.get(ids.get(lengths.id()));
                    assertTrue(ir.scalarReads().get(ir.actualReads().get(1)));
                    assertEquals(type.byteWidth(),facts.batchBindings().get(ir.actualReads().get(1)).bind(batch).referencedElementSpan()*type.byteWidth());
                    scalarPut(type,splat,0,1.25);
                    for (long i = 1; i < splat.byteSize()/type.byteWidth(); i++) scalarPut(type,splat,i,Double.NaN);
                    for (long i = 0; i < 2L*batch; i++) scalarPut(type,ih,i,.125*(i%5));
                    for (long i = 0; i < 2L*time*batch; i++) scalarPut(type,xs,i,.25+.125*(i%5));
                    for (int r = 0; r < batch; r++) ls.setAtIndex(ValueLayout.JAVA_LONG,r,Math.min(r%4,time));
                    var args = new ArrayList<MemorySegment>(); unit.boundaryValues().forEach(id -> args.add(storage.get(id)));
                    args.add(arena.allocate(facts.geometry().workspaceSize().byteSize(batch),type.byteWidth()));
                    assertSame(splat,args.get(ir.actualReads().get(1))); assertSame(splat,args.get(ir.actualReads().get(3)));
                    long[] pack = facts.pack(batch);
                    generated.entryPoint().asSpreader(0,MemorySegment[].class,args.size()).invokeExact(args.toArray(MemorySegment[]::new),pack,0L,pack[1]);
                    double[] expected = new double[2*batch], shared = {1.25,1.25};
                    for (int i = 0; i < expected.length; i++) expected[i] = scalarGet(type,ih,i);
                    for (int step = 0; step < time; step++) {
                        int t = reverse ? time-1-step : step; boolean active = false; double[] contribution = new double[2];
                        for (int r = 0; r < batch; r++) if (t < ls.getAtIndex(ValueLayout.JAVA_LONG,r)) {
                            active = true;
                            for (int c = 0; c < 2; c++) {
                                double value = scalarGet(type,xs,(t*batch+r)*2L+c);
                                expected[r*2+c] = rounded(type,rounded(type,expected[r*2+c]+value)+1.25);
                                contribution[c] = batch == 1 ? value : rounded(type,contribution[c]+value);
                            }
                        }
                        if (active) for (int c = 0; c < 2; c++) shared[c] = rounded(type,shared[c]+contribution[c]);
                    }
                    int first = ir.scalarReads().size();
                    for (int i = 0; i < expected.length; i++) assertEquals(expected[i],scalarGet(type,args.get(first),i),"invariant stays at initial1.25 while carry evolves");
                    for (int c = 0; c < 2; c++) assertEquals(shared[c],scalarGet(type,args.get(first+1),c));
                    assertEquals(1.25,scalarGet(type,splat,0),"borrowed aliased read is not modified");
                }
            }
    }


    /**
     * Qualifies runtime T and public seeds against independently compiled fixed-T
     * actual Compiler reverse siblings. The same artifact runs every B and length pattern.
     * Poisoned padding tests skipping before row/history/seed reads, not masking.
     * @throws Exception if fixture construction or generated execution fails
     */
    @Test void cpu0045EquivalentJavaActualReverseRuntimeOperandsAndEmptyDomains() throws Exception {
        int cases = 0;
        for (int time : new int[]{0,1,4}) for (boolean reverse : List.of(false,true)) {
            var family = new CpuSymbolicLengthScanFixture.Family(DataType.FLOAT32,reverse,time);
            for (int batch : new int[]{0,1,7,32}) for (int pattern = 0; pattern < 3; pattern++)
                try (var f = new CpuSymbolicLengthScanFixture(family,batch)) {
                    for (int row = 0; row < batch; row++) f.lengths.setAtIndex(ValueLayout.JAVA_LONG,row,
                            pattern == 0 ? 0 : pattern == 1 ? row%(time+1) : time);
                    for (int t = 0; t < time; t++) for (int row = 0; row < batch; row++)
                        if (t >= f.lengths.getAtIndex(ValueLayout.JAVA_LONG,row)) {
                            for (int k = 0; k < 3; k++) f.put(f.x,(t*batch+row)*3L+k,Double.NaN);
                            for (int k = 0; k < 2; k++)
                                f.es.setAtIndex(ValueLayout.JAVA_INT,(t*batch+row)*2L+k,0x7fc12345);
                        }
                    for (int variant = 0; variant < 3; variant++) {
                        f.fill(f.fs,new double[]{.1,.125,-.75}[variant]);
                        f.fill(f.es,new double[]{.05,-.25,1.5}[variant]);
                        for (int t = 0; t < time; t++) for (int row = 0; row < batch; row++)
                            if (t >= f.lengths.getAtIndex(ValueLayout.JAVA_LONG,row))
                                for (int k = 0; k < 2; k++)
                                    f.es.setAtIndex(ValueLayout.JAVA_INT,(t*batch+row)*2L+k,0x7fc12345);
                        f.execute(); f.oracleForward(); f.oracleReverse();
                        assertEquivalentOutputsCpu0045(f); cases++;
                        for (int t = 0; t < time; t++) for (int row = 0; row < batch; row++)
                            if (t >= f.lengths.getAtIndex(ValueLayout.JAVA_LONG,row)) for (int k = 0; k < 3; k++)
                                assertEquals(0,f.actual.get(3).getAtIndex(ValueLayout.JAVA_INT,(t*batch+row)*3L+k),"raw positive-zero dX padding");
                        for (int row = 0; row < batch; row++) if (f.lengths.getAtIndex(ValueLayout.JAVA_LONG,row) == 0)
                            for (int k = 0; k < 2; k++) assertEquals(f.fs.getAtIndex(ValueLayout.JAVA_INT,row*2L+k),
                                    f.actual.get(2).getAtIndex(ValueLayout.JAVA_INT,row*2L+k),"raw inactive cotangent carry");
                    }
                }
        }
        assertEquals(216,cases);
    }

    /**
     * Exercises raw public seeds and independently computed special-value derivatives.
     * Current derivative constants and saved histories are not injectable boundary formals.
     */
    @Test void cpu0045PublicSeedSpecialsAndInactivePrefixes() {
        for (boolean reverse : List.of(false,true)) {
            var singleton = new CpuSymbolicLengthScanFixture.Family(DataType.FLOAT32,reverse,1);
            for (int batch : new int[]{1,7}) try (var f = new CpuSymbolicLengthScanFixture(singleton,batch)) {
                for (var segment : List.of(f.initial,f.x,f.bias)) segment.fill((byte)0);
                f.lengths.fill((byte)0); f.lengths.setAtIndex(ValueLayout.JAVA_LONG,0,1);
                f.fs.fill((byte)0); f.es.fill((byte)0);
                for (int k = 0; k < 2; k++) {
                    f.fs.setAtIndex(ValueLayout.JAVA_INT,k,0x80000000);
                    f.es.setAtIndex(ValueLayout.JAVA_INT,k,0x80000000);
                }
                f.execute(); f.oracleForward(); f.oracleReverse(); assertEquivalentOutputsCpu0045(f);
                for (int bits : new int[]{0x7fc12345,0x7f800000,0xff800000,0x00000001,0x80000001}) {
                    f.fs.setAtIndex(ValueLayout.JAVA_INT,0,bits);
                    f.execute(); f.oracleForward(); f.oracleReverse(); assertEquivalentOutputsCpu0045(f);
                }
                for (int bits : new int[]{0x00000000,0x80000000,0x7fc12345,0x7f800000,0xff800000,0x00000001,0x80000001}) {
                    f.fill(f.fs,.1); f.fill(f.es,.05);
                    for (long i = 0; i < f.fs.byteSize()/4; i++) f.fs.setAtIndex(ValueLayout.JAVA_INT,i,bits);
                    f.execute(); f.oracleForward(); f.oracleReverse(); assertEquivalentOutputsCpu0045(f);
                }
            }
            for (int time : new int[]{0,4}) {
                var family = new CpuSymbolicLengthScanFixture.Family(DataType.FLOAT32,reverse,time);
                for (int batch : new int[]{0,1,7,32}) for (int bits : new int[]{0x80000000,0x7fc12345,0x3e123456})
                    try (var f = new CpuSymbolicLengthScanFixture(family,batch)) {
                        f.lengths.fill((byte)0);
                        for (int row = 0; row < batch; row++) for (int k = 0; k < 2; k++)
                            f.fs.setAtIndex(ValueLayout.JAVA_INT,row*2L+k,bits);
                        f.execute();
                        f.oracleForward(); f.oracleReverse();
                        assertEquivalentOutputsCpu0045(f);
                        for (int port : new int[]{2,4,5,6}) for (long i = 0; i < f.actual.get(port).byteSize()/4; i++)
                            assertEquals(port == 2 ? bits : 0,f.actual.get(port).getAtIndex(ValueLayout.JAVA_INT,i),"raw inactive seed or positive-zero shared gradient");
                    }
            }
        }
    }

    /**
     * Checks all five reverse publications at represented bits, allowing only NaN payload variation.
     * @param fixture live F32 fixture with latest generated and independent Java results
     * @throws AssertionError if any represented value or non-NaN bit differs
     */
    private static void assertEquivalentOutputsCpu0045(CpuSymbolicLengthScanFixture fixture) {
        for (int port = 2; port < 7; port++) for (long i = 0; i < fixture.actual.get(port).byteSize()/4; i++) {
            int expected = fixture.expected.get(port).getAtIndex(ValueLayout.JAVA_INT,i);
            int actual = fixture.actual.get(port).getAtIndex(ValueLayout.JAVA_INT,i);
            if (Float.isNaN(Float.intBitsToFloat(expected))) assertTrue(Float.isNaN(Float.intBitsToFloat(actual)),"NaN class");
            else assertEquals(expected,actual,"port="+port+" cell="+i);
        }
    }



    /**
     * Proves generation-only product sharing uses complete symbolic coefficients and a fresh
     * dominance lifetime, not equality at B=1 or reuse after the original-time local changes.
     */
    @Test void cpu0044RepairProductSharingRejectsUnequalCoefficientsAndExpiredLifetime() {
        var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.symbolicLengthQuery(DataType.FLOAT32,3,false));
        var spec = CpuGeneralScanGeneratedKernelTest.specialization(facts);
        byte[] bytes = ClassFile.of().build(java.lang.constant.ClassDesc.of("io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.RepairProductProof"),builder -> {
            builder.withVersion(ClassFile.JAVA_26_VERSION,0);
            builder.withMethod("invoke",java.lang.constant.MethodTypeDesc.ofDescriptor(spec.entryType().descriptorString()),
                    java.lang.reflect.AccessFlag.PUBLIC.mask()|java.lang.reflect.AccessFlag.STATIC.mask(),method -> method.withCode(code -> {
                        CpuCarrierEmitter.prepareSegmentLayouts(code,spec.boundaryDataTypes(),spec.carrierPattern(),true,true);
                        var state = new CpuGeneralScanEmitter.Shaped(code,spec,facts.ir());
                        state.bindSymbolicGeometry();
                        code.loadConstant(2L).lstore(state.time);
                        int fixed = state.byteFact(0,8,0,1), equal = state.byteFact(0,4,0,2);
                        int atB1Equal = state.byteFact(3,0,8,1), unequal = state.byteFact(0,12,0,1);
                        assertEquals(fixed,equal); assertNotEquals(fixed,atB1Equal);
                        var lifetime = new HashMap<Integer,Integer>();
                        int product = state.lengthTimeProduct(fixed,lifetime);
                        assertEquals(product,state.lengthTimeProduct(equal,lifetime));
                        assertNotEquals(product,state.lengthTimeProduct(atB1Equal,lifetime));
                        assertNotEquals(product,state.lengthTimeProduct(unequal,lifetime));
                        code.loadConstant(1L).lstore(state.time);
                        assertNotEquals(product,state.lengthTimeProduct(fixed,new HashMap<>()));
                        code.return_();
                    }));
        });
        assertTrue(ClassFile.of().verify(bytes).isEmpty());
    }

    /**
     * Retains the loop-domain strides of an uncut first scalar value when a later fused value
     * is materialized. This is a consumption proof, not a per-value materialization test.
     * @throws Throwable if generation or direct typed execution fails
     */
    @Test void cpu0044RepairRegistersUncutFirstFusedStageDomain() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true)) {
            var b = new DynamicDimension("rows"); var row = Shape.ofDimensions(b,new StaticDimension(2));
            var descriptor = new TensorDescriptor(type,row,Optional.empty(),false);
            Tensor h = TensorFactory.create(descriptor), x = TensorFactory.create(descriptor);
            Tensor first = h.add(x), next = first.mul(type == DataType.FLOAT32 ? ScalarValue.float32(2) : ScalarValue.float64(2));
            var scan = Scan.apply(List.of(TensorFactory.create(descriptor)),List.of(TensorFactory.create(new TensorDescriptor(type,
                    Shape.ofDimensions(new StaticDimension(3),b,new StaticDimension(2)),Optional.empty(),false))),List.of(),
                    TensorFactory.create(new TensorDescriptor(DataType.INT64,Shape.ofDimensions(b),Optional.empty(),false)),
                    1,List.of(OptionalInt.of(0)),new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(next)),reverse);
            var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst())));
            var stages = facts.ir().stages().orElseThrow();
            assertTrue(stages.order().stream().anyMatch(s -> s.output() < 0 && s.values().size() > 1
                    && stages.cuts().stream().noneMatch(c -> c.value() == s.values().getFirst())
                    && s.values().stream().skip(1).anyMatch(v -> stages.cuts().stream().anyMatch(c -> c.value() == v))),
                    "regression must have an uncut first value and a later materialized value");
            var entry = CpuGeneralScanGeneratedKernelTest.generated(facts).entryPoint()
                    .asSpreader(0,MemorySegment[].class,facts.ir().boundaryAccesses().size()+1);
            for (int batch : new int[]{0,1,3}) try (var arena = Arena.ofConfined()) {
                var args = repairArguments(facts,batch,arena);
                for (long i = 0; i < args[0].byteSize()/type.byteWidth(); i++) scalarPut(type,args[0],i,.125);
                for (long i = 0; i < args[1].byteSize()/type.byteWidth(); i++) scalarPut(type,args[1],i,.25);
                for (int r = 0; r < batch; r++) args[facts.ir().lengthRead()].setAtIndex(ValueLayout.JAVA_LONG,r,r+1);
                long[] pack = facts.pack(batch); entry.invokeExact(args,pack,0L,pack[1]);
                int output = facts.ir().scalarReads().size();
                for (int r = 0; r < batch; r++) {
                    double expected = .125;
                    for (int t = 0; t <= r; t++) expected = rounded(type,rounded(type,expected+.25)*2);
                    for (int f = 0; f < 2; f++) assertEquals(expected,scalarGet(type,args[output],r*2+f));
                }
            }
        }
    }

    /**
     * Copies actual scalar initializers under initialization domains, independently of time
     * activity. Nonuniform steps prove later carry versions are not cached; inactive copies
     * preserve signed zero and NaN payload bits. Empty features must not read an expired splat.
     * @throws Throwable if a generated entry or a checked raw copy fails
     */
    @Test void cpu0044RepairInitialSplatsRetainRawBitsAndEvolvingCarry() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true))
            for (int time : new int[]{0,3}) for (int features : new int[]{0,1,2}) {
                var b = new DynamicDimension("rows"); var row = Shape.ofDimensions(b,new StaticDimension(features));
                var descriptor = new TensorDescriptor(type,row,Optional.empty(),false);
                Tensor h = TensorFactory.create(descriptor), x = TensorFactory.create(descriptor);
                Tensor old = CpuGeneralScanLoweringTest.tensor(type,false,features);
                var scalar = new TensorDescriptor(type,Shape.of(features),Optional.of(LayoutDescriptor.of(Shape.of(features),new long[]{0},0,true)),false);
                var scan = Scan.apply(List.of(TensorFactory.create(descriptor),TensorFactory.create(scalar)),
                        List.of(TensorFactory.create(new TensorDescriptor(type,Shape.ofDimensions(new StaticDimension(time),b,new StaticDimension(features)),Optional.empty(),false))),
                        List.of(),TensorFactory.create(new TensorDescriptor(DataType.INT64,Shape.ofDimensions(b),Optional.empty(),false)),
                        1,List.of(OptionalInt.of(0)),new ScanBody(List.of(h,old),List.of(x),List.of(),
                                List.of(h.add(x),old.add(x.sumToShape(Shape.of(features)))),List.of()),reverse);
                var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(scan.finalCarry()));
                var ir = facts.ir(); var entry = CpuGeneralScanGeneratedKernelTest.generated(facts).entryPoint()
                        .asSpreader(0,MemorySegment[].class,ir.boundaryAccesses().size()+1);
                for (int batch : new int[]{0,1,3}) for (boolean active : List.of(false,true))
                    for (long bits : type == DataType.FLOAT32 ? new long[]{0x80000000L,0x7fc12345L,0x3e000000L}
                            : new long[]{0x8000000000000000L,0x7ff8123456789abcL,0x3fc0000000000000L}) try (var arena = Arena.ofConfined()) {
                        var args = repairArguments(facts,batch,arena);
                        for (int p = 0; p < 2; p++) {
                            for (long cell = 0; cell < args[p].byteSize()/type.byteWidth(); cell++) {
                                if (type == DataType.FLOAT32) args[p].setAtIndex(ValueLayout.JAVA_INT,cell,(int)bits);
                                else args[p].setAtIndex(ValueLayout.JAVA_LONG,cell,bits);
                            }
                            if (features == 0 || p == 0 && batch == 0) {
                                try (var expired = Arena.ofConfined()) { args[p] = expired.allocate(type.byteWidth(),type.byteWidth()); }
                            }
                        }
                        for (int r = 0; r < batch; r++) args[ir.lengthRead()].setAtIndex(ValueLayout.JAVA_LONG,r,active ? time : 0);
                        for (int t = 0; t < time; t++) for (int r = 0; r < batch; r++) for (int f = 0; f < features; f++)
                            scalarPut(type,args[2],(t*batch+r)*features+f,active ? .25*(1+f+r+t) : Double.NaN);
                        long[] pack = facts.pack(batch); entry.invokeExact(args,pack,0L,pack[1]);
                        double initial = type == DataType.FLOAT32 ? Float.intBitsToFloat((int)bits) : Double.longBitsToDouble(bits);
                        int output = ir.scalarReads().size();
                        for (int c = 0; c < 2; c++) for (int r = 0; r < (c == 0 ? batch : 1); r++) for (int f = 0; f < features; f++) {
                            double expected = initial;
                            if (active && batch > 0) for (int visit = 0; visit < time; visit++) {
                                int t = reverse ? time-1-visit : visit;
                                double contribution = c == 0 ? .25*(1+f+r+t) : 0;
                                if (c == 1) for (int contributor = 0; contributor < batch; contributor++)
                                    contribution = rounded(type,contribution+.25*(1+f+contributor+t));
                                expected = rounded(type,expected+contribution);
                            }
                            long index = (long)r*features+f;
                            assertEquals(expected,scalarGet(type,args[output+c],index),"actual initializer plus ordered evolving carry");
                            if (!active || time == 0 || batch == 0) {
                                long actualBits = type == DataType.FLOAT32 ? Integer.toUnsignedLong(args[output+c].getAtIndex(ValueLayout.JAVA_INT,index))
                                        : args[output+c].getAtIndex(ValueLayout.JAVA_LONG,index);
                                assertEquals(bits,actualBits,"raw inactive/T0/B0 initializer");
                            }
                        }
                    }
            }
    }

    /**
     * Allocates fresh direct carriers and the exact current numeric workspace for a test run.
     * @param facts complete typed lowering facts
     * @param batch nonnegative invocation B
     * @param arena live caller-owned allocation scope
     * @return fresh ordered boundary carriers followed by scratch
     */
    private static MemorySegment[] repairArguments(CpuGeneralScanLowering.Facts facts,int batch,Arena arena) {
        var ir = facts.ir(); var args = new MemorySegment[ir.boundaryAccesses().size()+1];
        for (int p = 0; p < ir.boundaryAccesses().size(); p++)
            args[p] = arena.allocate(facts.batchBindings().get(p).bind(batch).referencedElementSpan()*ir.boundaryType(p).byteWidth(),ir.boundaryType(p).byteWidth());
        args[args.length-1] = arena.allocate(facts.geometry().workspaceSize().byteSize(batch),ir.dataType().byteWidth());
        return args;
    }

    /**
     * Checks independent production-bound public outputs and their numerical results.
     * @throws Throwable if a bound invocation or semantic check fails
     */
    @Test void cpu0044BoundInvocationsKeepIndependentOutputsAndResources() throws Throwable {
        var family = new CpuSymbolicLengthScanFixture.Family(DataType.FLOAT32,false);
        for (int batch : new int[]{7,32}) try (var fixture = new CpuSymbolicLengthScanFixture(family,batch);
                var other = new CpuSymbolicLengthScanFixture(family,batch)) {
            fixture.verify(); other.verify();
            for (int p = 0; p < fixture.actual.size(); p++) {
                assertEquals(fixture.actual.get(p).byteSize(),other.actual.get(p).byteSize());
                assertNotEquals(fixture.actual.get(p).address(),other.actual.get(p).address());
                assertEquals(-1,fixture.actual.get(p).mismatch(other.actual.get(p)),"same public result "+p);
            }
        }
    }
    /**
     * Checks repeated prebound joint execution with poisoned public input padding.
     * All expected publications come from independently computed forward and adjoint histories.
     */
    @Test void cpu0043PreboundJointExecutionPreservesPoisonedPaddingResults() {
        var family = new CpuSymbolicLengthScanFixture.Family(DataType.FLOAT32,false);
        for (int batch : new int[]{32,7}) try (var fixture = new CpuSymbolicLengthScanFixture(family,batch)) {
            fixture.verify();
            var generated = fixture.preboundExecution();
            generated.run(); fixture.oracleForward(); fixture.oracleReverse(); fixture.verifyLatest();
            for (int t = 0; t < 4; t++) for (int row = 0; row < batch; row++) if (t >= row%5) {
                for (int k = 0; k < 3; k++) fixture.put(fixture.x,(t*batch+row)*3+k,Double.NaN);
                for (int k = 0; k < 2; k++) fixture.put(fixture.es,(t*batch+row)*2+k,Double.NaN);
            }
            generated.run(); fixture.oracleForward(); fixture.oracleReverse(); fixture.verifyLatest();
        }
    }

    /**
     * Checks all 24 dtype/traversal/batch cases against independently owned typed Java references.
     */
    @Test void combinedTypedBoundForwardAndActualReverseMatchSpecializedJava() {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true)) {
            var family = new CpuSymbolicLengthScanFixture.Family(type,reverse);
            for (int batch : new int[]{32,7,1,0,3,2}) try (var fixture = new CpuSymbolicLengthScanFixture(family,batch)) {
                if (batch == 1) fixture.lengths.setAtIndex(ValueLayout.JAVA_LONG,0,4);
                fixture.verify();
                for (int t = 0; t < 4; t++) for (int row = 0; row < batch; row++) if (t >= fixture.lengths.getAtIndex(ValueLayout.JAVA_LONG,row)) {
                    for (int k = 0; k < 3; k++) fixture.put(fixture.x,(t*batch+row)*3+k,Double.NaN);
                    for (int k = 0; k < 2; k++) fixture.put(fixture.es,(t*batch+row)*2+k,Double.NaN);
                }
                fixture.verify();
            }
        }
    }

    /**
     * Makes B1 genuinely active and changes public final seeds. Independent typed one-step
     * equations check singleton bias reduction followed by its distinct carry ADD and all
     * actual AD publications; the four-step retry checks evolving recurrent state.
     * @throws Throwable if a generated reverse entry cannot execute
     */
    @Test void cpu0044ActiveSingletonUsesPublicSeedsAndNeverMutableCarry() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true)) {
            var family = new CpuSymbolicLengthScanFixture.Family(type,reverse);
            try (var f = new CpuSymbolicLengthScanFixture(family,1)) {
                f.lengths.setAtIndex(ValueLayout.JAVA_LONG,0,1);
                for (double changed : new double[]{2.5,-.75}) {
                    for (int col = 0; col < 2; col++) f.put(f.fs,col,changed+col*.125);
                    f.execute();
                    f.oracleForward();
                    double[] local = new double[2];
                    for (int col = 0; col < 2; col++) {
                        double y = f.get(f.oracleSaved,col);
                        double seed = rounded(type,f.get(f.fs,col)+f.get(f.es,col));
                        local[col] = rounded(type,seed*rounded(type,1-rounded(type,y*y)));
                        assertEquals(rounded(type,0d+local[col]),f.get(f.actual.get(6),col),"singleton raw SUM then typed carry ADD");
                    }
                    for (int k = 0; k < 3; k++) {
                        double dx = 0;
                        for (int col = 0; col < 2; col++) {
                            dx = rounded(type,dx+rounded(type,local[col]*f.get(f.w,k*2+col)));
                            assertEquals(rounded(type,0d+rounded(type,f.get(f.x,k)*local[col])),f.get(f.actual.get(4),k*2+col));
                        }
                        assertEquals(dx,f.get(f.actual.get(3),k));
                    }
                    for (int k = 0; k < 2; k++) {
                        double dh = 0;
                        for (int col = 0; col < 2; col++) {
                            dh = rounded(type,dh+rounded(type,local[col]*f.get(f.u,k*2+col)));
                            assertEquals(rounded(type,0d+rounded(type,f.get(f.initial,k)*local[col])),f.get(f.actual.get(5),k*2+col));
                        }
                        assertEquals(dh,f.get(f.actual.get(2),k));
                    }
                }
                f.lengths.setAtIndex(ValueLayout.JAVA_LONG,0,4); f.verify(); f.verifyLatest();
            }
        }
    }

    /**
     * Applies the represented type at each independent one-step arithmetic site.
     * @param type exact F32 or F64 producing type
     * @param value unrounded arithmetic value
     * @return represented arithmetic result
     */
    private static double rounded(DataType type,double value) { return type == DataType.FLOAT32 ? (float)value : value; }

    /**
     * Exercises scalar actual reuse below physical admission: inaccessible splats must not be
     * touched for all-zero lengths, T0, B0 or empty features. Active cases use changing non-one
     * values and scalar initial state that evolves into dense carry; poison padding is unread.
     * @throws Throwable if a guarded direct generated entry violates its checked-access contract
     */
    @Test void cpu0044ScalarReuseIsActivityAndDomainDominatedWithMutableCarry() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean reverse : List.of(false,true))
            for (int time : new int[]{0,3}) for (int batch : new int[]{0,1,3}) for (int features : new int[]{0,2}) {
                var b = new DynamicDimension("rows"); var h = new StaticDimension(features);
                var row = Shape.ofDimensions(b,h); var stack = Shape.ofDimensions(new StaticDimension(time),b,h);
                var hd = new TensorDescriptor(type,row,Optional.empty(),false);
                Tensor state = TensorFactory.create(hd), step = TensorFactory.create(hd), weight = CpuGeneralScanLoweringTest.tensor(type,false,features);
                Tensor old = CpuGeneralScanLoweringTest.tensor(type,false,features);
                var body = new ScanBody(List.of(state,old),List.of(step),List.of(weight),List.of(state.mul(weight).add(step),old.add(step.sumToShape(Shape.of(features)))),List.of());
                var scalarWeight = new TensorDescriptor(type,Shape.of(features),Optional.of(LayoutDescriptor.of(Shape.of(features),new long[]{0},0,true)),false);
                var scan = Scan.apply(List.of(TensorFactory.create(hd),TensorFactory.create(scalarWeight)),List.of(TensorFactory.create(new TensorDescriptor(type,stack,Optional.empty(),false))),
                        List.of(TensorFactory.create(scalarWeight)),TensorFactory.create(new TensorDescriptor(DataType.INT64,Shape.ofDimensions(b),Optional.empty(),false)),1,List.of(OptionalInt.of(0)),body,reverse);
                var query = CpuGeneralScanLoweringTest.query(scan.finalCarry());
                var facts = CpuGeneralScanLowering.analyze(query); var ir = facts.ir();
                var entry = CpuGeneralScanGeneratedKernelTest.generated(facts).entryPoint().asSpreader(0,MemorySegment[].class,ir.boundaryAccesses().size()+1);
                for (boolean active : List.of(false,true)) try (var arena = Arena.ofConfined()) {
                    var args = new MemorySegment[ir.boundaryAccesses().size()+1];
                    for (int p = 0; p < ir.boundaryAccesses().size(); p++) {
                        long bytes = facts.batchBindings().get(p).bind(batch).referencedElementSpan()*ir.boundaryType(p).byteWidth();
                        args[p] = arena.allocate(bytes,ir.boundaryType(p).byteWidth());
                    }
                    args[args.length-1] = arena.allocate(facts.geometry().workspaceSize().byteSize(batch),type.byteWidth());
                    int w = ir.actualReads().get(3), lengths = ir.lengthRead();
                    for (int rowIndex = 0; rowIndex < batch; rowIndex++) args[lengths].setAtIndex(ValueLayout.JAVA_LONG,rowIndex,active ? time : 0);
                    for (long i = 0; i < args[0].byteSize()/type.byteWidth(); i++) scalarPut(type,args[0],i,.125);
                    if (args[1].byteSize() > 0) scalarPut(type,args[1],0,.125);
                    for (long i = 0; i < args[2].byteSize()/type.byteWidth(); i++) scalarPut(type,args[2],i,active ? .25 : Double.NaN);
                    boolean consumes = active && time > 0 && batch > 0 && features > 0;
                    if (!consumes) {
                        try (var expired = Arena.ofConfined()) { args[w] = expired.allocate(Math.max(type.byteWidth(),args[w].byteSize()),type.byteWidth()); }
                    }
                    for (double factor : new double[]{.375,-.5}) {
                        if (consumes) scalarPut(type,args[w],0,factor);
                        long[] pack = facts.pack(batch); entry.invokeExact(args,pack,0L,pack[1]);
                        double expected = .125;
                        if (consumes) for (int t = 0; t < time; t++) expected = rounded(type,rounded(type,expected*factor)+.25);
                        for (long i = 0; i < (long)batch*features; i++) assertEquals(expected,scalarGet(type,args[ir.scalarReads().size()],i),"mutable carry uses new version each time");
                        double shared = .125;
                        if (active && batch > 0) for (int t = 0; t < time; t++) {
                            double sum = 0; for (int r = 0; r < batch; r++) sum = rounded(type,sum+.25);
                            shared = rounded(type,shared+sum);
                        }
                        for (int i = 0; i < features; i++) assertEquals(shared,scalarGet(type,args[ir.scalarReads().size()+1],i),"scalar initializer is not invariant evolving carry");
                    }
                }
            }
    }

    /**
     * Writes a typed test value into its borrowed segment.
     * @param type exact F32 or F64 type
     * @param segment non-null live writable carrier
     * @param index represented element index
     * @param value value narrowed only for F32 storage
     */
    private static void scalarPut(DataType type,MemorySegment segment,long index,double value) {
        if (type == DataType.FLOAT64) segment.setAtIndex(ValueLayout.JAVA_DOUBLE,index,value); else segment.setAtIndex(ValueLayout.JAVA_FLOAT,index,(float)value);
    }

    /**
     * Reads a represented typed test value without modifying its carrier.
     * @param type exact F32 or F64 type
     * @param segment non-null live readable carrier
     * @param index represented element index
     * @return represented value widened exactly when stored as F32
     */
    private static double scalarGet(DataType type,MemorySegment segment,long index) {
        return type == DataType.FLOAT64 ? segment.getAtIndex(ValueLayout.JAVA_DOUBLE,index) : segment.getAtIndex(ValueLayout.JAVA_FLOAT,index);
    }



    /** Production admission retains checked borrowed inputs and outputs for both matrix families. */
    @Test void symbolicReverseCarriersRetainCheckedBoundsAndSessions() throws Throwable {
        for (int[] dimensions:new int[][] {{3,4,3},{4,32,32}}) {
            var family=new CpuSymbolicScanTimingTest.PreparedFamily(dimensions[0],dimensions[1],dimensions[2],false);
            var weightId=family.compiled.constants().bindableInputBindings().stream()
                    .filter(binding->binding.tensorId().equals(family.inputs.get(2).id())).findFirst().orElseThrow().valueId();
            var gradientId=family.compiled.publication().gradientBindings().getFirst().valueId();
            for (var id:List.of(weightId,gradientId)) {
                long bytes=id.equals(weightId)?4L*dimensions[1]*dimensions[2]:4L*dimensions[2];
                try (var storage=Arena.ofConfined()) {
                    var shortStorage=storage.allocate(bytes-4,4);
                    assertThrows(RuntimeException.class,()->{
                        try (var consumer=new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuTypedEmptyScanConsumerTest.Consumer(
                                family.compiled,OptionalLong.of(1),Map.of(id,shortStorage))) { consumer.bind(); consumer.execute(); }
                    });
                }
                MemorySegment closed;
                try (var storage=Arena.ofConfined()) { closed=storage.allocate(bytes,4); }
                assertThrows(RuntimeException.class,()->{
                    try (var consumer=new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuTypedEmptyScanConsumerTest.Consumer(
                            family.compiled,OptionalLong.of(1),Map.of(id,closed))) { consumer.bind(); consumer.execute(); }
                });
                try (var storage=Arena.ofConfined();
                        var consumer=new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuTypedEmptyScanConsumerTest.Consumer(
                                family.compiled,OptionalLong.of(1),Map.of(id,storage.allocate(bytes,4)))) {
                    consumer.bind();
                    var failure=new java.util.concurrent.atomic.AtomicReference<Throwable>();
                    var thread=new Thread(()->{ try { consumer.execute(); } catch (Throwable problem) { failure.set(problem); } });
                    thread.start(); thread.join();
                    assertInstanceOf(RuntimeException.class,failure.get(),"confined borrowed storage rejects another thread");
                }
            }
            for (int batch:new int[]{1,0})
                try (var fixture=new CpuSymbolicScanTimingTest.Fixture(family,batch)) { fixture.verify(); }
        }
    }

    @Test void actualReverseMatchesBothActivationsForSpecialPublicInputBits() {
        for (boolean sigmoid:List.of(false,true)) try (var f=new CpuSymbolicScanTimingTest.Fixture(1,1,4,4,sigmoid)) {
            f.x.fill((byte)0); f.w.fill((byte)0);
            float[] inputs={.25f,-0f,Float.intBitsToFloat(0x7fc12345),Float.POSITIVE_INFINITY};
            for (int i=0;i<4;i++) {
                f.initial.setAtIndex(ValueLayout.JAVA_FLOAT,i,inputs[i]);
                f.fs.setAtIndex(ValueLayout.JAVA_FLOAT,i,1);
                f.es.setAtIndex(ValueLayout.JAVA_FLOAT,i,0);
            }
            f.verify();
            for (int i=0;i<4;i++) {
                float y=f.actual.get(1).getAtIndex(ValueLayout.JAVA_FLOAT,i), expected=sigmoid?y*(1-y):1-y*y;
                float actual=f.actual.get(2).getAtIndex(ValueLayout.JAVA_FLOAT,i);
                if (Float.isNaN(expected)) assertTrue(Float.isNaN(actual));
                else assertEquals(Float.floatToRawIntBits(expected),Float.floatToRawIntBits(actual),"public derivative lane="+i);
            }
        }
    }
    @Test void fusedSumToShapeKeepsProducerTailAndSingletonRepresentedIdentity() throws Throwable {
        var b=new DynamicDimension("B"); Tensor h=symbolic(false,b,new StaticDimension(3)), x=symbolic(false,b,new StaticDimension(3)), bias=fixed(false,3);
        Tensor contribution=h.neg(), sum=contribution.sumToShape(Shape.of(1,3)), tail=sum.add(bias);
        var scan=Scan.apply(List.of(symbolic(false,b,new StaticDimension(3))),List.of(symbolic(false,new StaticDimension(1),b,new StaticDimension(3))),List.of(fixed(false,3)),
                new ScanBody(List.of(h),List.of(x),List.of(bias),List.of(h),List.of(tail)),false);
        var facts=CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(scan.outputs()));
        var stage=facts.ir().stages().orElseThrow().order().stream().filter(s->s.reduction()).findFirst().orElseThrow();
        assertTrue(stage.sumToShape()); assertFalse(stage.values().isEmpty()); assertFalse(stage.epilogue().isEmpty());
        assertTrue(facts.ir().stages().orElseThrow().cuts().stream().noneMatch(c->stage.values().contains(c.value())),"single-use contributor is not materialized");
        var generated=CpuGeneralScanGeneratedKernelTest.generated(facts); inspect(generated.classBytes());
        for (int batch:new int[]{0,1,4,7}) {
            float[] input=data(batch*3,.125f), add={-0f,.25f,0f};
            if (batch==1) { input[0]=0f; input[1]=Float.intBitsToFloat(1); input[2]=Float.intBitsToFloat(0x7fc12345); }
            float[] expected=new float[3];
            for (int col=0;col<3;col++) { float value=batch==1?-input[col]:0f; if (batch!=1) for (int row=0;row<batch;row++) value+=-input[row*3+col]; expected[col]=value+add[col]; }
            var actual=execute(facts,generated,batch,List.of(input,new float[batch*3],add)).getFirst();
            for (int col=0;col<3;col++) assertEquals(Float.floatToRawIntBits(expected[col]),Float.floatToRawIntBits(actual[col]),"B="+batch+" col="+col);
        }
    }

    @Test void specializedCleanJavaMatchesActualCompilerReverseWithPreallocatedSegments() {
        for (int[] c : new int[][] {{3,0,4,3},{3,1,4,3},{3,40,4,3},{3,64,4,3},{4,64,32,32},
                {3,3,3,3},{3,4,4,4},{3,5,2,5},{0,0,4,3},{0,4,4,3},{3,4,0,3}})
            try (var fixture = new CpuSymbolicScanTimingTest.Fixture(c[0],c[1],c[2],c[3])) { fixture.verify(); }
    }


    /** The final int increment is safe at MAX, and packing rejects MAX+1 before any invocation. */
    @Test void symbolicContractionIntEndpointIsProvedEvenWithNonemptySharedOutputAtB0() throws Throwable {
        var b=new DynamicDimension("B"); Tensor h=fixed(false,1,1), x=symbolic(false,b,new StaticDimension(1));
        Tensor next=h.add(x.permute(1,0).matmul(x));
        var scan=Scan.apply(List.of(fixed(false,1,1)),List.of(symbolic(false,new StaticDimension(1),b,new StaticDimension(1))),List.of(),
                new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(next)),false);
        var roots=new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
        var facts=CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(roots));
        var generated=CpuGeneralScanGeneratedKernelTest.generated(facts); inspect(generated.classBytes());
        assertEquals(1,intContractions(generated.classBytes()));
        assertDoesNotThrow(()->facts.pack(Integer.MAX_VALUE));
        assertThrows(ArithmeticException.class,()->facts.pack(Integer.MAX_VALUE+1L));
        assertEquals(Integer.MAX_VALUE,Math.addExact(Integer.MAX_VALUE-1,1));
        for (int batch:new int[]{0,1,3}) {
            float[] input=data(batch,.125f); float expected=.25f, sum=0f;
            for (float v:input) sum+=v*v; expected+=sum;
            for (var result:execute(facts,generated,batch,List.of(new float[]{.25f},input))) assertArrayEquals(new float[]{expected},result);
        }
    }

    /** Shared nonempty matrix work must survive empty row domains and scalar-to-dense binding. */
    @Test void fixedSharedScalarCarryBecomesDenseInBothTraversalsBesideEmptyBatchWork() throws Throwable {
        for (boolean reverse:List.of(false,true)) for (int time:new int[]{0,1,3}) {
            var b=new DynamicDimension("B");
            Tensor rows=symbolic(false,b,new StaticDimension(3)), shared=fixed(false,2,3);
            Tensor x=symbolic(false,b,new StaticDimension(3)), w=fixed(false,3,3);
            Tensor nextRows=rows.add(x), nextShared=shared.matmul(w).add(x.sum(0,false));
            var body=new ScanBody(List.of(rows,shared),List.of(x),List.of(w),List.of(nextRows,nextShared),List.of(nextRows,nextShared,shared));
            var scan=Scan.apply(List.of(symbolic(false,b,new StaticDimension(3)),fixed(false).expand(Shape.of(2,3))),
                    List.of(symbolic(false,new StaticDimension(time),b,new StaticDimension(3))),List.of(fixed(false,3,3)),body,reverse);
            var roots=new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            var facts=CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(roots));
            var generated=CpuGeneralScanGeneratedKernelTest.generated(facts); inspect(generated.classBytes());
            assertTrue(facts.ir().scalarReads().get(1));
            for (int batch:new int[]{64,40,1,64,0,3}) {
                float[] initial=data(batch*3,.015625f), sequence=data(time*batch*3,.03125f), weight=data(9,.0625f);
                var actual=execute(facts,generated,batch,List.of(initial,new float[]{.25f},sequence,weight));
                float[] row=initial.clone(), oldShared=new float[6]; Arrays.fill(oldShared,.25f);
                float[] rowHistory=new float[time*batch*3], sharedHistory=new float[time*6], previousHistory=new float[time*6];
                for (int step=0;step<time;step++) {
                    int t=reverse?time-1-step:step;
                    System.arraycopy(oldShared,0,previousHistory,t*6,6);
                    float[] sum=new float[3], next=new float[6];
                    for (int r=0;r<batch;r++) for (int col=0;col<3;col++) {
                        float value=sequence[(t*batch+r)*3+col]; sum[col]+=value; row[r*3+col]+=value;
                    }
                    for (int r=0;r<2;r++) for (int col=0;col<3;col++) {
                        float product=0; for (int k=0;k<3;k++) product+=oldShared[r*3+k]*weight[k*3+col];
                        next[r*3+col]=product+sum[col];
                    }
                    oldShared=next; System.arraycopy(row,0,rowHistory,t*batch*3,row.length);
                    System.arraycopy(next,0,sharedHistory,t*6,6);
                }
                assertArrayEquals(row,actual.get(0)); assertArrayEquals(oldShared,actual.get(1));
                assertArrayEquals(rowHistory,actual.get(2)); assertArrayEquals(sharedHistory,actual.get(3)); assertArrayEquals(previousHistory,actual.get(4));
            }
        }
    }

    /** Actual opposite-traversal AD retains every published forward and gradient value. */
    @Test void actualCompilerReverseInBothTraversalsMatchesAllFiveOutputs() throws Throwable {
        for (boolean reverse:List.of(false,true)) for (int time:new int[]{0,3}) for (int[] widths:new int[][]{{4,3},{3,3}}) {
            int f=widths[0], h=widths[1]; var b=new DynamicDimension("B");
            Tensor sh=symbolic(true,b,new StaticDimension(h)), sx=symbolic(true,b,new StaticDimension(f)), sw=fixed(true,f,h);
            Tensor next=sx.matmul(sw).add(sh).tanh();
            Tensor initial=symbolic(true,b,new StaticDimension(h)), sequence=symbolic(true,new StaticDimension(time),b,new StaticDimension(f)), w=fixed(true,f,h);
            var scan=Scan.apply(List.of(initial),List.of(sequence),List.of(w),new ScanBody(List.of(sh),List.of(sx),List.of(sw),List.of(next),List.of(next)),reverse);
            var roots=List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
            Tensor fs=symbolic(false,b,new StaticDimension(h)), es=symbolic(false,new StaticDimension(time),b,new StaticDimension(h));
            var request=new io.github.pho001.synaptik.compiler.FunctionalGradientRequest(List.of(new io.github.pho001.synaptik.compiler.FunctionalGradientRequest.Stage(
                    roots.stream().map(r->(io.github.pho001.synaptik.compiler.FunctionalGradientRequest.OutputReference)
                            new io.github.pho001.synaptik.compiler.FunctionalGradientRequest.ForwardTensorReference(r)).toList(),
                    List.of(Optional.of(fs),Optional.of(es)),List.of(initial,sequence,w),false,
                    time == 0 ? io.github.pho001.synaptik.compiler.FunctionalGradientRequest.DisconnectedPolicy.ZERO
                            : io.github.pho001.synaptik.compiler.FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
            var compiled=io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest.compile(roots,Optional.of(request),false,new CpuCapabilityProvider());
            for (int batch:new int[]{64,40,1,64,0,3,4}) try (var consumer =
                    new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuTypedEmptyScanConsumerTest.Consumer(compiled,OptionalLong.of(batch))) {
                float[] ih=data(batch*h,.01f), xs=data(time*batch*f,.02f), ws=data(f*h,.03f), hs=data(batch*h,.1f), ys=data(time*batch*h,.05f);
                var inputs=List.of(initial,sequence,w,fs,es); var values=List.of(ih,xs,ws,hs,ys);
                for (int i=0;i<inputs.size();i++) {
                    float[] source=values.get(i); double[] numbers=new double[source.length];
                    for (int cell=0;cell<source.length;cell++) numbers[cell]=source[cell];
                    consumer.numbers(inputs.get(i),numbers);
                }
                consumer.bind(); consumer.execute();
                float[] carry=ih.clone(), history=new float[time*batch*h], dh=hs.clone(), dx=new float[time*batch*f], dw=new float[f*h];
                for (int step=0;step<time;step++) {
                    int t=reverse?time-1-step:step;
                    for (int row=0;row<batch;row++) for (int col=0;col<h;col++) {
                        float sum=0; for (int k=0;k<f;k++) sum+=xs[(t*batch+row)*f+k]*ws[k*h+col];
                        carry[row*h+col]=(float)StrictMath.tanh(sum+carry[row*h+col]); history[(t*batch+row)*h+col]=carry[row*h+col];
                    }
                }
                for (int step=0;step<time;step++) {
                    int t=reverse?step:time-1-step;
                    for (int i=0;i<batch*h;i++) { float y=history[t*batch*h+i]; dh[i]=(dh[i]+ys[t*batch*h+i])*(1-y*y); }
                    for (int k=0;k<f;k++) for (int col=0;col<h;col++) {
                        float sum=0; for (int row=0;row<batch;row++) sum+=xs[(t*batch+row)*f+k]*dh[row*h+col]; dw[k*h+col]+=sum;
                    }
                    for (int row=0;row<batch;row++) for (int k=0;k<f;k++) {
                        float sum=0; for (int col=0;col<h;col++) sum+=dh[row*h+col]*ws[k*h+col]; dx[(t*batch+row)*f+k]=sum;
                    }
                }
                var expected=List.of(carry,history,dh,dx,dw);
                var actual=List.of(consumer.forward(0),consumer.forward(1),consumer.gradient(0),consumer.gradient(1),consumer.gradient(2));
                for (int p=0;p<5;p++) {
                    float[] result=new float[actual.get(p).length];
                    for (int i=0;i<result.length;i++) result[i]=(float)actual.get(p)[i];
                    assertArrayEquals(expected.get(p),result,"reverse="+reverse+" B="+batch+" T="+time+" port="+p);
                }
            }
        }
    }

    /**
     * Counts actual full-K loops by their two reads/one multiply-add and no cell publication.
     * Each backedge must use IINC and an int top test, as in the unchanged literal Java oracle.
     * @param bytes complete unmodified generated Class-File
     * @return number of proved int-controlled contraction loops
     */
    private static int intContractions(byte[] bytes) {
        var code=(java.lang.classfile.attribute.CodeAttribute)ClassFile.of().parse(bytes).methods().getFirst().code().orElseThrow();
        var at=new TreeMap<Integer,Instruction>(); int bci=0, count=0;
        for (var e:code) if (e instanceof Instruction i) { at.put(bci,i); bci+=i.sizeInBytes(); }
        for (var entry:at.entrySet()) if (entry.getValue() instanceof BranchInstruction branch && branch.opcode()==Opcode.GOTO) {
            int target=code.labelToBci(branch.target()); if (target>=entry.getKey()) continue;
            var loop=List.copyOf(at.subMap(target,true,entry.getKey(),false).values());
            long gets=loop.stream().filter(i->i instanceof InvokeInstruction call && call.name().equalsString("get")).count();
            boolean store=loop.stream().anyMatch(i->i instanceof InvokeInstruction call && call.name().equalsString("set"));
            if (gets!=2 || store || loop.stream().filter(i->i.opcode()==Opcode.FMUL).count()!=1
                    || loop.stream().filter(i->i.opcode()==Opcode.FADD).count()!=1) continue;
            assertEquals(Opcode.IINC,loop.getLast().opcode());
            assertTrue(loop.stream().anyMatch(i->i.opcode()==Opcode.IF_ICMPGE));
            assertFalse(loop.stream().anyMatch(i->i.opcode()==Opcode.LCMP)); count++;
        }
        return count;
    }

    /**
     * Extracts actual executable instructions for packing and loop-shape assertions.
     * @param bytes generated Class-File, not mutated
     * @return immutable encounter-ordered entry instructions
     */
    private static List<Instruction> instructions(byte[] bytes) {
        return ClassFile.of().parse(bytes).methods().getFirst().code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
    }
    @Test void acceptedStaticV4ControlsRemainExactAndApprovedMatmulV5RetainsCanonicalProjection() throws Exception {
        try (var stream = getClass().getResourceAsStream("symbolic-scan-static-baseline.tsv")) {
            assertNotNull(stream);
            for (String line : new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8).lines().toList()) {
                if (line.startsWith("#") || line.isBlank()) continue;
                String[] expected = line.split("\\t");
                var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.denseReadFixture(DataType.valueOf(expected[0]),Integer.parseInt(expected[1]),3,false));
                var specialization = CpuGeneralScanGeneratedKernelTest.specialization(facts);
                assertEquals(expected[2],facts.ir().structuralKey());
                if (Integer.parseInt(expected[1]) == 2) {
                    assertNotEquals(expected[3],specialization.structuralKey());
                    assertEquals(expected[4],sha(new String(specialization.classIdentityBytes(),java.nio.charset.StandardCharsets.US_ASCII)
                            .replace("|scan-matmul-v5=","|scan-matmul-v4=").getBytes(java.nio.charset.StandardCharsets.US_ASCII)));
                    assertEquals(expected[5],sha(new String(specialization.compatibilityBytes(),java.nio.charset.StandardCharsets.US_ASCII)
                            .replace("|scan-matmul-v5=","|scan-matmul-v4=").getBytes(java.nio.charset.StandardCharsets.US_ASCII)));
                    assertNotEquals(expected[6],sha(CpuGeneralScanGeneratedKernelTest.generated(facts).classBytes()));
                } else {
                    assertEquals(expected[3],specialization.structuralKey());
                    assertEquals(expected[4],sha(specialization.classIdentityBytes()));
                    assertEquals(expected[5],sha(specialization.compatibilityBytes()));
                    assertEquals(expected[6],sha(CpuGeneralScanGeneratedKernelTest.generated(facts).classBytes()));
                }
            }
        }
        var shape = Shape.of(1,2,3); var target = Shape.of(2,3);
        var source = new TensorDescriptor(DataType.FLOAT32,shape,Optional.of(LayoutDescriptor.contiguous(shape)),false);
        var output = new TensorDescriptor(DataType.FLOAT32,target,Optional.of(LayoutDescriptor.contiguous(target)),false);
        var op = new io.github.pho001.synaptik.model.operation.Operation(io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.SUM,
                new io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs(target));
        var route = new CpuPartitionPreparer().analyze(CpuNonAffineMovementLoweringTest.context(op,List.of(0),List.of(source),output)).plan().units().getFirst().portablePlan();
        assertFalse(route.specialization().ordinaryBatch(),"accepted static rank3 singleton AD retains its original non-batch emitter");
        assertEquals("1a8e23310e76382bbd18d0e82348746a8b281ae17013c970bb1265d2ece6e665",route.specialization().structuralKey());
        assertEquals("ca6158ed406d972303ee98e2a819b854ee8f9c7b406ed1503a55008f164e6321",sha(route.specialization().compatibilityBytes()));
        assertEquals("ad2ab60944d4e9bbca278d9eab8c9e8e87b9cf247a7f0ddd0383dde42253a3c8",sha(new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(),route.kernelIr())));
    }

    @Test void staticSymbolicAndChangedTimeRecipeArtifactsCannotCrossDecode() {
        var b = new DynamicDimension("B"); Tensor h = symbolic(false,b,new StaticDimension(3)), x = symbolic(false,b,new StaticDimension(3));
        var scan = Scan.apply(List.of(symbolic(false,b,new StaticDimension(3))),List.of(symbolic(false,new StaticDimension(3),b,new StaticDimension(3))),List.of(),
                new ScanBody(List.of(h),List.of(x),List.of(),List.of(h.add(x)),List.of()),false);
        var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(scan.finalCarry())); var ir = facts.ir(); var p = ir.stages().orElseThrow();
        var stride = p.sequenceStrides().getFirst();
        var changed = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr.Stages(p.views(),p.order(),p.cuts(),p.carryOffsets(),p.carryPorts(),p.scratchBytes(),
                p.carryOffsetFactors(),p.scratchBytesPerBatch(),true,List.of(new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(stride.constantBytes()+4,stride.bytesPerBatch(),4)));
        var other = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr(ir.arithmetic(),ir.formalOrdinals(),ir.actualReads(),ir.scalarReads(),ir.nextOrdinals(),ir.portOrdinals(),
                ir.demandedPorts(),ir.carries(),ir.boundaryAccesses(),ir.carryCount(),ir.sequenceCount(),ir.reverse(),ir.scratchBanks(),Optional.of(changed));
        var altered = new CpuGeneralScanLowering.Facts(other,facts.geometry(),List.of(),List.of(),facts.batchBindings());
        var original = CpuGeneralScanGeneratedKernelTest.specialization(facts); var replacement = CpuGeneralScanGeneratedKernelTest.specialization(altered);
        assertFalse(Arrays.equals(original.classIdentityBytes(),replacement.classIdentityBytes()));
        assertFalse(Arrays.equals(original.compatibilityBytes(),replacement.compatibilityBytes()));
        var bytes = CpuGeneralScanGeneratedKernelTest.generated(facts).classBytes(); var generator = new CpuClassFileKernelGenerator();
        assertThrows(IllegalArgumentException.class,() -> generator.defineClassBytes(replacement, altered.ir().encodedKernelIr(), bytes));
        var fixed = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.denseReadFixture(DataType.FLOAT32,0,3,false));
        assertThrows(IllegalArgumentException.class,() -> generator.defineClassBytes(CpuGeneralScanGeneratedKernelTest.specialization(fixed), fixed.ir().encodedKernelIr(), bytes));
        assertThrows(IllegalArgumentException.class,() -> generator.defineClassBytes(original, facts.ir().encodedKernelIr(), CpuGeneralScanGeneratedKernelTest.generated(fixed).classBytes()));
    }


    @Test void namedTransposeSharedFramesAndStructuralZeroWorkspaceAreNotSampled() throws Throwable {
        for (int time : new int[] {0,3}) {
            var b = new DynamicDimension("B");
            Tensor h = symbolic(false,b,new StaticDimension(3)), x = symbolic(false,b,new StaticDimension(4));
            Tensor w = fixed(false,4,3), u = fixed(false,3,3);
            Tensor shared = w.matmul(u), next = h.add(x.matmul(shared));
            var body = new ScanBody(List.of(h),List.of(x),List.of(w,u),List.of(next),List.of(next,shared,x.permute(1,0).sum(1,false)));
            var scan = Scan.apply(List.of(symbolic(false,b,new StaticDimension(3))),
                    List.of(symbolic(false,new StaticDimension(time),b,new StaticDimension(4))),List.of(fixed(false,4,3),fixed(false,3,3)),body,false);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            var query = CpuGeneralScanLoweringTest.query(roots); io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertCurrentSupport(new CpuCapabilityProvider(), query);
            var facts = CpuGeneralScanLowering.analyze(query); var ir = facts.ir(); var stages = ir.stages().orElseThrow();
            assertTrue(ir.symbolic()); assertEquals(1,ir.scratchBanks());
            assertTrue(ir.encodedKernelIr().familyIdentity().endsWith(":symbolic-batch-v1"));
            assertTrue(ir.bakedSpecializationIdentity().startsWith("|scan-symbolic-batch-v3="));
            var generated = CpuGeneralScanGeneratedKernelTest.generated(facts); inspect(generated.classBytes());
            var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(query));
            assertTrue(analysis.plan().units().getFirst().runtimeFacts().workspaceDeclaration().isPresent());
            assertEquals(time*4*3*4L,facts.batchBindings().get(6).bind(7).referencedElementSpan()*4L,"shared [T,F,H] is not multiplied by B");
            for (int batch : new int[] {64,40,1,64,0,3,4}) {
                float[] initial = data(batch*3,.01f), sequence = data(time*batch*4,.02f), weights = data(12,.03f), recurrent = data(9,.04f);
                var actual = execute(facts,generated,batch,List.of(initial,sequence,weights,recurrent));
                float[] sharedValues = new float[12];
                for (int row = 0; row < 4; row++) for (int col = 0; col < 3; col++) for (int k = 0; k < 3; k++)
                    sharedValues[row*3+col] += weights[row*3+k]*recurrent[k*3+col];
                float[] carry = initial.clone(), history = new float[time*batch*3];
                for (int t = 0; t < time; t++) for (int row = 0; row < batch; row++) for (int col = 0; col < 3; col++) {
                    float sum = 0; for (int k = 0; k < 4; k++) sum += sequence[(t*batch+row)*4+k]*sharedValues[k*3+col];
                    carry[row*3+col] += sum; history[(t*batch+row)*3+col] = carry[row*3+col];
                }
                assertArrayEquals(carry,actual.get(0)); assertArrayEquals(history,actual.get(1));
                for (int t = 0; t < time; t++) assertArrayEquals(sharedValues,Arrays.copyOfRange(actual.get(2),t*12,(t+1)*12));
                for (int t = 0; t < time; t++) for (int k = 0; k < 4; k++) {
                    float sum = 0; for (int row = 0; row < batch; row++) sum += sequence[(t*batch+row)*4+k];
                    assertEquals(sum,actual.get(3)[t*4+k]);
                }
                assertTrue(generated.entryPoint().type().parameterCount() > 0,"same artifact is retained across invocation B, not regenerated");
            }
        }
    }

    @Test void actualByteHooksRetainReadAliasesAndRejectWritableCapacityTailsBeforeWrites() {
        var b = new DynamicDimension("B"); Tensor h = symbolic(false,b,new StaticDimension(3)), x = symbolic(false,b,new StaticDimension(3));
        var scan = Scan.apply(List.of(symbolic(false,b,new StaticDimension(3))),List.of(symbolic(false,new StaticDimension(3),b,new StaticDimension(3))),List.of(),
                new ScanBody(List.of(h),List.of(x),List.of(),List.of(h.add(x)),List.of()),false);
        var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(CpuGeneralScanLoweringTest.query(scan.finalCarry())));
        for (int batch : new int[] {0,1,40}) for (int mode = 0; mode < 3; mode++) {
            try (var f = new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(analysis,mode,true,true,Map.of(),OptionalLong.of(batch))) {
                f.fill(0,.25f); f.fill(1,.125f); f.executable.bind(f.state).execute();
                for (float value : f.floats(2)) assertEquals(.625f,value);
            }
        }
        try (var arena = Arena.ofConfined()) {
            var memory = arena.allocate(256,4); memory.fill((byte)0x5a);
            try (var f = new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(analysis,1,true,false,
                    Map.of(new io.github.pho001.synaptik.model.graph.ValueId(0),memory.asSlice(0,12),
                            new io.github.pho001.synaptik.model.graph.ValueId(1),memory.asSlice(0,36)),OptionalLong.of(1))) {
                f.fill(1,.125f); f.executable.bind(f.state).execute();
                assertArrayEquals(new float[] {.5f,.5f,.5f},f.floats(2),"read/read physical aliases remain legal");
            }
            // Logical write prefix [0,12) is disjoint from input [32,44), but its selected
            // capacity [0,48) is not. The child hook accepts capacity then rejects full overlap.
            try (var f = new io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture(analysis,1,false,false,
                    Map.of(new io.github.pho001.synaptik.model.graph.ValueId(0),memory.asSlice(32,12),
                            new io.github.pho001.synaptik.model.graph.ValueId(2),memory.asSlice(0,48)),OptionalLong.of(1))) {
                var child = ((io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedPartitionExecutable)f.executable).children().getFirst();
                byte[] before = memory.toArray(ValueLayout.JAVA_BYTE);
                assertThrows(IllegalArgumentException.class,() -> child.bind(f.state));
                assertArrayEquals(before,memory.toArray(ValueLayout.JAVA_BYTE));
            }
        }
    }

    @Test void rankThreeOuterSelectedAxisProjectionAndConditionalExactScratchAreComplete() throws Throwable {
        for (int mask = 1; mask < 8; mask++) for (boolean mean : List.of(false,true)) {
            var b = new DynamicDimension("B"); Shape shape = Shape.ofDimensions(new StaticDimension(3),b,new StaticDimension(4));
            var axes = new ArrayList<Integer>(); var dimensions = new ArrayList<Dimension>();
            for (int axis = 0; axis < 3; axis++) if ((mask & (1<<axis)) != 0) axes.add(axis); else dimensions.add(shape.dimension(axis));
            Shape target = Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
            var input = new TensorDescriptor(DataType.FLOAT32,shape,Optional.empty(),false);
            var output = new TensorDescriptor(DataType.FLOAT32,target,target.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(target)) : Optional.empty(),false);
            var operation = new io.github.pho001.synaptik.model.operation.Operation(mean
                    ? io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.MEAN
                    : io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.SUM,
                    new io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs(axes,false));
            var route = new CpuPartitionPreparer().analyze(CpuNonAffineMovementLoweringTest.context(operation,List.of(0),List.of(input),output)).plan().units().getFirst().portablePlan();
            var generator = new CpuClassFileKernelGenerator();
            var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
            for (int batch : new int[] {0,1,4,7}) try (var arena = Arena.ofConfined()) {
                var geometry = route.batchAggregate().orElseThrow().bind(batch); float[] source = data(3*batch*4,.125f);
                float[] actual = new float[Math.toIntExact(geometry.outputCount())], expected = new float[actual.length]; double[] sum = new double[actual.length];
                for (int t = 0; t < 3; t++) for (int row = 0; row < batch; row++) for (int f = 0; f < 4; f++) {
                    int[] coordinate = {t,row,f}, extents = {3,batch,4}; int out = 0;
                    for (int axis = 0; axis < 3; axis++) if (!axes.contains(axis)) out = out*extents[axis]+coordinate[axis];
                    sum[out] += source[(t*batch+row)*4+f];
                }
                for (int i = 0; i < expected.length; i++) expected[i] = (float)(mean ? sum[i]/geometry.domainCount() : sum[i]);
                long scratchBytes = route.batchAggregate().orElseThrow().byteSize(batch);
                assertEquals(actual.length == 0 ? 0 : 48,scratchBytes);
                MemorySegment in = MemorySegment.ofArray(source), out = MemorySegment.ofArray(actual), scratch = arena.allocate(scratchBytes,8);
                artifact.entryPoint().invokeExact(in,out,scratch,geometry.pack(new long[2]),0L,geometry.outputCount());
                assertArrayEquals(expected,actual,"axes="+axes+" B="+batch+" mean="+mean);
            }
        }
    }

    @Test void zeroSlopeStillDeclaresWorkspaceAndRequiresBatchAndSingletonSumToShapeIsRaw() throws Throwable {
        var b = new DynamicDimension("B");
        Tensor h = symbolic(false,b,new StaticDimension(0)), x = symbolic(false,b,new StaticDimension(0));
        var scan = Scan.apply(List.of(symbolic(false,b,new StaticDimension(0))),List.of(symbolic(false,new StaticDimension(3),b,new StaticDimension(0))),
                List.of(),new ScanBody(List.of(h),List.of(x),List.of(),List.of(h.add(x)),List.of()),false);
        var analysis = new CpuPartitionPreparer().analyze(CpuGeneralScanPreparationTest.context(CpuGeneralScanLoweringTest.query(scan.finalCarry())));
        var workspace = analysis.plan().units().getFirst().runtimeFacts().workspaceDeclaration().orElseThrow();
        assertTrue(workspace.size().fixedByteSize().isEmpty()); assertEquals(0,workspace.size().byteSize(0)); assertEquals(0,workspace.size().byteSize(64));
        Tensor sh = symbolic(false,b,new StaticDimension(3)), sx = symbolic(false,b,new StaticDimension(3));
        var sum = sh.sumToShape(Shape.of(1,3));
        var s = Scan.apply(List.of(symbolic(false,b,new StaticDimension(3))),List.of(symbolic(false,new StaticDimension(1),b,new StaticDimension(3))),List.of(),
                new ScanBody(List.of(sh),List.of(sx),List.of(),List.of(sh),List.of(sum,sum)),false);
        var facts = CpuGeneralScanLowering.analyze(CpuGeneralScanLoweringTest.query(s.outputs()));
        float[] bits = {Float.intBitsToFloat(0x80000000),Float.intBitsToFloat(1),Float.intBitsToFloat(0x7fc12345)};
        var generated=CpuGeneralScanGeneratedKernelTest.generated(facts);
        for (int batch:new int[]{0,1,4}) {
            float[] input=batch==1?bits:data(batch*3,.125f), expected=new float[3];
            if (batch==1) expected=bits;
            else for (int row=0;row<batch;row++) for (int col=0;col<3;col++) expected[col]+=input[row*3+col];
            var output=execute(facts,generated,batch,List.of(input,new float[batch*3])); assertEquals(2,output.size());
            for (var out:output) for (int i=0;i<3;i++) assertEquals(Float.floatToRawIntBits(expected[i]),Float.floatToRawIntBits(out[i]),"distinct poisoned wrapper B="+batch);
        }
    }

    static List<float[]> execute(CpuGeneralScanLowering.Facts facts,CpuGeneratedKernel generated,int batch,List<float[]> input) throws Throwable {
        try (var arena = Arena.ofConfined()) {
            var args = new ArrayList<MemorySegment>(); var out = new ArrayList<float[]>();
            for (int p = 0; p < facts.batchBindings().size(); p++) {
                long span = facts.batchBindings().get(p).bind(batch).referencedElementSpan();
                float[] data = p < input.size() ? input.get(p) : new float[Math.toIntExact(span)];
                if (p>=input.size()) Arrays.fill(data,Float.intBitsToFloat(0x7fa77777));
                assertEquals(span,data.length); args.add(MemorySegment.ofArray(data)); if (p >= input.size()) out.add(data);
            }
            args.add(arena.allocate(facts.geometry().workspaceSize().byteSize(batch),4));
            MethodHandle entry = generated.entryPoint().asSpreader(0,MemorySegment[].class,args.size());
            long[] packed = facts.pack(batch); entry.invokeExact(args.toArray(MemorySegment[]::new),packed,0L,packed[1]);
            return out;
        }
    }
    static Tensor symbolic(boolean grad,Dimension... dims) {
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,Shape.ofDimensions(dims),Optional.empty(),grad));
    }
    static Tensor fixed(boolean grad,long... dims) {
        var shape = Shape.of(dims); return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,shape,Optional.of(LayoutDescriptor.contiguous(shape)),grad));
    }
    static float[] data(int n,float scale) { float[] result = new float[n]; for (int i = 0; i < n; i++) result[i] = (i%11-5)*scale; return result; }
    static String sha(byte[] bytes) throws Exception { return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)); }
    static void inspect(byte[] bytes) {
        var file = ClassFile.of().parse(bytes); assertTrue(ClassFile.of().verify(bytes).isEmpty());
        var instructions = file.methods().getFirst().code().orElseThrow().elementStream().filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        int lastPacking = -1, firstRead = Integer.MAX_VALUE;
        for (int i = 0; i < instructions.size(); i++) {
            var instruction = instructions.get(i);
            assertFalse(instruction instanceof NewObjectInstruction || instruction instanceof NewPrimitiveArrayInstruction || instruction instanceof NewReferenceArrayInstruction);
            if (instruction.opcode() == Opcode.LALOAD) lastPacking = i;
            if (instruction instanceof InvokeInstruction call) {
                String owner = call.owner().asInternalName(), name = call.name().stringValue();
                assertFalse(owner.startsWith("java/util/") || owner.startsWith("java/lang/reflect/") || name.equals("valueOf") || owner.contains("synaptik"),owner+"."+name);
                if (owner.equals("java/lang/foreign/MemorySegment") && name.equals("get")) firstRead = Math.min(firstRead,i);
            }
        }
        assertTrue(lastPacking >= 0 && lastPacking < firstRead,"packing stays outside generated arithmetic loops");
    }
}
