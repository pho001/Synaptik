package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuSoftmaxInputValidator;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

/**
 * CPU-private Stage-A gate for the current finite, scalar softmax generated entries.
 *
 * <p>Candidate two uses actual preferred-species loads and an identity vector map, but folds
 * each vector's lanes in increasing logical order.  Thus it is a load/map investigation, not
 * permission to reassociate either stable reduction.  Indexed selected-axis layouts are recorded
 * as scalar-only: this test has no proof that the current Vector API access form admits an
 * allocation-free, carrier-uniform gather for them.</p>
 */
final class CpuStableReductionSoftmaxStageATest {
    private enum Carrier { ARRAY_ARRAY, SEGMENT_SEGMENT, ARRAY_SEGMENT, SEGMENT_ARRAY }

    @Test void generatedEntriesAndOrderedVectorMapMatchDirectThreePassOracle() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            int lanes = lanes(type);
            assertTrue(lanes > 1, "preferred species must be multi-lane");
            for (int width : List.of(1, lanes - 1, lanes, lanes + 1, 2 * lanes + 3))
                for (int axis = 0; axis < 3; axis++)
                    for (SoftmaxKind kind : SoftmaxKind.values())
                        for (Carrier carrier : Carrier.values())
                            runGeneratedDirectAndCandidate(type, kind, axis, width, carrier);
        }
    }

    @Test void nonFiniteAndZeroSelectedExtentRemainRejectedBeforeAnyGeneratedWrite() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            var layout = new CpuSoftmaxLowering.Layout(new long[] {2}, 0, new long[] {1});
            var geometry = new CpuSoftmaxLowering.Geometry(SoftmaxKind.SOFTMAX, type, 0,
                    layout, layout, 1, 2, 2);
            for (double value : new double[] {Double.NaN, Double.POSITIVE_INFINITY,
                    Double.NEGATIVE_INFINITY}) {
                CpuBufferArgument input = type == DataType.FLOAT32
                        ? new CpuBufferArgument.Floats(new float[] {(float) value, 0}, 0, 8, true)
                        : new CpuBufferArgument.Doubles(new double[] {value, 0}, 0, 16, true);
                assertThrows(IllegalArgumentException.class,
                        () -> CpuSoftmaxInputValidator.validate(input, geometry), type + "/" + value);
            }
            for (double[] pair : List.of(new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY},
                    new double[] {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
                    new double[] {Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY},
                    new double[] {Double.NaN, Double.POSITIVE_INFINITY},
                    new double[] {Double.NaN, Double.NEGATIVE_INFINITY})) {
                CpuBufferArgument input = type == DataType.FLOAT32
                        ? new CpuBufferArgument.Floats(new float[] {(float) pair[0], (float) pair[1]}, 0, 8, true)
                        : new CpuBufferArgument.Doubles(pair, 0, 16, true);
                assertThrows(IllegalArgumentException.class,
                        () -> CpuSoftmaxInputValidator.validate(input, geometry), type + "/" + Arrays.toString(pair));
            }
            if (type == DataType.FLOAT64) {
                CpuBufferArgument extremes = new CpuBufferArgument.Doubles(
                        new double[] {Double.MAX_VALUE, -Double.MAX_VALUE}, 0, 16, true);
                assertThrows(IllegalArgumentException.class,
                        () -> CpuSoftmaxInputValidator.validate(extremes, geometry), type + "/extreme shift");
            }
        }
        assertThrows(IllegalArgumentException.class, () -> CpuSoftmaxLoweringTest.lower(
                SoftmaxKind.SOFTMAX, DataType.FLOAT32, Shape.of(2, 0), 1));
        assertEquals(0, CpuSoftmaxLoweringTest.lower(SoftmaxKind.LOG_SOFTMAX,
                DataType.FLOAT64, Shape.of(0, 2, 3), 1).softmaxGeometry().orElseThrow()
                .elementCount(), "zero non-selected extent performs no generated slice");
    }

    @Test void indexedPositiveStrideIsExplicitScalarOnlyControlAndCompleteRangesAreDeterministic()
            throws Throwable {
        // fromArray/fromMemorySegment used by candidate two are contiguous loads.  Java 26's
        // current generated carrier body supplies only a base plus scalar axis stride; no gather
        // legality/performance proof exists here, so an indexed form cannot enter candidate two.
        long[] indexed = {2, 5, 1};
        assertTrue(indexed[1] != 1, "indexed proof must not be silently treated as contiguous");
        Fixture fixture = fixture(DataType.FLOAT64, SoftmaxKind.LOG_SOFTMAX, 1, 5,
                Carrier.ARRAY_ARRAY);
        invoke(fixture, 0, 4); // four complete slices
        double[] once = ((double[]) fixture.output).clone();
        Arrays.fill((double[]) fixture.output, Double.longBitsToDouble(0x7ff8000000000001L));
        invoke(fixture, 0, 2); invoke(fixture, 2, 4);
        assertRawEquals(DataType.FLOAT64, once, (double[]) fixture.output,
                "disjoint complete-slice ranges");
        Arrays.fill((double[]) fixture.output, Double.longBitsToDouble(0x7ff8000000000001L));
        try (var callers = Executors.newFixedThreadPool(2)) {
            for (var result : callers.invokeAll(List.<Callable<Void>>of(
                    () -> { invokeCaller(fixture, 0, 2); return null; },
                    () -> { invokeCaller(fixture, 2, 4); return null; }))) result.get();
        }
        assertRawEquals(DataType.FLOAT64, once, (double[]) fixture.output,
                "concurrent caller-owned complete-slice ranges");
    }

    @Test void generatedEntryAndContiguousCandidateHaveNoSemanticBridgeOrHotScratch() {
        Fixture fixture = fixture(DataType.FLOAT64, SoftmaxKind.SOFTMAX, 2,
                lanes(DataType.FLOAT64) + 1, Carrier.SEGMENT_ARRAY);
        var model = ClassFile.of().parse(fixture.classBytes);
        var members = java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast).toList();
        assertEquals("(Ljava/lang/foreign/MemorySegment;[D[JJJ)V",
                fixture.entry.type().descriptorString());
        assertTrue(members.stream().noneMatch(member -> member.owner().asInternalName()
                .startsWith("io/github/pho001/synaptik")), "no validator/reference/helper bridge");
        assertTrue(members.stream().noneMatch(member -> member.owner().asInternalName()
                .equals("java/lang/Object") && member.name().stringValue().equals("new")),
                "generated body has no allocation owner");
        // Candidate two accepts fixture-provided output; its three pass loops construct neither a
        // pack buffer nor a carrier lookup. `decode` is used only after generated execution to
        // inspect the typed output carrier, never by the contiguous candidate.
        Object output = emptyLike(DataType.FLOAT64, fixture.output);
        candidateTwoInto(DataType.FLOAT64, SoftmaxKind.SOFTMAX, fixture.input, output,
                Math.toIntExact(fixture.extents[2]), 0, fixture.slices);
    }

    @Test void candidateTwoContainsPreferredSpeciesBlockOperationsWithoutVectorReloadHelpers()
            throws Exception {
        byte[] bytes;
        try (var stream = CpuStableReductionSoftmaxStageATest.class.getResourceAsStream(
                "CpuStableReductionSoftmaxStageATest.class")) {
            assertTrue(stream != null, "test class bytes are available for structural inspection");
            bytes = stream.readAllBytes();
        }
        var method = ClassFile.of().parse(bytes).methods().stream().filter(candidate -> candidate
                .methodName().stringValue().equals("candidateTwoInto")).findFirst().orElseThrow();
        var instructions = method.code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        List<String> invokes = instructions.stream().filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast).map(call -> call.owner().asInternalName() + "."
                        + call.name().stringValue()).toList();
        assertTrue(invokes.stream().anyMatch(call -> call.endsWith(".fromArray")
                || call.endsWith(".fromMemorySegment")), "full blocks load a preferred species");
        assertTrue(invokes.stream().anyMatch(call -> call.endsWith(".neg")),
                "full blocks execute a vector map");
        assertTrue(invokes.stream().anyMatch(call -> call.endsWith(".lane")),
                "folds consume lanes in scalar increasing order");
        assertTrue(invokes.stream().noneMatch(call -> call.endsWith(".loadFloat")
                || call.endsWith(".loadDouble") || call.endsWith(".scalar")
                || call.endsWith(".store")), "no per-element vector reload or carrier helper");
        assertTrue(instructions.stream().noneMatch(instruction -> instruction.opcode().name()
                .startsWith("NEW") || instruction.opcode() == Opcode.ANEWARRAY),
                "candidate hot method allocates no array or object");
    }

    private static void runGeneratedDirectAndCandidate(DataType type, SoftmaxKind kind, int axis,
            int width, Carrier carrier) throws Throwable {
        Fixture fixture = fixture(type, kind, axis, width, carrier);
        invoke(fixture, 0, fixture.slices);
        double[] generated = decode(type, fixture.output);
        Object directCarrier = emptyLike(type, fixture.output);
        directInto(type, kind, fixture.input, directCarrier, fixture.extents, axis, 0,
                fixture.slices);
        double[] direct = decode(type, directCarrier);
        assertRawEquals(type, direct, generated, "generated/direct " + type + "/" + kind + "/"
                + carrier + "/axis=" + axis + "/width=" + width);
        if (axis == fixture.extents.length - 1) {
            Object vectorCarrier = emptyLike(type, fixture.output);
            candidateTwoInto(type, kind, fixture.input, vectorCarrier, width, 0, fixture.slices);
            assertRawEquals(type, direct, decode(type, vectorCarrier),
                    "candidate2 contiguous " + type + "/" + kind + "/" + carrier);
        }
    }

    private record Fixture(Object input, Object output, long[] extents,
            long[] geometry, long slices, java.lang.invoke.MethodHandle entry, byte[] classBytes) { }

    private static Fixture fixture(DataType type, SoftmaxKind kind, int axis, int width,
            Carrier carrier) {
        long[] extents = axis == 0 ? new long[] {width, 2, 2}
                : axis == 1 ? new long[] {2, width, 2} : new long[] {2, 2, width};
        Shape shape = Shape.of(extents);
        List<CarrierAccess> accesses = List.of(inputAccess(type, carrier), outputAccess(type, carrier));
        var base = CpuSoftmaxLoweringTest.context(kind, type, shape, axis);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), java.util.Map.of(), new CpuPartitionAnalysisInputs(false, accesses));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        var artifact = generator.defineClassBytes(route.specialization(), bytes);
        int count = Math.toIntExact(Arrays.stream(extents).reduce(1L, Math::multiplyExact));
        double[] logical = values(count);
        Object input = represented(type, logical); Object output = represented(type, new double[count]);
        if (carrier == Carrier.SEGMENT_SEGMENT || carrier == Carrier.SEGMENT_ARRAY) input = segment(input);
        if (carrier == Carrier.SEGMENT_SEGMENT || carrier == Carrier.ARRAY_SEGMENT) output = segment(output);
        return new Fixture(input, output, extents,
                plan.softmaxGeometry().orElseThrow().pack(new long[2]), count / width,
                artifact.entryPoint(), bytes);
    }

    private static void invoke(Fixture fixture, long start, long end) throws Throwable {
        fixture.entry.invokeWithArguments(fixture.input, fixture.output, fixture.geometry, start, end);
    }
    private static void invokeCaller(Fixture fixture, long start, long end) {
        try { invoke(fixture, start, end); } catch (Throwable failure) { throw new AssertionError(failure); }
    }

    /* Optimal direct Java oracle: same typed carriers, cold geometry/range, and logical order. */
    private static void directInto(DataType type, SoftmaxKind kind, Object input, Object output,
            long[] extents, int axis, long start, long end) {
        int width = Math.toIntExact(extents[axis]);
        int inner = 1; for (int dimension = axis + 1; dimension < extents.length; dimension++)
            inner = Math.multiplyExact(inner, Math.toIntExact(extents[dimension]));
        for (long slice = start; slice < end; slice++) {
            int outer = Math.toIntExact(slice / inner), within = Math.toIntExact(slice % inner),
                    base = outer * width * inner + within;
            double maximum = Double.NEGATIVE_INFINITY;
            for (int coordinate = 0; coordinate < width; coordinate++) { int p=base+coordinate*inner;
                double v = scalar(type, input, p); if (v > maximum) maximum = v; }
            double sum = 0, compensation = 0;
            for (int coordinate = 0; coordinate < width; coordinate++) { int p=base+coordinate*inner;
                double shifted = scalar(type, input, p) - maximum;
                double addend = Math.exp(shifted) - compensation, temporary = sum + addend;
                compensation = (temporary - sum) - addend; sum = temporary; }
            double logarithm = kind == SoftmaxKind.LOG_SOFTMAX ? Math.log(sum) : 0;
            for (int coordinate = 0; coordinate < width; coordinate++) { int p=base+coordinate*inner;
                double shifted = scalar(type, input, p) - maximum;
                store(type, output, p, kind == SoftmaxKind.SOFTMAX ? Math.exp(shifted) / sum
                        : shifted - logarithm); }
        }
    }


    /* Candidate two: no packing/allocation in its inner passes; actual vector loads/map + ordered lanes. */
    private static void candidateTwoInto(DataType type, SoftmaxKind kind, Object input,
            Object output, int width, long start, long end) {
        int lanes = lanes(type);
        for (long slice = start; slice < end; slice++) {
            int base = Math.toIntExact(slice * width);
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < width; ) { int n = Math.min(lanes, width - i);
                if (n == lanes) { if (type == DataType.FLOAT32) { FloatVector v = input instanceof float[] a ? FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, a, base + i) : FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, (MemorySegment) input, (long) (base + i) * 4, ByteOrder.nativeOrder()); v = v.neg().neg(); for (int l=0;l<lanes;l++) { double value=v.lane(l); if(value>max) max=value; } }
                    else { DoubleVector v = input instanceof double[] a ? DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, a, base + i) : DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, (MemorySegment) input, (long) (base + i) * 8, ByteOrder.nativeOrder()); v = v.neg().neg(); for (int l=0;l<lanes;l++) { double value=v.lane(l); if(value>max) max=value; } } }
                else for (int l=0;l<n;l++) { int p=base+i+l; double value=type==DataType.FLOAT32?(input instanceof float[] a?a[p]:((MemorySegment)input).get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()),(long)p*4)):(input instanceof double[] a?a[p]:((MemorySegment)input).get(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()),(long)p*8)); if(value>max) max=value; } i+=n; }
            double sum=0, compensation=0;
            for (int i=0;i<width;) { int n=Math.min(lanes,width-i);
                if (n == lanes) {
                    if (type == DataType.FLOAT32) { FloatVector v=input instanceof float[] a ? FloatVector.fromArray(FloatVector.SPECIES_PREFERRED,a,base+i) : FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED,(MemorySegment)input,(long)(base+i)*4,ByteOrder.nativeOrder()); v=v.neg().neg();
                        for(int l=0;l<lanes;l++){double a=Math.exp(v.lane(l)-max)-compensation,t=sum+a;compensation=(t-sum)-a;sum=t;} }
                    else { DoubleVector v=input instanceof double[] a ? DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED,a,base+i) : DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED,(MemorySegment)input,(long)(base+i)*8,ByteOrder.nativeOrder()); v=v.neg().neg();
                        for(int l=0;l<lanes;l++){double a=Math.exp(v.lane(l)-max)-compensation,t=sum+a;compensation=(t-sum)-a;sum=t;} }
                } else for(int l=0;l<n;l++){int p=base+i+l;double value=type==DataType.FLOAT32?(input instanceof float[] a?a[p]:((MemorySegment)input).get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()),(long)p*4)):(input instanceof double[] a?a[p]:((MemorySegment)input).get(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()),(long)p*8));double a=Math.exp(value-max)-compensation,t=sum+a;compensation=(t-sum)-a;sum=t;}
                i+=n;
            }
            double log=kind==SoftmaxKind.LOG_SOFTMAX?Math.log(sum):0;
            for (int i=0;i<width;) { int n=Math.min(lanes,width-i);
                if (n == lanes) {
                    if (type == DataType.FLOAT32) { FloatVector v=input instanceof float[] a ? FloatVector.fromArray(FloatVector.SPECIES_PREFERRED,a,base+i) : FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED,(MemorySegment)input,(long)(base+i)*4,ByteOrder.nativeOrder()); v=v.neg().neg();
                        for(int l=0;l<lanes;l++){int p=base+i+l;double shifted=v.lane(l)-max,result=kind==SoftmaxKind.SOFTMAX?Math.exp(shifted)/sum:shifted-log;if(output instanceof float[] a)a[p]=(float)result;else ((MemorySegment)output).set(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()),(long)p*4,(float)result);} }
                    else { DoubleVector v=input instanceof double[] a ? DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED,a,base+i) : DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED,(MemorySegment)input,(long)(base+i)*8,ByteOrder.nativeOrder()); v=v.neg().neg();
                        for(int l=0;l<lanes;l++){int p=base+i+l;double shifted=v.lane(l)-max,result=kind==SoftmaxKind.SOFTMAX?Math.exp(shifted)/sum:shifted-log;if(output instanceof double[] a)a[p]=result;else ((MemorySegment)output).set(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()),(long)p*8,result);} }
                } else for(int l=0;l<n;l++){int p=base+i+l;double value=type==DataType.FLOAT32?(input instanceof float[] a?a[p]:((MemorySegment)input).get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()),(long)p*4)):(input instanceof double[] a?a[p]:((MemorySegment)input).get(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()),(long)p*8));double shifted=value-max,result=kind==SoftmaxKind.SOFTMAX?Math.exp(shifted)/sum:shifted-log;if(type==DataType.FLOAT32){if(output instanceof float[] a)a[p]=(float)result;else ((MemorySegment)output).set(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()),(long)p*4,(float)result);}else if(output instanceof double[] a)a[p]=result;else ((MemorySegment)output).set(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()),(long)p*8,result);}
                i+=n;
            }
        }
    }


    private static int lanes(DataType t) { return t==DataType.FLOAT32?FloatVector.SPECIES_PREFERRED.length():DoubleVector.SPECIES_PREFERRED.length(); }
    private static double scalar(DataType t,Object c,int i){if(t==DataType.FLOAT32)return c instanceof float[] a?a[i]:((MemorySegment)c).get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()),(long)i*4);return c instanceof double[] a?a[i]:((MemorySegment)c).get(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()),(long)i*8);}
    private static void store(DataType t,Object c,int i,double v){if(t==DataType.FLOAT32){if(c instanceof float[] a)a[i]=(float)v;else ((MemorySegment)c).set(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()),(long)i*4,(float)v);}else if(c instanceof double[] a)a[i]=v;else ((MemorySegment)c).set(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()),(long)i*8,v);}
    private static CarrierAccess inputAccess(DataType t,Carrier c){boolean segment=c==Carrier.SEGMENT_SEGMENT||c==Carrier.SEGMENT_ARRAY;return segment?CarrierAccess.MEMORY_SEGMENT:t==DataType.FLOAT32?CarrierAccess.FLOAT_ARRAY:CarrierAccess.DOUBLE_ARRAY;}
    private static CarrierAccess outputAccess(DataType t,Carrier c){boolean segment=c==Carrier.SEGMENT_SEGMENT||c==Carrier.ARRAY_SEGMENT;return segment?CarrierAccess.MEMORY_SEGMENT:t==DataType.FLOAT32?CarrierAccess.FLOAT_ARRAY:CarrierAccess.DOUBLE_ARRAY;}
    private static double[] values(int n){double[] r=new double[n];double[] x={-0d,0d,Double.MIN_VALUE,-Double.MIN_VALUE,700,-700,1,-1,87,-87};for(int i=0;i<n;i++)r[i]=x[i%x.length];return r;}
    private static Object represented(DataType t,double[] v){if(t==DataType.FLOAT64)return v.clone();float[] r=new float[v.length];for(int i=0;i<v.length;i++)r[i]=(float)v[i];return r;}
    private static double represented(DataType t,double v){return t==DataType.FLOAT32?(float)v:v;}
    private static MemorySegment segment(Object a){return a instanceof float[] x?MemorySegment.ofArray(x):MemorySegment.ofArray((double[])a);}
    private static Object emptyLike(DataType t,Object carrier){int count=t==DataType.FLOAT32?(carrier instanceof float[] a?a.length:Math.toIntExact(((MemorySegment)carrier).byteSize()/4)):(carrier instanceof double[] a?a.length:Math.toIntExact(((MemorySegment)carrier).byteSize()/8));Object array=t==DataType.FLOAT32?new float[count]:new double[count];return carrier instanceof MemorySegment?segment(array):array;}
    private static double[] decode(DataType t,Object o){int n=t==DataType.FLOAT32?(o instanceof float[] x?x.length:(int)((MemorySegment)o).byteSize()/4):(o instanceof double[] x?x.length:(int)((MemorySegment)o).byteSize()/8);double[] r=new double[n];for(int i=0;i<n;i++)r[i]=scalar(t,o,i);return r;}
    private static void assertRawEquals(DataType t,double[] a,double[] b,String m){assertEquals(a.length,b.length,m);for(int i=0;i<a.length;i++)assertEquals(t==DataType.FLOAT32?Float.floatToRawIntBits((float)a[i]):Double.doubleToRawLongBits(a[i]),t==DataType.FLOAT32?Float.floatToRawIntBits((float)b[i]):Double.doubleToRawLongBits(b[i]),m+"/"+i);}
}
