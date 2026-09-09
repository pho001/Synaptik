package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaskedReductionLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Direct generated-entry, clean-Java semantic closure for the masked and advanced reductions. */
class CpuMaskedAdvancedReductionSemanticClosureTest {
    private static final Shape INPUT = Shape.of(2, 3);
    private static final List<DataType> TYPES = List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16);
    private static final CpuPartitionAnalysisInputs.MaterializationPolicy MATERIALIZATION =
            new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 1, 20, 1, 3, 1_000_000, 1, 1);
    private static final String BASE = "/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/";

    @Test void everyMaskedAndAdvancedInventoryOwnerDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle()
            throws Throwable {
        Set<String> owners = new TreeSet<>();
        Map<String, Integer> counts = new TreeMap<>();
        for (AggregateReductionKind kind : List.of(AggregateReductionKind.SUM, AggregateReductionKind.MEAN)) {
            for (DataType type : TYPES) for (Shape mask : List.of(Shape.scalar(), Shape.of(3), Shape.of(1, 3))) {
                for (int axis : List.of(0, 1)) for (Request request : Request.values()) {
                    String owner = "specialized:masked/" + kind + '/' + type + '/' + mask + '/' + axis + '/' + request;
                    assertTrue(owners.add(owner), owner);
                    invokeMasked(kind, type, mask, axis, request, owner);
                    counts.merge(kind == AggregateReductionKind.SUM ? "MASKED_SUM" : "MASKED_MEAN", 1, Integer::sum);
                }
            }
        }
        for (AggregateReductionKind kind : List.of(AggregateReductionKind.LOG_SUM_EXP, AggregateReductionKind.L1_NORM,
                AggregateReductionKind.L2_NORM, AggregateReductionKind.VARIANCE, AggregateReductionKind.STANDARD_DEVIATION)) {
            for (DataType type : TYPES) for (List<Integer> axes : List.of(List.of(0), List.of(1), List.of(0, 1))) {
                for (boolean keep : List.of(false, true)) for (Request request : Request.values()) {
                    String owner = "specialized:advanced/" + kind + '/' + type + '/' + axes + '/' + keep + '/' + request;
                    assertTrue(owners.add(owner), owner);
                    invokeAdvanced(kind, type, axes, keep, request, owner);
                    counts.merge(kind.name(), 1, Integer::sum);
                }
            }
        }
        assertEquals(630, owners.size());
        for (String form : List.of("MASKED_SUM", "MASKED_MEAN", "LOG_SUM_EXP", "L1_NORM", "L2_NORM", "VARIANCE", "STANDARD_DEVIATION"))
            assertEquals(90, counts.get(form), form);
        assertEquals(inventoryOwners(), owners, "missing, duplicate, orphaned, or stale inventory owner");
    }

    private static void invokeMasked(AggregateReductionKind kind, DataType type, Shape maskShape, int axis,
            Request request, String owner) throws Throwable {
        PrepareContext<CpuPartitionAnalysisInputs> context = configure(CpuMaskedReductionLoweringTest.context(kind, type, INPUT, maskShape, axis), request);
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        byte[] bytes = bytes(route.specialization(), route.kernelIr(), owner);
        Object data = array(type, capacity(context.values().get(0).descriptor()));
        Object mask = new byte[capacity(context.values().get(1).descriptor())];
        Object output = array(type, capacity(context.values().get(2).descriptor()));
        Object expected = array(type, capacity(context.values().get(2).descriptor()));
        fillData(data, type, context.values().get(0).descriptor());
        fillMask((byte[]) mask, context.values().get(1).descriptor());
        Object inputBefore = copy(data);
        Arrays.fill((byte[]) mask, (byte) 1);
        fillMask((byte[]) mask, context.values().get(1).descriptor());
        fill(output, type, -91);
        fill(expected, type, -91);
        long start = 1;
        long end = plan.elementCount();
        maskedOracle(kind, type, context, data, (byte[]) mask, expected, axis, start, end);
        invoke(bytes, route.specialization(), List.of(data, mask, output), plan.maskedReductionGeometry().orElseThrow().pack(new long[3], 0),
                plan.maskedReductionGeometry().orElseThrow().scratchSliceBytes(), start, end);
        assertRaw(expected, output, owner);
        assertRaw(inputBefore, data, owner + " immutable input");
    }

    private static void invokeAdvanced(AggregateReductionKind kind, DataType type, List<Integer> axes, boolean keep,
            Request request, String owner) throws Throwable {
        var attrs = kind == AggregateReductionKind.VARIANCE || kind == AggregateReductionKind.STANDARD_DEVIATION
                ? new StatisticalReductionAttrs(axes, keep, axes.size() == 2 ? 1 : 0) : new MultiAxisReductionAttrs(axes, keep);
        Shape outputShape = outputShape(axes, keep);
        PrepareContext<CpuPartitionAnalysisInputs> context = configure(CpuAggregateLoweringTest.context(kind, type, INPUT, attrs, outputShape), request);
        var plan = new CpuPartitionPreparer().analyze(context).plan(); var route = plan.units().getFirst().portablePlan();
        byte[] bytes = bytes(route.specialization(), route.kernelIr(), owner);
        Object input = array(type, capacity(context.values().getFirst().descriptor())), output = array(type, capacity(context.values().getLast().descriptor())), expected = array(type, capacity(context.values().getLast().descriptor()));
        fillData(input, type, context.values().getFirst().descriptor()); Object before = copy(input); fill(output, type, -91); fill(expected, type, -91);
        long start = 1, end = plan.elementCount(); advancedOracle(kind, type, context, input, expected, axes, keep, axes.size() == 2 ? 1 : 0, start, end);
        invoke(bytes, route.specialization(), List.of(input, output), plan.advancedReductionGeometry().orElseThrow().pack(new long[2]),
                plan.advancedReductionGeometry().orElseThrow().scratchSliceBytes(), start, end);
        assertRaw(expected, output, owner); assertRaw(before, input, owner + " immutable input");
    }

    private static byte[] bytes(io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization spec,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr ir, String owner) {
        var generator = new CpuClassFileKernelGenerator(); byte[] first = generator.generateClassBytes(spec, ir);
        assertArrayEquals(first, generator.generateClassBytes(spec, ir), owner + " deterministic bytes"); return first;
    }
    private static void invoke(byte[] bytes, io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization spec,
            List<Object> stores, long[] geometry, long scratchBytes, long start, long end) throws Throwable {
        var entry = new CpuClassFileKernelGenerator().defineClassBytes(spec, bytes).entryPoint(); var args = new ArrayList<Object>();
        for (int i = 0; i < stores.size(); i++) args.add(spec.carrierPattern().get(i) == CarrierAccess.MEMORY_SEGMENT ? segment(stores.get(i)) : stores.get(i));
        try (Arena arena = Arena.ofConfined()) { if (scratchBytes != 0) args.add(arena.allocate(scratchBytes, Long.BYTES)); args.add(geometry); args.add(start); args.add(end); entry.invokeWithArguments(args); }
    }

    /* Independent scalar oracle: logical coordinates and arithmetic only; no lowering or reference kernel. */
    private static void maskedOracle(AggregateReductionKind kind, DataType type, PrepareContext<?> c, Object input, byte[] mask, Object out, int axis, long start, long end) {
        TensorDescriptor in = c.values().get(0).descriptor(), md = c.values().get(1).descriptor(), od = c.values().get(2).descriptor();
        long[] os = od.shape().toLongArray();
        for (long cell = start; cell < end; cell++) { long[] oc = coord(cell, os), ic = new long[2]; int p = 0; for (int a=0;a<2;a++) if(a!=axis) ic[a]=oc[p++];
            double sum=0; int n=0; for(int r=0;r<INPUT.toLongArray()[axis];r++){ic[axis]=r; long[] mc=maskCoord(ic, md.shape().toLongArray()); if(mask[(int)address(md.layout().orElseThrow(),mc)]==0)continue; sum+=get(input,type,address(in.layout().orElseThrow(),ic));n++;}
            set(out,type,address(od.layout().orElseThrow(),oc), kind==AggregateReductionKind.SUM ? sum : n==0 ? Double.NaN : sum/n); }
    }
    private static void advancedOracle(AggregateReductionKind kind, DataType type, PrepareContext<?> c, Object input, Object out, List<Integer> axes, boolean keep, long correction, long start, long end) {
        TensorDescriptor in=c.values().getFirst().descriptor(), od=c.values().getLast().descriptor(); long[] os=od.shape().toLongArray();
        for(long cell=start;cell<end;cell++){long[] oc=coord(cell,os); List<Double> values=new ArrayList<>(); for(int i=0;i<6;i++){long[] ic=coord(i,INPUT.toLongArray()); if(matches(ic,oc,axes,keep))values.add(get(input,type,address(in.layout().orElseThrow(),ic)));} double v;
            if(kind==AggregateReductionKind.LOG_SUM_EXP){double max=values.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NEGATIVE_INFINITY); double s=0;for(double x:values)s+=Math.exp(x-max);v=max+Math.log(s);}
            else if(kind==AggregateReductionKind.L1_NORM){v=0;for(double x:values)v+=Math.abs(x);}
            else if(kind==AggregateReductionKind.L2_NORM){v=0;for(double x:values)v+=x*x;v=Math.sqrt(v);}
            else {double mean=values.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN), s=0;for(double x:values){double d=x-mean;s+=d*d;}v=s/(values.size()-correction);if(kind==AggregateReductionKind.STANDARD_DEVIATION)v=Math.sqrt(v);}
            set(out,type,address(od.layout().orElseThrow(),oc),v); }
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> configure(PrepareContext<CpuPartitionAnalysisInputs> base, Request r) {
        var values=new ArrayList<GraphValue>();var memory=new ArrayList<LogicalMemoryRequirement>();
        for(int i=0;i<base.values().size();i++){var old=base.values().get(i);var d=old.descriptor();if(r.general&&d.shape().rank()!=0){long[] s=d.layout().orElseThrow().strides();for(int a=0;a<s.length;a++)s[a]*=2;d=new TensorDescriptor(d.dataType(),d.shape(),Optional.of(LayoutDescriptor.of(d.shape(),s,1,true)),false);}values.add(new GraphValue(old.id(),d));var m=base.memoryRequirements().get(i);memory.add(new LogicalMemoryRequirement(m.valueId(),d,m.producerPartition(),m.consumerPartitions(),m.graphOutput()));}
        var carriers=new ArrayList<CarrierAccess>();for(int i=0;i<values.size();i++)carriers.add(r.segment||r.mixed&&i%2==1?CarrierAccess.MEMORY_SEGMENT:heap(values.get(i).descriptor().dataType()));
        return new PrepareContext<>(base.partition(),base.nodes(),values,memory,base.constants(),new CpuPartitionAnalysisInputs(false,carriers,r.execution,r.materialization?MATERIALIZATION:CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
    }
    private static Set<String> inventoryOwners() throws Exception {String text;try(InputStream in=CpuMaskedAdvancedReductionSemanticClosureTest.class.getResourceAsStream(BASE+"generated-coverage-inventory.tsv")){assertNotNull(in);text=new String(in.readAllBytes(),StandardCharsets.UTF_8);}assertEquals("527f36de41b64c228dd215c6f3182138b4c70db24c915117c4d7f82743ac41cc",java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(text.getBytes(StandardCharsets.UTF_8))));Set<String>s=new TreeSet<>();for(String line:text.split("\\n")){String[]r=line.split("\\t",-1);if(r.length==27&&r[0].startsWith("specialized:")&&Set.of("MASKED_SUM","MASKED_MEAN","LOG_SUM_EXP","L1_NORM","L2_NORM","VARIANCE","STANDARD_DEVIATION").contains(r[2]))assertTrue(s.add(r[0]),"duplicate "+r[0]);}return s;}
    private static Shape outputShape(List<Integer> axes,boolean keep){long[]in=INPUT.toLongArray();var d=new ArrayList<Long>();for(int a=0;a<2;a++)if(axes.contains(a)){if(keep)d.add(1L);}else d.add(in[a]);return Shape.of(d.stream().mapToLong(Long::longValue).toArray());}
    private static boolean matches(long[]ic,long[]oc,List<Integer>axes,boolean keep){int p=0;for(int a=0;a<2;a++){if(axes.contains(a)){if(keep)p++;}else if(ic[a]!=oc[p++])return false;}return true;}
    private static long[] maskCoord(long[]ic,long[]ms){long[]r=new long[ms.length];for(int a=0;a<ms.length;a++)r[a]=ms[a]==1?0:ic[ic.length-ms.length+a];return r;}
    private static long address(LayoutDescriptor l,long[]c){long v=l.storageOffset();long[]s=l.strides();for(int a=0;a<c.length;a++)v+=c[a]*s[a];return v;}
    private static long[] coord(long ordinal,long[]shape){long[]r=new long[shape.length];for(int a=shape.length-1;a>=0;a--){r[a]=ordinal%shape[a];ordinal/=shape[a];}return r;}
    private static int capacity(TensorDescriptor d){long max=d.layout().orElseThrow().storageOffset();long[]s=d.layout().orElseThrow().strides(),sh=d.shape().toLongArray();for(int a=0;a<s.length;a++)max+=(sh[a]-1)*s[a];return (int)max+1;}
    private static CarrierAccess heap(DataType t){return switch(t){case FLOAT64->CarrierAccess.DOUBLE_ARRAY;case FLOAT32->CarrierAccess.FLOAT_ARRAY;case BFLOAT16->CarrierAccess.SHORT_ARRAY;case BOOL->CarrierAccess.BYTE_ARRAY;default->throw new AssertionError(t);};}
    private static Object array(DataType t,int n){return t==DataType.FLOAT64?new double[n]:t==DataType.FLOAT32?new float[n]:new short[n];}
    private static MemorySegment segment(Object a){if(a instanceof double[]x)return MemorySegment.ofArray(x);if(a instanceof float[]x)return MemorySegment.ofArray(x);if(a instanceof short[]x)return MemorySegment.ofArray(x);return MemorySegment.ofArray((byte[])a);}
    private static void fillData(Object a,DataType t,TensorDescriptor d){double[]v={1,-2,3,4,-5,6};for(int i=0;i<6;i++)set(a,t,address(d.layout().orElseThrow(),coord(i,INPUT.toLongArray())),v[i]);}
    private static void fillMask(byte[]a,TensorDescriptor d){long n=1;for(long x:d.shape().toLongArray())n*=x;for(long i=0;i<n;i++)a[(int)address(d.layout().orElseThrow(),coord(i,d.shape().toLongArray()))]=(byte)((i&1)==0?1:0);}
    private static double get(Object a,DataType t,long i){return t==DataType.FLOAT64?((double[])a)[(int)i]:t==DataType.FLOAT32?((float[])a)[(int)i]:Float.intBitsToFloat(Short.toUnsignedInt(((short[])a)[(int)i])<<16);}
    private static void set(Object a,DataType t,long i,double v){if(t==DataType.FLOAT64)((double[])a)[(int)i]=v;else if(t==DataType.FLOAT32)((float[])a)[(int)i]=(float)v;else ((short[])a)[(int)i]=(short)(Float.floatToRawIntBits((float)v)>>>16);}
    private static void fill(Object a,DataType t,double v){for(int i=0;i<java.lang.reflect.Array.getLength(a);i++)set(a,t,i,v);}
    private static Object copy(Object a){if(a instanceof double[]x)return x.clone();if(a instanceof float[]x)return x.clone();return ((short[])a).clone();}
    private static void assertRaw(Object e,Object a,String m){if(e instanceof double[]x){double[]y=(double[])a;for(int i=0;i<x.length;i++)assertTrue(Double.isNaN(x[i])?Double.isNaN(y[i]):Math.abs(x[i]-y[i])<=Math.ulp(x[i])*2,m+" at "+i);}else if(e instanceof float[]x){float[]y=(float[])a;for(int i=0;i<x.length;i++)assertTrue(Float.isNaN(x[i])?Float.isNaN(y[i]):Math.abs(x[i]-y[i])<=Math.ulp(x[i])*2,m+" at "+i);}else {short[]x=(short[])e,y=(short[])a;assertEquals(x.length,y.length,m);for(int i=0;i<x.length;i++)assertTrue(Math.abs(Short.toUnsignedInt(x[i])-Short.toUnsignedInt(y[i]))<=1,m+" BFLOAT16 represented rounding at "+i);}}
    private enum Request {HEAP_CONTIGUOUS_SCALAR(false,false,false,false,CpuGeneratedDirectEvidenceClosureTest.scalar(1)),SEGMENT_CONTIGUOUS_VECTOR(true,false,false,false,CpuGeneratedDirectEvidenceClosureTest.vector(1)),MIXED_GENERAL_PARALLEL_VECTOR(false,true,true,false,CpuGeneratedDirectEvidenceClosureTest.vector(4)),HEAP_GENERAL_PARALLEL_SCALAR(false,false,true,false,CpuGeneratedDirectEvidenceClosureTest.scalar(4)),HEAP_GENERAL_MATERIALIZATION(false,false,true,true,CpuGeneratedDirectEvidenceClosureTest.scalar(1));final boolean segment,mixed,general,materialization;final CpuPartitionAnalysisInputs.PortableExecutionConfig execution;Request(boolean s,boolean x,boolean g,boolean m,CpuPartitionAnalysisInputs.PortableExecutionConfig e){segment=s;mixed=x;general=g;materialization=m;execution=e;}}
}
