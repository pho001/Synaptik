package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Direct generated-entry semantic closure for every ordinary gather and ordering matrix row. */
class CpuIndexingOrderingSemanticClosureTest {
    @Test void everyOrdinaryGatherAndOrderingRowExecutesAgainstAnIndependentOracle() throws Throwable {
        var candidates = CpuOrdinaryNonPointwiseGeneratedMatrixTest.indexingOrOrderingCandidates();
        assertEquals(336, candidates.size());
        assertEquals(48, candidates.stream().filter(c -> c.ownerId().startsWith("gather/")).count());
        assertEquals(48, candidates.stream().filter(c -> c.ownerId().startsWith("gather-elements/")).count());
        assertEquals(48, candidates.stream().filter(c -> c.ownerId().startsWith("gather-nd/")).count());
        assertEquals(48, candidates.stream().filter(c -> c.ownerId().startsWith("sort/")).count());
        assertEquals(48, candidates.stream().filter(c -> c.ownerId().startsWith("argsort/")).count());
        assertEquals(96, candidates.stream().filter(c -> c.ownerId().startsWith("top-k/")).count());
        assertEquals(candidates.size(), candidates.stream().map(CpuOrdinaryNonPointwiseGeneratedMatrixTest.IndexingOrOrderingCandidate::ownerId).distinct().count());
        for (var candidate : candidates) execute(candidate);
    }

    private static void execute(CpuOrdinaryNonPointwiseGeneratedMatrixTest.IndexingOrOrderingCandidate candidate) throws Throwable {
        var context = candidate.context(); var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        byte[] first = generator.generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(first, generator.generateClassBytes(route.specialization(), route.kernelIr()), candidate.ownerId());
        var entry = generator.defineClassBytes(route.specialization(), first).entryPoint();
        var raw = new ArrayList<Object>(); var expected = new ArrayList<Object>();
        for (int i = 0; i < context.values().size(); i++) {
            var type = context.values().get(i).descriptor().dataType();
            Object actual = storage(type, capacity(context, i)); fill(actual, type, i == 0 ? 0 : -77);
            if (i == 0) fillInput(context, actual, type);
            if (isIndexInput(context, i)) fillIndices(context, actual, type);
            raw.add(actual); expected.add(copy(actual));
        }
        long start = 1, end = plan.indexingGeometry().isPresent()
                ? Math.max(start + 1, plan.elementCount() - 1) : 2;
        if (plan.indexingGeometry().isPresent()) indexingOracle(context, expected, start, end);
        else orderingOracle(context, expected, start, end);
        var carriers = new ArrayList<Object>();
        for (int i = 0; i < raw.size(); i++) carriers.add(route.specialization().carrierPattern().get(i) == CarrierAccess.MEMORY_SEGMENT ? segment(raw.get(i)) : raw.get(i));
        long[] geometry = plan.indexingGeometry().isPresent()
                ? plan.indexingGeometry().orElseThrow().pack(new long[raw.size()], start, end)
                : plan.orderingGeometry().orElseThrow().pack(new long[raw.size()], start, end, 0);
        try (Arena arena = Arena.ofConfined()) {
            var args = new ArrayList<>(carriers);
            if (plan.workspaceDeclaration().isPresent()) args.add(arena.allocate(plan.workspaceDeclaration().orElseThrow().byteSize(), 8));
            args.add(geometry); args.add(start); args.add(end); entry.invokeWithArguments(args);
        }
        for (int i = 0; i < raw.size(); i++) assertRaw(expected.get(i), raw.get(i), candidate.ownerId() + "/role=" + i);
    }

    private static boolean isIndexInput(io.github.pho001.synaptik.prepare.analysis.PrepareContext<?> c, int i) {
        return c.nodes().getFirst().operation().kind() instanceof AxisGatherKind || c.nodes().getFirst().operation().kind() instanceof GatherNdKind ? i == 1 : false;
    }
    private static void indexingOracle(io.github.pho001.synaptik.prepare.analysis.PrepareContext<?> c, List<Object> a, long start, long end) {
        var kind = c.nodes().getFirst().operation().kind(); var type = c.values().getFirst().descriptor().dataType();
        long[] outShape = shape(c.values().getLast().descriptor().shape());
        for (long ordinal = start; ordinal < end; ordinal++) {
            long[] out = coords(ordinal, outShape); long[] in = out.clone();
            if (kind == AxisGatherKind.GATHER) { int axis = ((io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs)c.nodes().getFirst().operation().attrs()).axis(); in[axis] = longAt(c, 1, a.get(1), new long[] {out[axis]}); }
            else if (kind == AxisGatherKind.GATHER_ELEMENTS) { int axis = ((io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs)c.nodes().getFirst().operation().attrs()).axis(); in[axis] = longAt(c, 1, a.get(1), out); }
            else { long index = longAt(c, 1, a.get(1), Arrays.copyOf(out, out.length - 1)); in[0] = index; }
            set(a.getLast(), type, address(c, c.values().size() - 1, out), get(a.getFirst(), type, address(c, 0, in)));
        }
    }
    private static void orderingOracle(io.github.pho001.synaptik.prepare.analysis.PrepareContext<?> c, List<Object> a, long start, long end) {
        var operation = c.nodes().getFirst().operation(); var type = c.values().getFirst().descriptor().dataType();
        long[] inputShape = shape(c.values().getFirst().descriptor().shape()); int axis = operation.kind() == TopKKind.TOP_K ? ((io.github.pho001.synaptik.model.operation.ordering.TopKAttrs) operation.attrs()).axis() : ((io.github.pho001.synaptik.model.operation.ordering.SortAttrs) operation.attrs()).axis();
        boolean descending = operation.kind() == TopKKind.TOP_K ? ((io.github.pho001.synaptik.model.operation.ordering.TopKAttrs) operation.attrs()).largest() : ((io.github.pho001.synaptik.model.operation.ordering.SortAttrs) operation.attrs()).descending();
        int k = operation.kind() == TopKKind.TOP_K ? Math.toIntExact(((io.github.pho001.synaptik.model.operation.ordering.TopKAttrs) operation.attrs()).k()) : (int) inputShape[axis];
        long slices = elems(inputShape) / inputShape[axis];
        for (long slice = start; slice < Math.min(end, slices); slice++) {
            long[] outer = outerCoords(slice, inputShape, axis); var order = new ArrayList<Integer>();
            for (int j = 0; j < inputShape[axis]; j++) order.add(j);
            order.sort((x, y) -> compare(type, get(a.getFirst(), type, address(c, 0, insert(outer, axis, x))), get(a.getFirst(), type, address(c, 0, insert(outer, axis, y))), descending));
            boolean sorted = operation.kind() != TopKKind.TOP_K || ((io.github.pho001.synaptik.model.operation.ordering.TopKAttrs) operation.attrs()).sorted();
            if (operation.kind() == TopKKind.TOP_K && !sorted) { order = new ArrayList<>(order.subList(0, k)); order.sort(Comparator.naturalOrder()); }
            for (int j = 0; j < k; j++) { int source = order.get(j); long[] out = insert(outer, axis, j);
                if (operation.kind() == OrderingKind.ARGSORT) set(a.getLast(), DataType.INT64, address(c, 1, out), (long) source);
                else { set(a.get(1), type, address(c, 1, out), get(a.getFirst(), type, address(c, 0, insert(outer, axis, source)))); if (operation.kind() == TopKKind.TOP_K) set(a.get(2), DataType.INT64, address(c, 2, out), (long) source); }
            }
        }
    }
    private static int compare(DataType t, Object x, Object y, boolean descending) { double a = number(t,x), b = number(t,y); int r = Double.isNaN(a) ? Double.isNaN(b) ? 0 : 1 : Double.isNaN(b) ? -1 : Double.compare(a,b); return descending && r != 0 ? -r : r; }
    private static long[] outerCoords(long n,long[] shape,int axis){long[] p=new long[shape.length-1];for(int i=shape.length-1,j=p.length-1;i>=0;i--){if(i==axis)continue;p[j--]=n%shape[i];n/=shape[i];}return p;}
    private static long[] insert(long[] outer,int axis,long x){long[] p=new long[outer.length+1];for(int i=0,j=0;i<p.length;i++)p[i]=i==axis?x:outer[j++];return p;}
    private static long longAt(io.github.pho001.synaptik.prepare.analysis.PrepareContext<?> c,int role,Object a,long[] p){return ((Number)get(a,c.values().get(role).descriptor().dataType(),address(c,role,p))).longValue();}
    private static void fillInput(io.github.pho001.synaptik.prepare.analysis.PrepareContext<?> c,Object a,DataType t){long[] s=shape(c.values().getFirst().descriptor().shape());for(long i=0,n=elems(s);i<n;i++){double v=switch((int)(i%6)){case 0->3;case 1->-0.0;case 2->0.0;case 3->3;case 4->-2;default->1;};set(a,t,address(c,0,coords(i,s)),represented(t,v));}}
    private static void fillIndices(io.github.pho001.synaptik.prepare.analysis.PrepareContext<?> c,Object a,DataType t){long[] s=shape(c.values().get(1).descriptor().shape());for(long i=0,n=elems(s);i<n;i++)set(a,t,address(c,1,coords(i,s)),represented(t,i%3));}
    private static long capacity(io.github.pho001.synaptik.prepare.analysis.PrepareContext<?> c,int i){var d=c.values().get(i).descriptor();var l=d.layout().orElseThrow();long m=l.storageOffset();long[] s=l.strides(),q=shape(d.shape());for(int j=0;j<s.length;j++)m+=s[j]*(q[j]-1);return m+3;}
    private static long address(io.github.pho001.synaptik.prepare.analysis.PrepareContext<?> c,int i,long[] p){var l=c.values().get(i).descriptor().layout().orElseThrow();long x=l.storageOffset();long[] s=l.strides();for(int j=0;j<p.length;j++)x+=s[j]*p[j];return x;}
    private static long[] shape(io.github.pho001.synaptik.model.shape.Shape s){return s.dimensions().stream().mapToLong(d->((io.github.pho001.synaptik.model.shape.StaticDimension)d).size()).toArray();}
    private static long elems(long[] s){long n=1;for(long x:s)n*=x;return n;} private static long[] coords(long n,long[] s){long[] p=new long[s.length];for(int i=s.length-1;i>=0;i--){p[i]=n%s[i];n/=s[i];}return p;}
    private static double number(DataType t,Object v){return t==DataType.BFLOAT16?Float.intBitsToFloat(((Short)v&0xffff)<<16):t==DataType.BOOL?((Byte)v):((Number)v).doubleValue();}
    private static Object represented(DataType t,double v){return switch(t){case FLOAT64->v;case FLOAT32->(float)v;case BFLOAT16->bfloat((float)v);case INT64->(long)v;case INT32->(int)v;case BOOL->(byte)(v!=0?1:0);};}
    private static short bfloat(float v){int b=Float.floatToRawIntBits(v),u=b>>>16,l=b&0xffff;if((b&0x7f800000)==0x7f800000&&(b&0x7fffff)!=0)u|=0x40;else if(l>0x8000||l==0x8000&&(u&1)!=0)u++;return(short)u;}
    private static Object storage(DataType t,long n){return switch(t){case FLOAT64->new double[(int)n];case FLOAT32->new float[(int)n];case BFLOAT16->new short[(int)n];case INT64->new long[(int)n];case INT32->new int[(int)n];case BOOL->new byte[(int)n];};}
    private static Object copy(Object x){int n=Array.getLength(x);Object y=Array.newInstance(x.getClass().componentType(),n);System.arraycopy(x,0,y,0,n);return y;} private static void fill(Object a,DataType t,int v){for(int i=0;i<Array.getLength(a);i++)set(a,t,i,represented(t,v));}
    private static Object get(Object a,DataType t,long i){return switch(t){case FLOAT64->((double[])a)[(int)i];case FLOAT32->((float[])a)[(int)i];case BFLOAT16->((short[])a)[(int)i];case INT64->((long[])a)[(int)i];case INT32->((int[])a)[(int)i];case BOOL->((byte[])a)[(int)i];};}
    private static void set(Object a,DataType t,long i,Object v){switch(t){case FLOAT64->((double[])a)[(int)i]=(Double)v;case FLOAT32->((float[])a)[(int)i]=(Float)v;case BFLOAT16->((short[])a)[(int)i]=(Short)v;case INT64->((long[])a)[(int)i]=(Long)v;case INT32->((int[])a)[(int)i]=(Integer)v;case BOOL->((byte[])a)[(int)i]=(Byte)v;}}
    private static MemorySegment segment(Object a){if(a instanceof double[] x)return MemorySegment.ofArray(x);if(a instanceof float[] x)return MemorySegment.ofArray(x);if(a instanceof short[] x)return MemorySegment.ofArray(x);if(a instanceof long[] x)return MemorySegment.ofArray(x);if(a instanceof int[] x)return MemorySegment.ofArray(x);return MemorySegment.ofArray((byte[])a);}
    private static void assertRaw(Object e,Object a,String m){if(e instanceof double[] x)assertArrayEquals(x,(double[])a,m);else if(e instanceof float[] x)assertArrayEquals(x,(float[])a,m);else if(e instanceof short[] x)assertArrayEquals(x,(short[])a,m);else if(e instanceof long[] x)assertArrayEquals(x,(long[])a,m);else if(e instanceof int[] x)assertArrayEquals(x,(int[])a,m);else assertArrayEquals((byte[])e,(byte[])a,m);}
}
