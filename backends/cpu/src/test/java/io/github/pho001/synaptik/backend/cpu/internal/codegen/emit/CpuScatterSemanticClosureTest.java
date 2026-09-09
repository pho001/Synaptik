package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Direct, per-inventory-row generated-entry semantics for ordinary scatter. */
class CpuScatterSemanticClosureTest {
    @Test void everyOrdinaryScatterInventoryCandidateExecutesAgainstTheIndependentOracle() throws Throwable {
        var candidates = CpuOrdinaryNonPointwiseGeneratedMatrixTest.scatterCandidates();
        assertEquals(416, candidates.size());
        assertEquals(208, candidates.stream().filter(c -> c.ownerId().startsWith("scatter-elements-")).count());
        assertEquals(208, candidates.stream().filter(c -> c.ownerId().startsWith("scatter-nd-")).count());
        for (var candidate : candidates) execute(candidate);
    }

    private static void execute(CpuOrdinaryNonPointwiseGeneratedMatrixTest.ScatterCandidate candidate) throws Throwable {
        var context = candidate.context();
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] first = generator.generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(first, generator.generateClassBytes(route.specialization(), route.kernelIr()), candidate.ownerId());
        var entry = generator.defineClassBytes(route.specialization(), first).entryPoint();
        DataType valueType = context.values().getFirst().descriptor().dataType();
        DataType indexType = context.values().get(1).descriptor().dataType();
        Object base = storage(valueType, capacity(context, 0));
        Object indexes = storage(indexType, capacity(context, 1));
        Object updates = storage(valueType, capacity(context, 2));
        Object actual = storage(valueType, capacity(context, 3));
        Object expected = storage(valueType, capacity(context, 3));
        fill(base, valueType, 2); fill(updates, valueType, 1); fill(actual, valueType, -77); fill(expected, valueType, -77);
        fillIndexes(context, indexes, indexType);
        long start = 1, end = plan.elementCount() - 1;
        oracle(context, base, indexes, updates, expected, start, end);
        var raw = List.of(base, indexes, updates, actual);
        var carriers = new ArrayList<Object>();
        for (int i = 0; i < raw.size(); i++) carriers.add(route.specialization().carrierPattern().get(i) == CarrierAccess.MEMORY_SEGMENT ? segment(raw.get(i)) : raw.get(i));
        long[] geometry = plan.scatterGeometry().orElseThrow().pack(new long[4], start, end, 0);
        try (Arena arena = Arena.ofConfined()) {
            var args = new ArrayList<>(carriers);
            if (plan.workspaceDeclaration().isPresent()) args.add(arena.allocate(plan.workspaceDeclaration().orElseThrow().byteSize(), 8));
            args.add(geometry); args.add(start); args.add(end); entry.invokeWithArguments(args);
        }
        assertRaw(expected, actual, candidate.ownerId());
    }

    private static void oracle(io.github.pho001.synaptik.prepare.analysis.PrepareContext<?> c, Object base, Object indexes,
            Object updates, Object output, long start, long end) {
        var kind = c.nodes().getFirst().operation().kind(); DataType type = c.values().getFirst().descriptor().dataType();
        ScatterReduction reduction = kind == AxisScatterKind.SCATTER_ELEMENTS
                ? ((io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs)c.nodes().getFirst().operation().attrs()).reduction()
                : ((io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs)c.nodes().getFirst().operation().attrs()).reduction();
        long[] shape = dimensions(c.values().get(3).descriptor().shape());
        for (long ordinal = start; ordinal < end; ordinal++) {
            long[] out = coordinates(ordinal, shape); long address = address(c, 3, out); set(output, type, address, get(base, type, address(c, 0, out)));
            boolean found = false;
            long[] updateShape = dimensions(c.values().get(2).descriptor().shape());
            for (long u = 0, n = elements(updateShape); u < n; u++) {
                long[] uc = coordinates(u, updateShape); long[] target;
                if (kind == AxisScatterKind.SCATTER_ELEMENTS) {
                    target = uc.clone(); target[0] = asLong(get(indexes, c.values().get(1).descriptor().dataType(), address(c, 1, uc)));
                } else {
                    target = new long[] {uc[0], asLong(get(indexes, c.values().get(1).descriptor().dataType(), address(c, 1, new long[] {uc[0], uc[1], 0}))), uc[2]};
                }
                if (!java.util.Arrays.equals(out, target)) continue;
                Object value = get(updates, type, address(c, 2, uc));
                if (!found && reduction == ScatterReduction.NONE) { set(output, type, address, value); found = true; }
                else { set(output, type, address, reduce(type, reduction, get(output, type, address), value)); found = true; }
            }
        }
    }

    private static Object reduce(DataType type, ScatterReduction r, Object a, Object b) {
        if (r == ScatterReduction.NONE) return b;
        double x = number(type, a), y = number(type, b); double z = switch (r) { case ADD -> x + y; case MUL -> x * y; case MIN -> Math.min(x, y); case MAX -> Math.max(x, y); case NONE -> y; };
        return represented(type, z);
    }
    private static double number(DataType t, Object v) { return t == DataType.BFLOAT16 ? Float.intBitsToFloat(((Short)v & 0xffff) << 16) : ((Number)v).doubleValue(); }
    private static Object represented(DataType t, double v) { return switch (t) { case FLOAT64 -> v; case FLOAT32 -> (float)v; case BFLOAT16 -> bfloat((float)v); case INT64 -> (long)v; case INT32 -> (int)v; case BOOL -> (byte)((int)v); }; }
    private static short bfloat(float value) { int bits = Float.floatToRawIntBits(value), upper = bits >>> 16, lower = bits & 0xffff; if ((bits & 0x7f800000) == 0x7f800000 && (bits & 0x7fffff) != 0) upper |= 0x40; else if (lower > 0x8000 || lower == 0x8000 && (upper & 1) != 0) upper++; return (short) upper; }
    private static long asLong(Object v) { return ((Number)v).longValue(); }
    private static long capacity(io.github.pho001.synaptik.prepare.analysis.PrepareContext<?> c, int i) { var d=c.values().get(i).descriptor(); long max=d.layout().orElseThrow().storageOffset(); long[] s=d.layout().orElseThrow().strides(), q=dimensions(d.shape()); for(int x=0;x<s.length;x++) max+=s[x]*(q[x]-1); return max+3; }
    private static long address(io.github.pho001.synaptik.prepare.analysis.PrepareContext<?> c, int i, long[] p) { var l=c.values().get(i).descriptor().layout().orElseThrow(); long x=l.storageOffset(); long[] s=l.strides(); for(int j=0;j<p.length;j++) x+=s[j]*p[j]; return x; }
    private static long elements(long[] s) { long n=1; for(long x:s)n*=x; return n; }
    private static long[] coordinates(long n,long[] s) { long[] p=new long[s.length]; for(int i=s.length-1;i>=0;i--){p[i]=n%s[i];n/=s[i];} return p; }
    private static Object storage(DataType t,long n) { return switch(t){case FLOAT64->new double[(int)n];case FLOAT32->new float[(int)n];case BFLOAT16->new short[(int)n];case INT64->new long[(int)n];case INT32->new int[(int)n];case BOOL->new byte[(int)n];}; }
    private static void fill(Object a,DataType t,int seed){for(int i=0;i<Array.getLength(a);i++)set(a,t,i,represented(t,seed));}
    private static void fillIndexes(io.github.pho001.synaptik.prepare.analysis.PrepareContext<?> c,Object x,DataType t){long[] s=dimensions(c.values().get(1).descriptor().shape());for(long i=0,n=elements(s);i<n;i++){long[] p=coordinates(i,s); set(x,t,address(c,1,p),represented(t,p[p.length-1]%3));}}
    private static long[] dimensions(io.github.pho001.synaptik.model.shape.Shape shape) { return shape.dimensions().stream().mapToLong(d -> ((io.github.pho001.synaptik.model.shape.StaticDimension) d).size()).toArray(); }
    private static Object get(Object a,DataType t,long i){return switch(t){case FLOAT64->((double[])a)[(int)i];case FLOAT32->((float[])a)[(int)i];case BFLOAT16->((short[])a)[(int)i];case INT64->((long[])a)[(int)i];case INT32->((int[])a)[(int)i];case BOOL->((byte[])a)[(int)i];};}
    private static void set(Object a,DataType t,long i,Object v){switch(t){case FLOAT64->((double[])a)[(int)i]=(Double)v;case FLOAT32->((float[])a)[(int)i]=(Float)v;case BFLOAT16->((short[])a)[(int)i]=(Short)v;case INT64->((long[])a)[(int)i]=(Long)v;case INT32->((int[])a)[(int)i]=(Integer)v;case BOOL->((byte[])a)[(int)i]=(Byte)v;}}
    private static MemorySegment segment(Object a){if(a instanceof double[] x)return MemorySegment.ofArray(x);if(a instanceof float[] x)return MemorySegment.ofArray(x);if(a instanceof short[] x)return MemorySegment.ofArray(x);if(a instanceof long[] x)return MemorySegment.ofArray(x);if(a instanceof int[] x)return MemorySegment.ofArray(x);return MemorySegment.ofArray((byte[])a);}
    private static void assertRaw(Object e,Object a,String m){if(e instanceof double[] x)assertArrayEquals(x,(double[])a,m);else if(e instanceof float[] x)assertArrayEquals(x,(float[])a,m);else if(e instanceof short[] x)assertArrayEquals(x,(short[])a,m);else if(e instanceof long[] x)assertArrayEquals(x,(long[])a,m);else if(e instanceof int[] x)assertArrayEquals(x,(int[])a,m);else assertArrayEquals((byte[])e,(byte[])a,m);}
}
