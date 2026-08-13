package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Emits the direct generated bridge and owns stable primitive-index ordering execution.
 *
 * <p>Each invocation orders complete independent logical-axis slices with a bottom-up stable
 * merge over two assigned INT64 scratch regions. Value comparison is allocation-free per
 * element: floating values use NaN-last order and directional signed zero, integral values use
 * signed order, and BOOL uses canonical byte order. SORT and TOP_K copy the selected represented
 * bits unchanged; ARGSORT and TOP_K write zero-based INT64 logical-axis coordinates. Unsorted
 * TOP_K deterministically reorders only the selected pairs by increasing original index.</p>
 */
public final class CpuOrderingEmitter {
    private static final ClassDesc OWNER = ClassDesc.of(CpuOrderingEmitter.class.getName());
    private static final DataType[] TYPES = DataType.values();

    /** Creates a stateless ordering emitter; it owns no carrier or workspace lifetime. */
    public CpuOrderingEmitter() { }

    /**
     * Emits one two- or three-boundary scalar ordering bridge with exact scratch.
     *
     * @param code non-null Class-File method body receiving the direct ordered carriers,
     *     workspace segment, packed geometry, and primitive slice bounds
     * @param specialization non-null selected scalar specialization with two or three boundaries
     *     and an explicit scratch parameter
     * @throws NullPointerException if a required argument is null
     * @throws IllegalArgumentException if boundary cardinality or the scratch signature is invalid
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization) {
        int count = specialization.carrierPattern().size();
        if ((count != 2 && count != 3) || !specialization.scratchParameter())
            throw new IllegalArgumentException("ordering requires two or three boundaries and scratch");
        code.aload(0).aload(1);
        if (count == 3) code.aload(2); else code.aconst_null();
        int scratch = count;
        code.aload(scratch).aload(scratch + 1).lload(scratch + 2).lload(scratch + 4);
        code.invokestatic(OWNER, "execute", MethodTypeDesc.of(ConstantDescs.CD_void,
                ConstantDescs.CD_Object, ConstantDescs.CD_Object, ConstantDescs.CD_Object,
                ClassDesc.of(MemorySegment.class.getName()), ConstantDescs.CD_long.arrayType(),
                ConstantDescs.CD_long, ConstantDescs.CD_long));
    }

    /**
     * Executes complete independent logical-axis slices in {@code [start,end)}.
     *
     * <p>The supplied workspace is borrowed for the call. The packed range index selects one
     * disjoint region, so parallel-scalar calls share neither scratch coordinates nor outputs.
     * The method mutates only the declared output carriers and its assigned scratch region.</p>
     *
     * @param input non-null primitive array or accessible native-order segment containing values
     * @param firstOutput non-null writable values output for SORT/TOP_K or INT64 index output for
     *     ARGSORT
     * @param secondOutput writable INT64 TOP_K index output, or {@code null} otherwise
     * @param scratch non-null accessible writable run-owned workspace
     * @param p non-null packed family, type, range, scratch, and boundary layout geometry
     * @param start inclusive logical-slice ordinal
     * @param end exclusive logical-slice ordinal
     * @throws NullPointerException if a required carrier, workspace, or geometry array is null
     * @throws IllegalArgumentException if a carrier has an unsupported runtime form
     * @throws IndexOutOfBoundsException if packed geometry does not fit a supplied carrier or
     *     assigned scratch region
     * @throws ArithmeticException if exact range or scratch arithmetic overflows
     * @throws IllegalStateException if a supplied segment is not accessible
     */
    public static void execute(Object input, Object firstOutput, Object secondOutput,
            MemorySegment scratch, long[] p, long start, long end) {
        int family = (int) p[0]; DataType type = TYPES[(int) p[1]]; int axis = (int) p[2];
        long k = p[3]; boolean descending = p[4] != 0; boolean sorted = p[5] != 0;
        int boundaryCount = (int) p[6]; long scratchSliceBytes = p[7]; long rangeIndex = p[8];
        int[] layout = new int[boundaryCount]; int x = 11;
        for (int i = 0; i < boundaryCount; i++) {
            layout[i] = x; int rank = (int) p[x]; x += 2 + 2 * rank;
        }
        int rank = (int) p[layout[0]]; long axisExtent = p[layout[0] + 2 + axis];
        long region = Math.multiplyExact(axisExtent, Long.BYTES);
        long scratchBase = Math.multiplyExact(rangeIndex, scratchSliceBytes);
        for (long slice = start; slice < end; slice++) {
            for (long i = 0; i < axisExtent; i++) scratch.set(ValueLayout.JAVA_LONG,
                    scratchBase + i * Long.BYTES, i);
            long source = scratchBase, target = scratchBase + region;
            for (long width = 1; width < axisExtent; width = Math.multiplyExact(width, 2)) {
                for (long left = 0; left < axisExtent; left += Math.multiplyExact(width, 2)) {
                    long middle = Math.min(left + width, axisExtent);
                    long right = Math.min(left + 2 * width, axisExtent);
                    long a = left, b = middle, out = left;
                    while (a < middle && b < right) {
                        long ia = scratch.get(ValueLayout.JAVA_LONG, source + a * 8);
                        long ib = scratch.get(ValueLayout.JAVA_LONG, source + b * 8);
                        if (compare(input, p, layout[0], slice, axis, ia, ib, type, descending) <= 0) {
                            scratch.set(ValueLayout.JAVA_LONG, target + out++ * 8, ia); a++;
                        } else { scratch.set(ValueLayout.JAVA_LONG, target + out++ * 8, ib); b++; }
                    }
                    while (a < middle) scratch.set(ValueLayout.JAVA_LONG, target + out++ * 8,
                            scratch.get(ValueLayout.JAVA_LONG, source + a++ * 8));
                    while (b < right) scratch.set(ValueLayout.JAVA_LONG, target + out++ * 8,
                            scratch.get(ValueLayout.JAVA_LONG, source + b++ * 8));
                }
                long swap = source; source = target; target = swap;
            }
            if (family == 2 && !sorted) insertionIndexOrder(scratch, source, k);
            long outputCount = family == 2 ? k : axisExtent;
            for (long position = 0; position < outputCount; position++) {
                long index = scratch.get(ValueLayout.JAVA_LONG, source + position * 8);
                if (family != 1) copyValue(input, firstOutput, p, layout[0], layout[1], slice,
                        axis, index, position, type, rank);
                else writeIndex(firstOutput, address(p, layout[1], slice, axis, position), index);
                if (family == 2) writeIndex(secondOutput,
                        address(p, layout[2], slice, axis, position), index);
            }
        }
    }

    private static void insertionIndexOrder(MemorySegment scratch, long base, long count) {
        for (long i = 1; i < count; i++) {
            long value = scratch.get(ValueLayout.JAVA_LONG, base + i * 8); long j = i;
            while (j > 0 && scratch.get(ValueLayout.JAVA_LONG, base + (j - 1) * 8) > value) {
                scratch.set(ValueLayout.JAVA_LONG, base + j * 8,
                        scratch.get(ValueLayout.JAVA_LONG, base + (j - 1) * 8)); j--;
            }
            scratch.set(ValueLayout.JAVA_LONG, base + j * 8, value);
        }
    }

    private static int compare(Object input, long[] p, int layout, long slice, int axis,
            long left, long right, DataType type, boolean descending) {
        long a = readBits(input, address(p, layout, slice, axis, left), type);
        long b = readBits(input, address(p, layout, slice, axis, right), type);
        int result = switch (type) {
            case FLOAT64 -> floatingCompare(Double.longBitsToDouble(a), Double.longBitsToDouble(b), descending);
            case FLOAT32 -> floatingCompare(Float.intBitsToFloat((int) a), Float.intBitsToFloat((int) b), descending);
            case BFLOAT16 -> floatingCompare(Float.intBitsToFloat(((int) a & 0xffff) << 16),
                    Float.intBitsToFloat(((int) b & 0xffff) << 16), descending);
            case INT32 -> Integer.compare((int) a, (int) b);
            case INT64 -> Long.compare(a, b);
            case BOOL -> Byte.compare((byte) a, (byte) b);
        };
        return descending && type != DataType.FLOAT64 && type != DataType.FLOAT32
                && type != DataType.BFLOAT16 ? -result : result;
    }

    private static int floatingCompare(double a, double b, boolean descending) {
        boolean an = Double.isNaN(a), bn = Double.isNaN(b);
        if (an || bn) return an == bn ? 0 : an ? 1 : -1;
        int comparison = Double.compare(a, b);
        return descending ? -comparison : comparison;
    }

    private static long address(long[] p, int layout, long slice, int axis, long axisIndex) {
        int rank = (int) p[layout]; long address = p[layout + 1]; long remainder = slice;
        for (int current = rank - 1; current >= 0; current--) {
            long coordinate;
            if (current == axis) coordinate = axisIndex;
            else { long extent = p[layout + 2 + current]; coordinate = extent == 0 ? 0 : remainder % extent;
                if (extent != 0) remainder /= extent; }
            address += coordinate * p[layout + 2 + rank + current];
        }
        return address;
    }

    private static void copyValue(Object input, Object output, long[] p, int inLayout,
            int outLayout, long slice, int axis, long inputIndex, long outputIndex,
            DataType type, int rank) {
        long bits = readBits(input, address(p, inLayout, slice, axis, inputIndex), type);
        writeBits(output, address(p, outLayout, slice, axis, outputIndex), type, bits);
    }

    private static long readBits(Object carrier, long address, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits(carrier instanceof double[] a ? a[Math.toIntExact(address)]
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_DOUBLE, address * 8));
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(carrier instanceof float[] a
                    ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_FLOAT, address * 4)));
            case BFLOAT16 -> Short.toUnsignedLong(carrier instanceof short[] a ? a[Math.toIntExact(address)]
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_SHORT, address * 2));
            case INT32 -> carrier instanceof int[] a ? a[Math.toIntExact(address)]
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_INT, address * 4);
            case INT64 -> carrier instanceof long[] a ? a[Math.toIntExact(address)]
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_LONG, address * 8);
            case BOOL -> carrier instanceof byte[] a ? a[Math.toIntExact(address)]
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_BYTE, address);
        };
    }

    private static void writeBits(Object carrier, long address, DataType type, long bits) {
        switch (type) {
            case FLOAT64 -> { double v = Double.longBitsToDouble(bits); if (carrier instanceof double[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_DOUBLE, address * 8, v); }
            case FLOAT32 -> { float v = Float.intBitsToFloat((int) bits); if (carrier instanceof float[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_FLOAT, address * 4, v); }
            case BFLOAT16 -> { short v = (short) bits; if (carrier instanceof short[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_SHORT, address * 2, v); }
            case INT32 -> { int v = (int) bits; if (carrier instanceof int[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_INT, address * 4, v); }
            case INT64 -> writeIndex(carrier, address, bits);
            case BOOL -> { byte v = (byte) bits; if (carrier instanceof byte[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_BYTE, address, v); }
        }
    }

    private static void writeIndex(Object carrier, long address, long value) {
        if (carrier instanceof long[] a) a[Math.toIntExact(address)] = value;
        else ((MemorySegment) carrier).set(ValueLayout.JAVA_LONG, address * 8, value);
    }
}
