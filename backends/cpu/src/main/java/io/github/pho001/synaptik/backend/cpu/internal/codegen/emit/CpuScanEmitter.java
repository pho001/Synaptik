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
 * Emits a generated direct bridge to the CPU-owned scan body for independent slice ranges.
 *
 * <p>Each range contains whole logical slices only. Within a slice, execution visits the selected
 * axis sequentially in forward or reverse order and applies inclusive or exclusive placement.
 * FLOAT64 and FLOAT32 retain same-format results, BFLOAT16 widens to FLOAT32 and rounds back after
 * every operation, and INT32/INT64 wrap at their represented width. The body allocates no
 * per-slice or per-element object and owns no worker, workspace, or persistent accumulator.</p>
 */
public final class CpuScanEmitter {
    private static final ClassDesc OWNER = ClassDesc.of(CpuScanEmitter.class.getName());
    private static final DataType[] TYPES = DataType.values();
    /** Creates a stateless emitter with no retained specialization or invocation state. */
    public CpuScanEmitter() { }

    /**
     * Emits one direct two-boundary scalar bridge to the CPU-owned scan body.
     *
     * @param code non-null Class-File method builder mutated with the bridge instructions
     * @param specialization non-null scalar, two-boundary, workspace-free specialization; read
     *     during emission and not retained
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the specialization has another boundary or scratch
     *     shape
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization) {
        if (specialization.carrierPattern().size() != 2 || specialization.scratchParameter())
            throw new IllegalArgumentException("scan requires two boundaries and no scratch");
        code.aload(0).aload(1).aload(2).lload(3).lload(5);
        code.invokestatic(OWNER, "execute", MethodTypeDesc.of(ConstantDescs.CD_void,
                ConstantDescs.CD_Object, ConstantDescs.CD_Object,
                ConstantDescs.CD_long.arrayType(), ConstantDescs.CD_long, ConstantDescs.CD_long));
    }

    /**
     * Executes complete scan slices in {@code [start,end)} without splitting a slice.
     *
     * <p>The packed geometry is invocation-owned mutable coordinate state. Input and output are
     * borrowed exact primitive arrays or accessible native-order memory segments; this method
     * neither retains nor closes them.</p>
     *
     * @param input non-null readable carrier matching the packed data type
     * @param output non-null writable, non-overlapping carrier matching the packed data type
     * @param packed non-null invocation-owned scan geometry and coordinate state; mutated during
     *     execution and not retained
     * @param start non-negative inclusive independent-slice ordinal
     * @param end exclusive independent-slice ordinal no greater than the packed slice count
     * @throws NullPointerException if a required carrier or {@code packed} is {@code null}
     * @throws ArithmeticException if an array address is outside the Java index range
     * @throws IndexOutOfBoundsException if a carrier cannot cover a packed address
     * @throws IllegalStateException if a supplied memory segment is inaccessible
     */
    public static void execute(Object input, Object output, long[] packed, long start, long end) {
        int kind = (int) packed[0]; DataType type = TYPES[(int) packed[1]];
        int rank = (int) packed[2], axis = (int) packed[3];
        boolean exclusive = packed[4] != 0, reverse = packed[5] != 0;
        long axisExtent = packed[7]; int coordinates = 8;
        int inputLayout = coordinates + rank;
        int outputLayout = inputLayout + 2 + 2 * rank;
        for (long slice = start; slice < end; slice++) {
            long remaining = slice;
            for (int current = rank - 1; current >= 0; current--) {
                if (current == axis) { packed[coordinates + current] = 0; continue; }
                long extent = packed[inputLayout + 2 + current];
                packed[coordinates + current] = remaining % extent;
                remaining /= extent;
            }
            long accumulator = identity(kind, type);
            for (long step = 0; step < axisExtent; step++) {
                long coordinate = reverse ? axisExtent - 1 - step : step;
                packed[coordinates + axis] = coordinate;
                long value = readBits(input, address(packed, inputLayout, coordinates), type);
                if (!exclusive) accumulator = apply(kind, accumulator, value, type);
                writeBits(output, address(packed, outputLayout, coordinates), type, accumulator);
                if (exclusive) accumulator = apply(kind, accumulator, value, type);
            }
        }
    }

    private static long identity(int kind, DataType type) {
        if (kind == 0) return 0;
        return switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits(1.0d);
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(1.0f));
            case BFLOAT16 -> 0x3f80L;
            case INT32, INT64 -> 1;
            case BOOL -> throw new AssertionError("BOOL scan is unsupported");
        };
    }
    private static long apply(int kind, long left, long right, DataType type) {
        return switch (type) {
            case INT32 -> kind == 0 ? (int) left + (int) right : (int) left * (int) right;
            case INT64 -> kind == 0 ? left + right : left * right;
            case FLOAT64 -> Double.doubleToRawLongBits(kind == 0
                    ? Double.longBitsToDouble(left) + Double.longBitsToDouble(right)
                    : Double.longBitsToDouble(left) * Double.longBitsToDouble(right));
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(kind == 0
                    ? Float.intBitsToFloat((int) left) + Float.intBitsToFloat((int) right)
                    : Float.intBitsToFloat((int) left) * Float.intBitsToFloat((int) right)));
            case BFLOAT16 -> Short.toUnsignedLong(floatToBfloat(kind == 0
                    ? bfloatToFloat((short) left) + bfloatToFloat((short) right)
                    : bfloatToFloat((short) left) * bfloatToFloat((short) right)));
            case BOOL -> throw new AssertionError("BOOL scan is unsupported");
        };
    }
    private static long address(long[] p, int layout, int coordinates) {
        long result = p[layout + 1]; int rank = (int) p[layout];
        for (int axis = 0; axis < rank; axis++)
            result += p[coordinates + axis] * p[layout + 2 + rank + axis];
        return result;
    }
    private static long readBits(Object carrier, long address, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits(carrier instanceof double[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_DOUBLE, address * 8));
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(carrier instanceof float[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_FLOAT, address * 4)));
            case BFLOAT16 -> Short.toUnsignedLong(carrier instanceof short[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_SHORT, address * 2));
            case INT32 -> carrier instanceof int[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_INT, address * 4);
            case INT64 -> carrier instanceof long[] a ? a[Math.toIntExact(address)] : ((MemorySegment) carrier).get(ValueLayout.JAVA_LONG, address * 8);
            case BOOL -> throw new AssertionError("BOOL scan is unsupported");
        };
    }
    private static void writeBits(Object carrier, long address, DataType type, long bits) {
        switch (type) {
            case FLOAT64 -> { double v = Double.longBitsToDouble(bits); if (carrier instanceof double[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_DOUBLE, address * 8, v); }
            case FLOAT32 -> { float v = Float.intBitsToFloat((int) bits); if (carrier instanceof float[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_FLOAT, address * 4, v); }
            case BFLOAT16 -> { short v = (short) bits; if (carrier instanceof short[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_SHORT, address * 2, v); }
            case INT32 -> { int v = (int) bits; if (carrier instanceof int[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_INT, address * 4, v); }
            case INT64 -> { if (carrier instanceof long[] a) a[Math.toIntExact(address)] = bits; else ((MemorySegment) carrier).set(ValueLayout.JAVA_LONG, address * 8, bits); }
            case BOOL -> throw new AssertionError("BOOL scan is unsupported");
        }
    }
    private static float bfloatToFloat(short bits) { return Float.intBitsToFloat(Short.toUnsignedInt(bits) << 16); }
    private static short floatToBfloat(float value) {
        int bits = Float.floatToRawIntBits(value);
        if ((bits & 0x7f800000) == 0x7f800000 && (bits & 0x7fffff) != 0)
            return (short) ((bits >>> 16) | 0x40);
        int upper = bits >>> 16, lower = bits & 0xffff;
        if (lower > 0x8000 || lower == 0x8000 && (upper & 1) != 0) upper++;
        return (short) upper;
    }
}
