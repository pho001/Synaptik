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
 * Emits a direct generated bridge to the CPU-owned ordinary aggregate body.
 *
 * <p>The static body reduces complete output cells only. It traverses every selected domain in
 * logical input row-major order, selects the first represented NaN, applies explicit signed-zero
 * extrema rules, and allocates no per-cell or per-element object. The generated class contains a
 * Class-File bridge to that body rather than an embedded reduction loop.</p>
 */
public final class CpuAggregateEmitter {
    private static final ClassDesc OWNER = ClassDesc.of(CpuAggregateEmitter.class.getName());
    private static final DataType[] TYPES = DataType.values();
    /** Creates a stateless bridge emitter. */
    public CpuAggregateEmitter() { }

    /**
     * Emits one direct two-boundary scalar bridge.
     * @param code non-null Class-File method builder mutated by this call
     * @param specialization non-null two-boundary, scratch-free specialization
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the specialization has another boundary shape
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization) {
        if (specialization.carrierPattern().size() != 2 || specialization.scratchParameter())
            throw new IllegalArgumentException("aggregate requires two boundaries and no scratch");
        code.aload(0).aload(1).aload(2).lload(3).lload(5);
        code.invokestatic(OWNER, "execute", MethodTypeDesc.of(ConstantDescs.CD_void,
                ConstantDescs.CD_Object, ConstantDescs.CD_Object,
                ConstantDescs.CD_long.arrayType(), ConstantDescs.CD_long, ConstantDescs.CD_long));
    }

    /**
     * Reduces complete flattened output cells in {@code [start,end)}.
     * @param input non-null readable primitive-array or native-segment carrier
     * @param output non-null writable non-overlapping carrier
     * @param packed non-null invocation-owned geometry and mutable coordinate state
     * @param start non-negative inclusive output-cell ordinal
     * @param end exclusive output-cell ordinal no greater than the output count
     * @throws NullPointerException if a carrier or {@code packed} is {@code null}
     * @throws ClassCastException if a carrier does not match the represented type selected during
     *     cold specialization
     * @throws ArithmeticException if an array address cannot be represented as {@code int}
     * @throws IndexOutOfBoundsException if a carrier does not cover a packed address
     * @throws IllegalStateException if a supplied memory segment is inaccessible
     */
    public static void execute(Object input, Object output, long[] packed, long start, long end) {
        int kind = (int) packed[0]; DataType type = TYPES[(int) packed[1]];
        boolean keep = packed[3] != 0; int inRank = (int) packed[4], outRank = (int) packed[5];
        long domainCount = packed[7]; int selected = 8;
        int inputCoordinates = selected + inRank;
        int outputCoordinates = inputCoordinates + inRank;
        int inputLayout = outputCoordinates + outRank;
        int outputLayout = inputLayout + 2 + 2 * inRank;
        for (long cell = start; cell < end; cell++) {
            decode(cell, packed, outputCoordinates, packed, outputLayout);
            int outAxis = 0;
            for (int axis = 0; axis < inRank; axis++) {
                if (packed[selected + axis] != 0) packed[inputCoordinates + axis] = 0;
                else packed[inputCoordinates + axis] = packed[outputCoordinates
                        + (keep ? axis : outAxis++)];
            }
            long accumulator = identity(kind, type);
            for (long domain = 0; domain < domainCount; domain++) {
                long remaining = domain;
                for (int axis = inRank - 1; axis >= 0; axis--) if (packed[selected + axis] != 0) {
                    long extent = packed[inputLayout + 2 + axis];
                    packed[inputCoordinates + axis] = remaining % extent; remaining /= extent;
                }
                long value = readBits(input, address(packed, inputLayout, inputCoordinates), type);
                accumulator = apply(kind, accumulator, value, type);
            }
            writeBits(output, address(packed, outputLayout, outputCoordinates), type, accumulator);
        }
    }

    private static void decode(long logical, long[] coordinatesOwner, int coordinates,
            long[] layoutOwner, int layout) {
        int rank = (int) layoutOwner[layout];
        for (int axis = rank - 1; axis >= 0; axis--) {
            long extent = layoutOwner[layout + 2 + axis];
            coordinatesOwner[coordinates + axis] = logical % extent; logical /= extent;
        }
    }
    private static long identity(int kind, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits(kind == 0
                    ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY);
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(kind == 0
                    ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY));
            case BFLOAT16 -> kind == 0 ? 0x7f80L : 0xff80L;
            case INT32 -> kind == 0 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            case INT64 -> kind == 0 ? Long.MAX_VALUE : Long.MIN_VALUE;
            case BOOL -> kind == 2 ? 1 : 0;
        };
    }
    private static long apply(int kind, long left, long right, DataType type) {
        if (type == DataType.BOOL) return kind == 2
                ? ((left != 0 && right != 0) ? 1 : 0) : ((left != 0 || right != 0) ? 1 : 0);
        if (type == DataType.INT32) return kind == 0
                ? Math.min((int) left, (int) right) : Math.max((int) left, (int) right);
        if (type == DataType.INT64) return kind == 0 ? Math.min(left, right) : Math.max(left, right);
        if (isNaN(left, type)) return left;
        if (isNaN(right, type)) return right;
        double l = floatingValue(left, type), r = floatingValue(right, type);
        if (l == 0.0 && r == 0.0) {
            boolean leftNegative = negative(left, type), rightNegative = negative(right, type);
            if (kind == 0) return leftNegative ? left : rightNegative ? right : left;
            return !leftNegative ? left : !rightNegative ? right : left;
        }
        return kind == 0 ? (r < l ? right : left) : (r > l ? right : left);
    }
    private static boolean isNaN(long bits, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.isNaN(Double.longBitsToDouble(bits));
            case FLOAT32 -> Float.isNaN(Float.intBitsToFloat((int) bits));
            case BFLOAT16 -> ((bits & 0x7f80L) == 0x7f80L) && (bits & 0x7fL) != 0;
            default -> false;
        };
    }
    private static double floatingValue(long bits, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.longBitsToDouble(bits);
            case FLOAT32 -> Float.intBitsToFloat((int) bits);
            case BFLOAT16 -> Float.intBitsToFloat((int) bits << 16);
            default -> throw new AssertionError("non-floating aggregate type");
        };
    }
    private static boolean negative(long bits, DataType type) {
        return switch (type) {
            case FLOAT64 -> bits < 0;
            case FLOAT32 -> ((int) bits) < 0;
            case BFLOAT16 -> ((short) bits) < 0;
            default -> false;
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
            case BOOL -> carrier instanceof byte[] a ? Byte.toUnsignedLong(a[Math.toIntExact(address)]) : Byte.toUnsignedLong(((MemorySegment) carrier).get(ValueLayout.JAVA_BYTE, address));
        };
    }
    private static void writeBits(Object carrier, long address, DataType type, long bits) {
        switch (type) {
            case FLOAT64 -> { double v = Double.longBitsToDouble(bits); if (carrier instanceof double[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_DOUBLE, address * 8, v); }
            case FLOAT32 -> { float v = Float.intBitsToFloat((int) bits); if (carrier instanceof float[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_FLOAT, address * 4, v); }
            case BFLOAT16 -> { short v = (short) bits; if (carrier instanceof short[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_SHORT, address * 2, v); }
            case INT32 -> { int v = (int) bits; if (carrier instanceof int[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_INT, address * 4, v); }
            case INT64 -> { if (carrier instanceof long[] a) a[Math.toIntExact(address)] = bits; else ((MemorySegment) carrier).set(ValueLayout.JAVA_LONG, address * 8, bits); }
            case BOOL -> { byte v = (byte) (bits == 0 ? 0 : 1); if (carrier instanceof byte[] a) a[Math.toIntExact(address)] = v; else ((MemorySegment) carrier).set(ValueLayout.JAVA_BYTE, address, v); }
        }
    }
}
