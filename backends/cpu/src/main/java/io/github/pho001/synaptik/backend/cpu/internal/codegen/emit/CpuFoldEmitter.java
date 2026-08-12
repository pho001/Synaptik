package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * Emits the direct generated fold bridge and owns allocation-free output-domain fold writers.
 * Every owned output coordinate starts from represented positive zero, scans logical input
 * occurrences in canonical row-major order, and is written exactly once. General-axis integral
 * folds use modular addition; both floating families use sequential same-format addition, with
 * BFLOAT16 rounding after every contribution.
 */
public final class CpuFoldEmitter {
    private static final ClassDesc OWNER = ClassDesc.of(CpuFoldEmitter.class.getName());
    private static final DataType[] TYPES = DataType.values();

    /** Creates a stateless fold emitter that owns no generated artifact or invocation state. */
    public CpuFoldEmitter() { }

    /**
     * Emits one two-boundary scalar fold bridge.
     * @param code non-null generated method body
     * @param specialization non-null matching scalar fold specialization
     * @throws IllegalArgumentException if the specialization is not an exact two-boundary,
     *     workspace-free scalar form
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization) {
        if (specialization.carrierPattern().size() != 2 || specialization.scratchParameter()) {
            throw new IllegalArgumentException("fold requires two boundaries and no scratch");
        }
        code.aload(0).aload(1).aload(2).lload(3).lload(5);
        code.invokestatic(OWNER, "execute",
                MethodTypeDesc.of(ConstantDescs.CD_void, ConstantDescs.CD_Object,
                        ConstantDescs.CD_Object, ConstantDescs.CD_long.arrayType(),
                        ConstantDescs.CD_long, ConstantDescs.CD_long));
    }

    /**
     * Executes one already validated disjoint output range using packed cold geometry.
     * The method reads no position outside the mapped input span and writes each logical output
     * ordinal in {@code [start, end)} once; callers must establish carrier compatibility,
     * writability, injectivity, and physical non-overlap before invocation.
     * @param input read-only input carrier
     * @param output writable output carrier
     * @param packed invocation-private layout and mapping geometry
     * @param start inclusive flattened output ordinal
     * @param end exclusive flattened output ordinal
     * @throws NullPointerException if a carrier or {@code packed} geometry is {@code null}
     * @throws RuntimeException if callers violate the internal validated carrier, geometry, or
     *     range contract
     */
    public static void execute(Object input, Object output, long[] packed, long start, long end) {
        int family = (int) packed[0];
        DataType type = TYPES[(int) packed[1]];
        int inputRank = (int) packed[2], outputRank = (int) packed[3];
        int inputCoordinates = 8;
        int outputCoordinates = inputCoordinates + inputRank;
        int outputSeed = outputCoordinates + outputRank;
        int inputLayout = outputSeed + outputRank;
        int outputLayout = inputLayout + 2 + 2 * inputRank;
        int mapping = outputLayout + 2 + 2 * outputRank;
        System.arraycopy(packed, outputSeed, packed, outputCoordinates, outputRank);
        long inputCount = packed[6];
        for (long outputOrdinal = start; outputOrdinal < end; outputOrdinal++) {
            long sum = 0;
            Arrays.fill(packed, inputCoordinates, inputCoordinates + inputRank, 0L);
            for (long inputOrdinal = 0; inputOrdinal < inputCount; inputOrdinal++) {
                if (matches(packed, family, inputCoordinates, outputCoordinates, inputLayout,
                        mapping, inputRank, outputRank)) {
                    sum = add(sum, readBits(input, address(packed, inputLayout, inputCoordinates),
                            type), type);
                }
                advance(packed, inputCoordinates, inputLayout, inputRank);
            }
            writeBits(output, address(packed, outputLayout, outputCoordinates), type, sum);
            advance(packed, outputCoordinates, outputLayout, outputRank);
        }
    }

    private static boolean matches(long[] p, int family, int input, int output, int inputLayout,
            int mapping, int inputRank, int outputRank) {
        if (family == 0) {
            int axis = (int) p[mapping]; long step = p[mapping + 2];
            for (int current = 0; current < outputRank; current++) {
                long target = current == axis
                        ? p[input + current] * step + p[input + inputRank - 1]
                        : p[input + current];
                if (target != p[output + current]) return false;
            }
            return true;
        }
        long khCount = p[mapping], kwCount = p[mapping + 1];
        long strideHeight = p[mapping + 2], strideWidth = p[mapping + 3];
        long paddingHeight = p[mapping + 4], paddingWidth = p[mapping + 5];
        long dilationHeight = p[mapping + 6], dilationWidth = p[mapping + 7];
        long columnWidth = p[mapping + 9];
        long q = p[input + 1], column = p[input + 2];
        long kernelArea = khCount * kwCount;
        long channel = q / kernelArea;
        long kernel = q - channel * kernelArea;
        long kh = kernel / kwCount, kw = kernel - kh * kwCount;
        long oh = column / columnWidth, ow = column - oh * columnWidth;
        long height = oh * strideHeight - paddingHeight + kh * dilationHeight;
        long width = ow * strideWidth - paddingWidth + kw * dilationWidth;
        return p[input] == p[output] && channel == p[output + 1]
                && height == p[output + 2] && width == p[output + 3];
    }

    private static long add(long left, long right, DataType type) {
        return switch (type) {
            case INT32 -> (int) left + (int) right;
            case INT64 -> left + right;
            case FLOAT64 -> Double.doubleToRawLongBits(
                    Double.longBitsToDouble(left) + Double.longBitsToDouble(right));
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(
                    Float.intBitsToFloat((int) left) + Float.intBitsToFloat((int) right)));
            case BFLOAT16 -> Short.toUnsignedLong(floatToBfloat(
                    bfloatToFloat((short) left) + bfloatToFloat((short) right)));
            case BOOL -> throw new AssertionError("BOOL fold is unsupported");
        };
    }

    private static long address(long[] p, int layout, int coordinate) {
        long result = p[layout + 1]; int rank = (int) p[layout];
        for (int axis = 0; axis < rank; axis++) {
            result += p[coordinate + axis] * p[layout + 2 + rank + axis];
        }
        return result;
    }

    private static void advance(long[] p, int coordinate, int layout, int rank) {
        for (int axis = rank - 1; axis >= 0; axis--) {
            long next = p[coordinate + axis] + 1;
            p[coordinate + axis] = next;
            if (next < p[layout + 2 + axis]) return;
            p[coordinate + axis] = 0;
        }
    }

    private static long readBits(Object carrier, long address, DataType type) {
        return switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits(carrier instanceof double[] values
                    ? values[Math.toIntExact(address)]
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_DOUBLE, address * 8));
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(
                    carrier instanceof float[] values ? values[Math.toIntExact(address)]
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_FLOAT, address * 4)));
            case BFLOAT16 -> Short.toUnsignedLong(carrier instanceof short[] values
                    ? values[Math.toIntExact(address)]
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_SHORT, address * 2));
            case INT32 -> carrier instanceof int[] values ? values[Math.toIntExact(address)]
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_INT, address * 4);
            case INT64 -> carrier instanceof long[] values ? values[Math.toIntExact(address)]
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_LONG, address * 8);
            case BOOL -> throw new AssertionError("BOOL fold is unsupported");
        };
    }

    private static void writeBits(Object carrier, long address, DataType type, long bits) {
        switch (type) {
            case FLOAT64 -> {
                double value = Double.longBitsToDouble(bits);
                if (carrier instanceof double[] values) values[Math.toIntExact(address)] = value;
                else ((MemorySegment) carrier).set(ValueLayout.JAVA_DOUBLE, address * 8, value);
            }
            case FLOAT32 -> {
                float value = Float.intBitsToFloat((int) bits);
                if (carrier instanceof float[] values) values[Math.toIntExact(address)] = value;
                else ((MemorySegment) carrier).set(ValueLayout.JAVA_FLOAT, address * 4, value);
            }
            case BFLOAT16 -> {
                short value = (short) bits;
                if (carrier instanceof short[] values) values[Math.toIntExact(address)] = value;
                else ((MemorySegment) carrier).set(ValueLayout.JAVA_SHORT, address * 2, value);
            }
            case INT32 -> {
                int value = (int) bits;
                if (carrier instanceof int[] values) values[Math.toIntExact(address)] = value;
                else ((MemorySegment) carrier).set(ValueLayout.JAVA_INT, address * 4, value);
            }
            case INT64 -> {
                if (carrier instanceof long[] values) values[Math.toIntExact(address)] = bits;
                else ((MemorySegment) carrier).set(ValueLayout.JAVA_LONG, address * 8, bits);
            }
            case BOOL -> throw new AssertionError("BOOL fold is unsupported");
        }
    }

    private static float bfloatToFloat(short bits) {
        return Float.intBitsToFloat(Short.toUnsignedInt(bits) << 16);
    }

    private static short floatToBfloat(float value) {
        int bits = Float.floatToRawIntBits(value);
        if ((bits & 0x7f800000) == 0x7f800000 && (bits & 0x7fffff) != 0) {
            return (short) ((bits >>> 16) | 0x40);
        }
        int upper = bits >>> 16, lower = bits & 0xffff;
        if (lower > 0x8000 || lower == 0x8000 && (upper & 1) != 0) upper++;
        return (short) upper;
    }
}
