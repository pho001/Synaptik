package io.github.pho001.synaptik.backend.cpu.internal.reference;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool2dIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool2dLowering;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Optimal clean-Java scalar oracle for the Shape-polymorphic generated Pool2d boundary.
 *
 * <p>The oracle follows the generated body's complete-output-cell loop and numerical dataflow for
 * max and fixed-divisor average pooling over exact heap-array or {@link MemorySegment} carriers.
 * It exists for semantic and generated-versus-direct evidence and is not a Runtime interpreter or
 * fallback execution route.
 */
public final class CpuPool2dReferenceKernel {
    private CpuPool2dReferenceKernel() {}

    /**
     * Evaluates a half-open range of complete NCHW output cells.
     *
     * @param geometry checked immutable pooling geometry
     * @param input exact typed array or native-order segment input carrier
     * @param output exact same-type mutable array or native-order segment output carrier
     * @param start inclusive logical NCHW output-cell ordinal
     * @param end exclusive logical NCHW output-cell ordinal
     * @throws NullPointerException if a required reference is null
     * @throws IllegalArgumentException if carrier type/capacity or range disagrees with geometry
     * @throws ArithmeticException if exact address-span validation overflows
     */
    public static void evaluate(
            CpuPool2dLowering.Geometry geometry, Object input, Object output, long start, long end) {
        Objects.requireNonNull(geometry, "geometry");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        if (start < 0 || end < start || end > geometry.outputCount())
            throw new IllegalArgumentException("Pool2d range is invalid");
        requireCarrier(geometry, geometry.input(), input);
        requireCarrier(geometry, geometry.output(), output);
        long[] ie = geometry.input().extents(),
                is = geometry.input().strides(),
                oe = geometry.output().extents(),
                os = geometry.output().strides();
        for (long cell = start; cell < end; cell++) {
            long r = cell, ow = r % oe[3];
            r /= oe[3];
            long oh = r % oe[2];
            r /= oe[2];
            long c = r % oe[1], n = r / oe[1];
            long out = address(geometry.output().offset(), os, n, c, oh, ow);
            long ih0 =
                    Math.subtractExact(
                            Math.multiplyExact(oh, geometry.strideHeight()), geometry.paddingHeight());
            long iw0 =
                    Math.subtractExact(
                            Math.multiplyExact(ow, geometry.strideWidth()), geometry.paddingWidth());
            if (geometry.kind() == CpuPool2dIr.Kind.MAX)
                max(geometry, input, output, ie, is, n, c, ih0, iw0, out);
            else average(geometry, input, output, ie, is, n, c, ih0, iw0, out);
        }
    }

    private static void max(
            CpuPool2dLowering.Geometry g,
            Object in,
            Object out,
            long[] e,
            long[] s,
            long n,
            long c,
            long ih0,
            long iw0,
            long address) {
        boolean found = false;
        long winner = 0;
        double best = Double.NEGATIVE_INFINITY;
        outer:
        for (long kh = 0; kh < g.kernelHeight(); kh++) {
            long ih = Math.addExact(ih0, Math.multiplyExact(kh, g.dilationHeight()));
            for (long kw = 0; kw < g.kernelWidth(); kw++) {
                long iw = Math.addExact(iw0, Math.multiplyExact(kw, g.dilationWidth()));
                if (ih < 0 || iw < 0 || ih >= e[2] || iw >= e[3]) continue;
                long a = address(g.input().offset(), s, n, c, ih, iw);
                long bits = bits(g, in, a);
                double value = decode(g, bits);
                if (Double.isNaN(value)) {
                    winner = bits;
                    found = true;
                    break outer;
                }
                if (!found
                        || value > best
                        || value == 0.0
                                && best == 0.0
                                && Double.doubleToRawLongBits(value) == 0L
                                && Double.doubleToRawLongBits(best) != 0L) {
                    found = true;
                    winner = bits;
                    best = value;
                }
            }
        }
        if (!found) winner = negativeInfinity(g);
        store(g, out, address, winner);
    }

    private static void average(
            CpuPool2dLowering.Geometry g,
            Object in,
            Object out,
            long[] e,
            long[] s,
            long n,
            long c,
            long ih0,
            long iw0,
            long address) {
        boolean allNegativeZero = true;
        if (g.dataType() == io.github.pho001.synaptik.model.datatype.DataType.FLOAT64) {
            double sum = 0.0;
            for (long kh = 0; kh < g.kernelHeight(); kh++) {
                long ih = Math.addExact(ih0, Math.multiplyExact(kh, g.dilationHeight()));
                for (long kw = 0; kw < g.kernelWidth(); kw++) {
                    long iw = Math.addExact(iw0, Math.multiplyExact(kw, g.dilationWidth()));
                    if (ih < 0 || iw < 0 || ih >= e[2] || iw >= e[3]) {
                        allNegativeZero = false;
                        continue;
                    }
                    double v =
                            Double.longBitsToDouble(bits(g, in, address(g.input().offset(), s, n, c, ih, iw)));
                    allNegativeZero &= Double.doubleToRawLongBits(v) == Long.MIN_VALUE;
                    sum += v;
                }
            }
            double result = sum / (double) g.divisor();
            if (result == 0.0) result = allNegativeZero ? -0.0 : +0.0;
            store(g, out, address, Double.doubleToRawLongBits(result));
        } else {
            float sum = 0.0f;
            for (long kh = 0; kh < g.kernelHeight(); kh++) {
                long ih = Math.addExact(ih0, Math.multiplyExact(kh, g.dilationHeight()));
                for (long kw = 0; kw < g.kernelWidth(); kw++) {
                    long iw = Math.addExact(iw0, Math.multiplyExact(kw, g.dilationWidth()));
                    if (ih < 0 || iw < 0 || ih >= e[2] || iw >= e[3]) {
                        allNegativeZero = false;
                        continue;
                    }
                    float v = (float) decode(g, bits(g, in, address(g.input().offset(), s, n, c, ih, iw)));
                    allNegativeZero &= Float.floatToRawIntBits(v) == Integer.MIN_VALUE;
                    sum += v;
                }
            }
            float result = sum / (float) g.divisor();
            if (result == 0.0f) result = allNegativeZero ? -0.0f : +0.0f;
            long represented =
                    g.dataType() == io.github.pho001.synaptik.model.datatype.DataType.FLOAT32
                            ? Float.floatToRawIntBits(result) & 0xffffffffL
                            : BFloat16Bits.fromFloat(result) & 0xffffL;
            store(g, out, address, represented);
        }
    }

    private static long address(long base, long[] s, long n, long c, long h, long w) {
        return Math.addExact(
                base,
                Math.addExact(
                        Math.multiplyExact(n, s[0]),
                        Math.addExact(
                                Math.multiplyExact(c, s[1]),
                                Math.addExact(Math.multiplyExact(h, s[2]), Math.multiplyExact(w, s[3])))));
    }

    private static long negativeInfinity(CpuPool2dLowering.Geometry g) {
        return switch (g.dataType()) {
            case FLOAT64 -> Double.doubleToRawLongBits(Double.NEGATIVE_INFINITY);
            case FLOAT32 -> Float.floatToRawIntBits(Float.NEGATIVE_INFINITY) & 0xffffffffL;
            case BFLOAT16 -> BFloat16Bits.fromFloat(Float.NEGATIVE_INFINITY) & 0xffffL;
            default -> throw new AssertionError();
        };
    }

    private static double decode(CpuPool2dLowering.Geometry g, long bits) {
        return switch (g.dataType()) {
            case FLOAT64 -> Double.longBitsToDouble(bits);
            case FLOAT32 -> Float.intBitsToFloat((int) bits);
            case BFLOAT16 -> BFloat16Bits.toFloat((short) bits);
            default -> throw new AssertionError();
        };
    }

    private static long bits(CpuPool2dLowering.Geometry g, Object carrier, long a) {
        long byteOffset = Math.multiplyExact(a, g.dataType().byteWidth());
        return switch (g.dataType()) {
            case FLOAT64 ->
                    carrier instanceof double[] v
                            ? Double.doubleToRawLongBits(v[Math.toIntExact(a)])
                            : ((MemorySegment) carrier).get(ValueLayout.JAVA_LONG_UNALIGNED, byteOffset);
            case FLOAT32 ->
                    carrier instanceof float[] v
                            ? Float.floatToRawIntBits(v[Math.toIntExact(a)]) & 0xffffffffL
                            : ((MemorySegment) carrier).get(ValueLayout.JAVA_INT_UNALIGNED, byteOffset)
                                    & 0xffffffffL;
            case BFLOAT16 ->
                    carrier instanceof short[] v
                            ? v[Math.toIntExact(a)] & 0xffffL
                            : ((MemorySegment) carrier).get(ValueLayout.JAVA_SHORT_UNALIGNED, byteOffset)
                                    & 0xffffL;
            default -> throw new AssertionError();
        };
    }

    private static void store(CpuPool2dLowering.Geometry g, Object carrier, long a, long bits) {
        long byteOffset = Math.multiplyExact(a, g.dataType().byteWidth());
        switch (g.dataType()) {
            case FLOAT64 -> {
                if (carrier instanceof double[] v) v[Math.toIntExact(a)] = Double.longBitsToDouble(bits);
                else ((MemorySegment) carrier).set(ValueLayout.JAVA_LONG_UNALIGNED, byteOffset, bits);
            }
            case FLOAT32 -> {
                if (carrier instanceof float[] v) v[Math.toIntExact(a)] = Float.intBitsToFloat((int) bits);
                else ((MemorySegment) carrier).set(ValueLayout.JAVA_INT_UNALIGNED, byteOffset, (int) bits);
            }
            case BFLOAT16 -> {
                if (carrier instanceof short[] v) v[Math.toIntExact(a)] = (short) bits;
                else
                    ((MemorySegment) carrier).set(ValueLayout.JAVA_SHORT_UNALIGNED, byteOffset, (short) bits);
            }
            default -> throw new AssertionError();
        }
    }

    private static void requireCarrier(
            CpuPool2dLowering.Geometry g, CpuPool2dLowering.Layout layout, Object carrier) {
        boolean ok =
                carrier instanceof MemorySegment
                        || switch (g.dataType()) {
                            case FLOAT64 -> carrier instanceof double[];
                            case FLOAT32 -> carrier instanceof float[];
                            case BFLOAT16 -> carrier instanceof short[];
                            default -> false;
                        };
        if (!ok) throw new IllegalArgumentException("Pool2d carrier type disagrees");
        long requiredElements = 0;
        long[] extents = layout.extents();
        if (java.util.Arrays.stream(extents).noneMatch(extent -> extent == 0)) {
            requiredElements = layout.offset();
            long[] strides = layout.strides();
            for (int axis = 0; axis < extents.length; axis++) {
                requiredElements =
                        Math.addExact(requiredElements, Math.multiplyExact(extents[axis] - 1, strides[axis]));
            }
            requiredElements = Math.addExact(requiredElements, 1);
        }
        long actualElements =
                switch (g.dataType()) {
                    case FLOAT64 ->
                            carrier instanceof double[] values
                                    ? values.length
                                    : ((MemorySegment) carrier).byteSize() / Long.BYTES;
                    case FLOAT32 ->
                            carrier instanceof float[] values
                                    ? values.length
                                    : ((MemorySegment) carrier).byteSize() / Float.BYTES;
                    case BFLOAT16 ->
                            carrier instanceof short[] values
                                    ? values.length
                                    : ((MemorySegment) carrier).byteSize() / Short.BYTES;
                    default -> throw new AssertionError();
                };
        if (actualElements < requiredElements) {
            throw new IllegalArgumentException("Pool2d carrier capacity disagrees");
        }
    }
}
