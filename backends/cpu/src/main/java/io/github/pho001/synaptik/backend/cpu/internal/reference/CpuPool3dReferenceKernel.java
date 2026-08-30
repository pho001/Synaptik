package io.github.pho001.synaptik.backend.cpu.internal.reference;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool3dIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool3dLowering;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.Objects;

/** Optimal clean-Java scalar oracle for the generated NCDHW Pool3d boundary. */
public final class CpuPool3dReferenceKernel {
    private CpuPool3dReferenceKernel() {}

    /**
     * Evaluates a half-open range of complete NCDHW output cells.
     *
     * @param geometry checked immutable Pool3d geometry
     * @param input exact typed array or native-order segment input carrier
     * @param output exact same-type mutable array or native-order segment output carrier
     * @param start inclusive logical output-cell ordinal
     * @param end exclusive logical output-cell ordinal
     * @throws NullPointerException if a required reference is null
     * @throws IllegalArgumentException if carrier type, capacity, or range disagrees
     * @throws ArithmeticException if checked address validation overflows
     */
    public static void evaluate(CpuPool3dLowering.Geometry geometry, Object input, Object output,
            long start, long end) {
        Objects.requireNonNull(geometry, "geometry");
        Objects.requireNonNull(input, "input"); Objects.requireNonNull(output, "output");
        if (start < 0 || end < start || end > geometry.outputCount())
            throw new IllegalArgumentException("Pool3d range is invalid");
        requireCarrier(geometry, geometry.input(), input);
        requireCarrier(geometry, geometry.output(), output);
        long[] ie = geometry.input().extents(), is = geometry.input().strides();
        long[] oe = geometry.output().extents(), os = geometry.output().strides();
        for (long cell = start; cell < end; cell++) {
            long r = cell, ow = r % oe[4]; r /= oe[4];
            long oh = r % oe[3]; r /= oe[3];
            long od = r % oe[2]; r /= oe[2];
            long c = r % oe[1], n = r / oe[1];
            long out = address(geometry.output().offset(), os, n, c, od, oh, ow);
            long id0 = Math.subtractExact(Math.multiplyExact(od, geometry.strideDepth()),
                    geometry.paddingDepth());
            long ih0 = Math.subtractExact(Math.multiplyExact(oh, geometry.strideHeight()),
                    geometry.paddingHeight());
            long iw0 = Math.subtractExact(Math.multiplyExact(ow, geometry.strideWidth()),
                    geometry.paddingWidth());
            if (geometry.kind() == CpuPool3dIr.Kind.MAX)
                max(geometry, input, output, ie, is, n, c, id0, ih0, iw0, out);
            else average(geometry, input, output, ie, is, n, c, id0, ih0, iw0, out);
        }
    }

    private static void max(CpuPool3dLowering.Geometry g, Object in, Object out, long[] e,
            long[] s, long n, long c, long id0, long ih0, long iw0, long output) {
        boolean found = false;
        long winner = 0;
        double best = Double.NEGATIVE_INFINITY;
        outer: for (long kd = 0; kd < g.kernelDepth(); kd++) {
            long id = Math.addExact(id0, Math.multiplyExact(kd, g.dilationDepth()));
            for (long kh = 0; kh < g.kernelHeight(); kh++) {
                long ih = Math.addExact(ih0, Math.multiplyExact(kh, g.dilationHeight()));
                for (long kw = 0; kw < g.kernelWidth(); kw++) {
                    long iw = Math.addExact(iw0, Math.multiplyExact(kw, g.dilationWidth()));
                    if (id < 0 || ih < 0 || iw < 0 || id >= e[2] || ih >= e[3]
                            || iw >= e[4]) continue;
                    long represented = bits(g, in,
                            address(g.input().offset(), s, n, c, id, ih, iw));
                    double value = decode(g, represented);
                    if (Double.isNaN(value)) {
                        winner = represented; found = true; break outer;
                    }
                    if (!found || value > best || value == 0.0 && best == 0.0
                            && Double.doubleToRawLongBits(value) == 0L
                            && Double.doubleToRawLongBits(best) != 0L) {
                        found = true; winner = represented; best = value;
                    }
                }
            }
        }
        store(g, out, output, found ? winner : negativeInfinity(g));
    }

    private static void average(CpuPool3dLowering.Geometry g, Object in, Object out, long[] e,
            long[] s, long n, long c, long id0, long ih0, long iw0, long output) {
        boolean allNegativeZero = true;
        if (g.dataType() == DataType.FLOAT64) {
            double sum = 0.0;
            for (long kd = 0; kd < g.kernelDepth(); kd++) {
                long id = Math.addExact(id0, Math.multiplyExact(kd, g.dilationDepth()));
                for (long kh = 0; kh < g.kernelHeight(); kh++) {
                    long ih = Math.addExact(ih0, Math.multiplyExact(kh, g.dilationHeight()));
                    for (long kw = 0; kw < g.kernelWidth(); kw++) {
                        long iw = Math.addExact(iw0, Math.multiplyExact(kw, g.dilationWidth()));
                        if (id < 0 || ih < 0 || iw < 0 || id >= e[2] || ih >= e[3]
                                || iw >= e[4]) { allNegativeZero = false; continue; }
                        double value = Double.longBitsToDouble(bits(g, in,
                                address(g.input().offset(), s, n, c, id, ih, iw)));
                        allNegativeZero &= Double.doubleToRawLongBits(value) == Long.MIN_VALUE;
                        sum += value;
                    }
                }
            }
            double result = sum / (double) g.divisor();
            if (result == 0.0) result = allNegativeZero ? -0.0 : +0.0;
            store(g, out, output, Double.doubleToRawLongBits(result));
        } else {
            float sum = 0.0f;
            for (long kd = 0; kd < g.kernelDepth(); kd++) {
                long id = Math.addExact(id0, Math.multiplyExact(kd, g.dilationDepth()));
                for (long kh = 0; kh < g.kernelHeight(); kh++) {
                    long ih = Math.addExact(ih0, Math.multiplyExact(kh, g.dilationHeight()));
                    for (long kw = 0; kw < g.kernelWidth(); kw++) {
                        long iw = Math.addExact(iw0, Math.multiplyExact(kw, g.dilationWidth()));
                        if (id < 0 || ih < 0 || iw < 0 || id >= e[2] || ih >= e[3]
                                || iw >= e[4]) { allNegativeZero = false; continue; }
                        long represented = bits(g, in,
                                address(g.input().offset(), s, n, c, id, ih, iw));
                        float value = (float) decode(g, represented);
                        allNegativeZero &= g.dataType() == DataType.FLOAT32
                                ? (int) represented == Integer.MIN_VALUE
                                : (represented & 0xffffL) == 0x8000L;
                        sum += value;
                    }
                }
            }
            float result = sum / (float) g.divisor();
            if (result == 0.0f) result = allNegativeZero ? -0.0f : +0.0f;
            long represented = g.dataType() == DataType.FLOAT32
                    ? Float.floatToRawIntBits(result) & 0xffffffffL
                    : BFloat16Bits.fromFloat(result) & 0xffffL;
            store(g, out, output, represented);
        }
    }

    private static long address(long base, long[] s, long n, long c, long d, long h, long w) {
        long address = base;
        long[] coordinates = {n, c, d, h, w};
        for (int axis = 0; axis < 5; axis++)
            address = Math.addExact(address, Math.multiplyExact(coordinates[axis], s[axis]));
        return address;
    }

    private static long negativeInfinity(CpuPool3dLowering.Geometry g) {
        return switch (g.dataType()) {
            case FLOAT64 -> Double.doubleToRawLongBits(Double.NEGATIVE_INFINITY);
            case FLOAT32 -> Float.floatToRawIntBits(Float.NEGATIVE_INFINITY) & 0xffffffffL;
            case BFLOAT16 -> BFloat16Bits.fromFloat(Float.NEGATIVE_INFINITY) & 0xffffL;
            default -> throw new AssertionError();
        };
    }

    private static double decode(CpuPool3dLowering.Geometry g, long bits) {
        return switch (g.dataType()) {
            case FLOAT64 -> Double.longBitsToDouble(bits);
            case FLOAT32 -> Float.intBitsToFloat((int) bits);
            case BFLOAT16 -> BFloat16Bits.toFloat((short) bits);
            default -> throw new AssertionError();
        };
    }

    private static long bits(CpuPool3dLowering.Geometry g, Object carrier, long address) {
        long bytes = Math.multiplyExact(address, g.dataType().byteWidth());
        return switch (g.dataType()) {
            case FLOAT64 -> carrier instanceof double[] v
                    ? Double.doubleToRawLongBits(v[Math.toIntExact(address)])
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_LONG_UNALIGNED, bytes);
            case FLOAT32 -> carrier instanceof float[] v
                    ? Float.floatToRawIntBits(v[Math.toIntExact(address)]) & 0xffffffffL
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_INT_UNALIGNED, bytes)
                            & 0xffffffffL;
            case BFLOAT16 -> carrier instanceof short[] v ? v[Math.toIntExact(address)] & 0xffffL
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_SHORT_UNALIGNED, bytes)
                            & 0xffffL;
            default -> throw new AssertionError();
        };
    }

    private static void store(CpuPool3dLowering.Geometry g, Object carrier, long address,
            long bits) {
        long bytes = Math.multiplyExact(address, g.dataType().byteWidth());
        switch (g.dataType()) {
            case FLOAT64 -> { if (carrier instanceof double[] v)
                    v[Math.toIntExact(address)] = Double.longBitsToDouble(bits);
                else ((MemorySegment) carrier).set(ValueLayout.JAVA_LONG_UNALIGNED, bytes, bits); }
            case FLOAT32 -> { if (carrier instanceof float[] v)
                    v[Math.toIntExact(address)] = Float.intBitsToFloat((int) bits);
                else ((MemorySegment) carrier).set(ValueLayout.JAVA_INT_UNALIGNED, bytes, (int) bits); }
            case BFLOAT16 -> { if (carrier instanceof short[] v) v[Math.toIntExact(address)] = (short) bits;
                else ((MemorySegment) carrier).set(ValueLayout.JAVA_SHORT_UNALIGNED, bytes, (short) bits); }
            default -> throw new AssertionError();
        }
    }

    private static void requireCarrier(CpuPool3dLowering.Geometry g,
            CpuPool3dLowering.Layout layout, Object carrier) {
        boolean valid = carrier instanceof MemorySegment || switch (g.dataType()) {
            case FLOAT64 -> carrier instanceof double[];
            case FLOAT32 -> carrier instanceof float[];
            case BFLOAT16 -> carrier instanceof short[];
            default -> false;
        };
        if (!valid) throw new IllegalArgumentException("Pool3d carrier type disagrees");
        long required = 0;
        long[] e = layout.extents(), s = layout.strides();
        if (Arrays.stream(e).noneMatch(x -> x == 0)) {
            required = layout.offset();
            for (int axis = 0; axis < 5; axis++)
                required = Math.addExact(required, Math.multiplyExact(e[axis] - 1, s[axis]));
            required = Math.addExact(required, 1);
        }
        long actual = switch (g.dataType()) {
            case FLOAT64 -> carrier instanceof double[] v ? v.length
                    : ((MemorySegment) carrier).byteSize() / Double.BYTES;
            case FLOAT32 -> carrier instanceof float[] v ? v.length
                    : ((MemorySegment) carrier).byteSize() / Float.BYTES;
            case BFLOAT16 -> carrier instanceof short[] v ? v.length
                    : ((MemorySegment) carrier).byteSize() / Short.BYTES;
            default -> throw new AssertionError();
        };
        if (actual < required) throw new IllegalArgumentException("Pool3d carrier capacity disagrees");
    }
}
