package io.github.pho001.synaptik.backend.cpu.internal.reference;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Objects;
import java.math.BigInteger;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLowering;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLowering;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLowering;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLowering;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuOrderingLowering;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLowering;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScanLowering;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuArgExtremaIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuArgExtremaLowering;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaskedReductionLowering;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.model.datatype.DataType;

/**
 * Scalar conformance realization for the bounded typed CPU portable semantics.
 * It evaluates already-lowered primitive arithmetic, exact extrema and clamp, direct Tensor
 * power, canonical-BOOL logic, the closed FLOAT32/FLOAT64 unary matrix, and the selected
 * scalar-power plan. Direct power uses {@link StrictMath#pow(double, double)} without
 * reclassifying an exponent. Unary evaluation preserves the specified exceptional-value
 * classifications, widens represented FLOAT32 values where required, and narrows once. It also
 * evaluates already-lowered affine, movement, indexing, functional slice-update,
 * functional-scatter, overlap-fold, stable ordering, explicit-state random, cumulative-scan, and
 * ordinary and right-aligned SUM-to-Shape aggregate mappings for differential tests. Numerical
 * aggregate evaluation uses
 * independent {@link BigInteger} integer/rational conversion rather than generated emitter
 * rounding logic. The
 * ordering oracle uses an independent primitive-index insertion algorithm while preserving the
 * same Model order and represented output bits. This is an unsupported cold-test/reference
 * contract and is never a Runtime IR interpreter.
 */
public final class CpuScalarReferenceKernel {
    private static final long RANDOM_KEY_BIAS = 0x9e3779b97f4a7c15L;
    private static final long RANDOM_M1 = 0xbf58476d1ce4e5b9L;
    private static final long RANDOM_M2 = 0x94d049bb133111ebL;
    private static final ValueLayout.OfDouble DOUBLE = ValueLayout.JAVA_DOUBLE_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    // Cephes erf/erfc rational coefficients, documented at
    // https://netlib.org/cephes/doubldoc.html and published in netlib cephes ndtr.c.
    private static final double[] ERF_T = {9.60497373987051638749E0,
            9.00260197203842689217E1, 2.23200534594684319226E3,
            7.00332514112805075473E3, 5.55923013010394962768E4};
    private static final double[] ERF_U = {3.35617141647503099647E1,
            5.21357949780152679795E2, 4.59432382970980127987E3,
            2.26290000613890934246E4, 4.92673942608635921086E4};
    private static final double[] ERFC_P = {2.46196981473530512524E-10,
            5.64189564831068821977E-1, 7.46321056442269912687E0,
            4.86371970985681366614E1, 1.96520832956077098242E2,
            5.26445194995477358631E2, 9.34528527171957607540E2,
            1.02755188689515710272E3, 5.57535335369399327526E2};
    private static final double[] ERFC_Q = {1.32281951154744992508E1,
            8.67072140885989742329E1, 3.54937778887819891062E2,
            9.75708501743205489753E2, 1.82390916687909736289E3,
            2.24633760818710981792E3, 1.65666309194161350182E3,
            5.57535340817727675546E2};
    private static final double[] ERFC_R = {5.64189583547755073984E-1,
            1.27536670759978104416E0, 5.01905042251180477414E0,
            6.16021097993053585195E0, 7.40974269950448939160E0,
            2.97886665372100240670E0};
    private static final double[] ERFC_S = {2.26052863220117276590E0,
            9.39603524938001434673E0, 1.20489539808096656605E1,
            1.70814450747565897222E1, 9.60896809063285878198E0,
            3.36907645100081516050E0};
    private CpuScalarReferenceKernel() { }

    /**
     * Independently evaluates one complete ordinary numerical, extrema, Boolean, or bound
     * SUM-to-Shape reduction. A SUM-to-Shape occurrence with no selected axis copies the exact
     * represented input value without numerical classification.
     *
     * <p>This oracle derives logical coordinates directly from Shapes and selected membership;
     * it does not call production aggregate execution, packing, lowering, or coordinate helpers.</p>
     * @param ir non-null aggregate semantics
     * @param geometry non-null matching cold layouts and domain counts
     * @param arguments non-null exact input/output carrier list; read but not retained
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if IR, geometry, or carrier cardinality disagrees
     */
    public static void execute(CpuAggregateIr ir, CpuAggregateLowering.Geometry geometry,
            List<CpuBufferArgument> arguments) {
        Objects.requireNonNull(ir, "ir"); Objects.requireNonNull(geometry, "geometry");
        Objects.requireNonNull(arguments, "arguments");
        if (arguments.size() != 2 || ir.kind() != geometry.kind()
                || ir.dataType() != geometry.dataType())
            throw new IllegalArgumentException("aggregate reference facts disagree");
        long[] inputExtents = geometry.input().extents();
        long[] outputExtents = geometry.output().extents();
        boolean[] selected = new boolean[inputExtents.length];
        for (int axis : geometry.selectedAxes()) selected[axis] = true;
        long[] inputCoordinates = new long[inputExtents.length];
        long[] outputCoordinates = new long[outputExtents.length];
        for (long cell = 0; cell < geometry.outputCount(); cell++) {
            decodeReference(cell, outputExtents, outputCoordinates);
            int outputAxis = 0;
            int leading = inputCoordinates.length - outputCoordinates.length;
            for (int axis = 0; axis < inputCoordinates.length; axis++) {
                inputCoordinates[axis] = selected[axis] ? 0
                        : outputCoordinates[geometry.form() == CpuAggregateIr.Form.SUM_TO_SHAPE
                            ? axis - leading
                            : geometry.keepDimensions() ? axis : outputAxis++];
            }
            if (geometry.form() == CpuAggregateIr.Form.SUM_TO_SHAPE
                    && geometry.selectedAxes().length == 0) {
                Object represented = load(arguments.getFirst(), ir.dataType(),
                        aggregateAddress(geometry.input(), inputCoordinates));
                store(arguments.getLast(), ir.dataType(),
                        aggregateAddress(geometry.output(), outputCoordinates), represented);
                continue;
            }
            Object accumulator = aggregateIdentity(ir.kind(), ir.dataType());
            if ((ir.kind() == CpuAggregateIr.Kind.SUM || ir.kind() == CpuAggregateIr.Kind.MEAN
                    || ir.kind() == CpuAggregateIr.Kind.PROD)
                    && ir.dataType() != DataType.INT32 && ir.dataType() != DataType.INT64) {
                accumulator = numericalFloatingReference(ir.kind(), ir.dataType(), arguments.getFirst(),
                        geometry, inputCoordinates, inputExtents, selected);
                store(arguments.getLast(), ir.dataType(),
                        aggregateAddress(geometry.output(), outputCoordinates), accumulator);
                continue;
            }
            for (long domain = 0; domain < geometry.domainCount(); domain++) {
                long remaining = domain;
                for (int axis = inputCoordinates.length - 1; axis >= 0; axis--) if (selected[axis]) {
                    inputCoordinates[axis] = remaining % inputExtents[axis];
                    remaining /= inputExtents[axis];
                }
                Object value = load(arguments.getFirst(), ir.dataType(),
                        aggregateAddress(geometry.input(), inputCoordinates));
                accumulator = aggregateApply(ir.kind(), ir.dataType(), accumulator, value);
            }
            store(arguments.getLast(), ir.dataType(),
                    aggregateAddress(geometry.output(), outputCoordinates), accumulator);
        }
    }

    private static void decodeReference(long logical, long[] extents, long[] coordinates) {
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            coordinates[axis] = logical % extents[axis]; logical /= extents[axis];
        }
    }
    private static long aggregateAddress(CpuAggregateLowering.Layout layout, long[] coordinates) {
        long result = layout.offset(); long[] strides = layout.strides();
        for (int axis = 0; axis < coordinates.length; axis++) result += coordinates[axis] * strides[axis];
        return result;
    }
    private static Object aggregateIdentity(CpuAggregateIr.Kind kind, DataType type) {
        boolean minimum = kind == CpuAggregateIr.Kind.MIN;
        return switch (type) {
            case FLOAT64 -> minimum ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
            case FLOAT32 -> minimum ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            case BFLOAT16 -> (short) (minimum ? 0x7f80 : 0xff80);
            case INT32 -> kind == CpuAggregateIr.Kind.SUM ? 0 : kind == CpuAggregateIr.Kind.PROD ? 1
                    : minimum ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            case INT64 -> kind == CpuAggregateIr.Kind.SUM ? 0L : kind == CpuAggregateIr.Kind.PROD ? 1L
                    : minimum ? Long.MAX_VALUE : Long.MIN_VALUE;
            case BOOL -> (byte) (kind == CpuAggregateIr.Kind.ALL ? 1 : 0);
        };
    }
    private static Object aggregateApply(CpuAggregateIr.Kind kind, DataType type,
            Object left, Object right) {
        if (type == DataType.BOOL) return (byte) (kind == CpuAggregateIr.Kind.ALL
                ? ((byte) left != 0 && (byte) right != 0 ? 1 : 0)
                : ((byte) left != 0 || (byte) right != 0 ? 1 : 0));
        if (type == DataType.INT32) return kind == CpuAggregateIr.Kind.SUM
                ? (int) left + (int) right : kind == CpuAggregateIr.Kind.PROD
                    ? (int) left * (int) right : kind == CpuAggregateIr.Kind.MIN
                        ? Math.min((int) left, (int) right) : Math.max((int) left, (int) right);
        if (type == DataType.INT64) return kind == CpuAggregateIr.Kind.SUM
                ? (long) left + (long) right : kind == CpuAggregateIr.Kind.PROD
                    ? (long) left * (long) right : kind == CpuAggregateIr.Kind.MIN
                        ? Math.min((long) left, (long) right) : Math.max((long) left, (long) right);
        double l = type == DataType.FLOAT64 ? (double) left
                : type == DataType.FLOAT32 ? (float) left : bfloat((short) left);
        double r = type == DataType.FLOAT64 ? (double) right
                : type == DataType.FLOAT32 ? (float) right : bfloat((short) right);
        if (Double.isNaN(l)) return left;
        if (Double.isNaN(r)) return right;
        if (l == 0.0 && r == 0.0) {
            boolean ln = rawNegative(type, left), rn = rawNegative(type, right);
            if (kind == CpuAggregateIr.Kind.MIN) return ln ? left : rn ? right : left;
            return !ln ? left : !rn ? right : left;
        }
        return kind == CpuAggregateIr.Kind.MIN ? (r < l ? right : left) : (r > l ? right : left);
    }

    private static Object numericalFloatingReference(CpuAggregateIr.Kind kind, DataType type,
            CpuBufferArgument input, CpuAggregateLowering.Geometry geometry,
            long[] coordinates, long[] extents, boolean[] selected) {
        int fractionBits = type == DataType.FLOAT64 ? 52 : type == DataType.FLOAT32 ? 23 : 7;
        int bias = type == DataType.FLOAT64 ? 1023 : 127;
        int unitExponent = type == DataType.FLOAT64 ? -1074 : type == DataType.FLOAT32 ? -149 : -133;
        long exponentMask = type == DataType.FLOAT64 ? 0x7ffL : 0xffL;
        long signMask = type == DataType.FLOAT64 ? Long.MIN_VALUE
                : type == DataType.FLOAT32 ? 1L << 31 : 1L << 15;
        long fractionMask = (1L << fractionBits) - 1;
        BigInteger exact = kind == CpuAggregateIr.Kind.PROD ? BigInteger.ONE : BigInteger.ZERO;
        long productExponent = 0; boolean negative = false;
        boolean nan = false, positiveInfinity = false, negativeInfinity = false, zero = false;
        boolean positiveZero = false, negativeZero = false, nonzero = false;
        for (long domain = 0; domain < geometry.domainCount(); domain++) {
            long remaining = domain;
            for (int axis = coordinates.length - 1; axis >= 0; axis--) if (selected[axis]) {
                coordinates[axis] = remaining % extents[axis]; remaining /= extents[axis];
            }
            Object represented = load(input, type, aggregateAddress(geometry.input(), coordinates));
            long bits = type == DataType.FLOAT64 ? Double.doubleToRawLongBits((double) represented)
                    : type == DataType.FLOAT32 ? Integer.toUnsignedLong(
                            Float.floatToRawIntBits((float) represented))
                    : Short.toUnsignedLong((short) represented);
            long fraction = bits & fractionMask;
            long exponentField = bits >>> fractionBits & exponentMask;
            boolean sign = (bits & signMask) != 0;
            if (kind == CpuAggregateIr.Kind.PROD) negative ^= sign;
            if (exponentField == exponentMask) {
                if (fraction != 0) nan = true;
                else if (kind == CpuAggregateIr.Kind.PROD) positiveInfinity = true;
                else if (sign) negativeInfinity = true; else positiveInfinity = true;
                continue;
            }
            if (exponentField == 0 && fraction == 0) {
                zero = true; if (sign) negativeZero = true; else positiveZero = true; continue;
            }
            nonzero = true;
            long significand = exponentField == 0 ? fraction : (1L << fractionBits) | fraction;
            int exponent = exponentField == 0 ? 1 - bias - fractionBits
                    : Math.toIntExact(exponentField) - bias - fractionBits;
            if (kind == CpuAggregateIr.Kind.PROD) {
                exact = exact.multiply(BigInteger.valueOf(significand));
                productExponent = Math.addExact(productExponent, exponent);
            } else {
                BigInteger coefficient = BigInteger.valueOf(significand)
                        .shiftLeft(exponent - unitExponent);
                exact = exact.add(sign ? coefficient.negate() : coefficient);
            }
        }
        long canonical = type == DataType.FLOAT64 ? 0x7ff8000000000000L
                : type == DataType.FLOAT32 ? 0x7fc00000L : 0x7fc0L;
        long infinity = exponentMask << fractionBits;
        long result;
        if (kind == CpuAggregateIr.Kind.PROD) {
            if (nan || zero && positiveInfinity) result = canonical;
            else if (positiveInfinity) result = (negative ? signMask : 0) | infinity;
            else if (zero) result = negative ? signMask : 0;
            else result = roundRational(exact, productExponent, BigInteger.ONE, negative,
                    fractionBits, bias, signMask);
        } else if (nan || positiveInfinity && negativeInfinity
                || kind == CpuAggregateIr.Kind.MEAN && geometry.domainCount() == 0) result = canonical;
        else if (positiveInfinity) result = infinity;
        else if (negativeInfinity) result = signMask | infinity;
        else if (exact.signum() == 0) result = negativeZero && !positiveZero && !nonzero ? signMask : 0;
        else result = roundRational(exact.abs(), unitExponent,
                kind == CpuAggregateIr.Kind.MEAN ? BigInteger.valueOf(geometry.domainCount())
                        : BigInteger.ONE, exact.signum() < 0, fractionBits, bias, signMask);
        return switch (type) {
            case FLOAT64 -> Double.longBitsToDouble(result);
            case FLOAT32 -> Float.intBitsToFloat((int) result);
            case BFLOAT16 -> (short) result;
            default -> throw new AssertionError("non-floating numerical aggregate");
        };
    }

    private static long roundRational(BigInteger numerator, long binaryExponent,
            BigInteger divisor, boolean negative, int fractionBits, int bias, long signMask) {
        int precision = fractionBits + 1, minimumNormal = 1 - bias, maximumExponent = bias;
        long exponent = (long) numerator.bitLength() - 1 + binaryExponent
                - (divisor.bitLength() - 1L);
        while (compareToPowerOfTwo(numerator, binaryExponent, divisor, exponent) < 0) exponent--;
        while (compareToPowerOfTwo(numerator, binaryExponent, divisor, exponent + 1) >= 0) exponent++;
        long sign = negative ? signMask : 0;
        if (exponent > maximumExponent) return sign | ((long) (2 * bias + 1) << fractionBits);
        long target = exponent >= minimumNormal ? exponent - (precision - 1L)
                : minimumNormal - (precision - 1L);
        BigInteger scaledNumerator = numerator, scaledDivisor = divisor;
        long scale = binaryExponent - target;
        if (scale >= 0) scaledNumerator = scaledNumerator.shiftLeft(Math.toIntExact(scale));
        else scaledDivisor = scaledDivisor.shiftLeft(Math.toIntExact(-scale));
        BigInteger[] qr = scaledNumerator.divideAndRemainder(scaledDivisor);
        int comparison = qr[1].shiftLeft(1).compareTo(scaledDivisor);
        BigInteger rounded = qr[0];
        if (comparison > 0 || comparison == 0 && rounded.testBit(0)) rounded = rounded.add(BigInteger.ONE);
        if (exponent >= minimumNormal) {
            if (rounded.bitLength() > precision) { rounded = rounded.shiftRight(1); exponent++; }
            if (exponent > maximumExponent) return sign | ((long) (2 * bias + 1) << fractionBits);
            return sign | (exponent + bias << fractionBits)
                    | rounded.longValue() & ((1L << fractionBits) - 1);
        }
        long value = rounded.longValue();
        return sign | value;
    }

    private static int compareToPowerOfTwo(BigInteger numerator, long binaryExponent,
            BigInteger divisor, long exponent) {
        long shift = binaryExponent - exponent;
        return shift >= 0 ? numerator.shiftLeft(Math.toIntExact(shift)).compareTo(divisor)
                : numerator.compareTo(divisor.shiftLeft(Math.toIntExact(-shift)));
    }
    private static boolean rawNegative(DataType type, Object value) {
        return switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits((double) value) < 0;
            case FLOAT32 -> Float.floatToRawIntBits((float) value) < 0;
            case BFLOAT16 -> (short) value < 0;
            default -> false;
        };
    }

    /**
     * Independently evaluates complete directional masked SUM/MEAN output cells.
     *
     * <p>The oracle derives right-aligned broadcast coordinates itself, reads the mask before the
     * corresponding data value, and uses independent {@link BigInteger} rational arithmetic for
     * the selected represented values.</p>
     *
     * @param ir non-null masked-reduction semantics
     * @param geometry non-null matching resolved data/mask/output layouts
     * @param arguments non-null ordered data, mask, and writable output carriers
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if IR, geometry, or boundary facts disagree
     */
    public static void execute(CpuMaskedReductionIr ir,
            CpuMaskedReductionLowering.Geometry geometry,
            List<CpuBufferArgument> arguments) {
        Objects.requireNonNull(ir, "ir"); Objects.requireNonNull(geometry, "geometry");
        arguments = List.copyOf(arguments);
        if (arguments.size() != 3 || ir.kind() != geometry.kind()
                || ir.dataType() != geometry.dataType() || ir.axis() != geometry.axis()) {
            throw new IllegalArgumentException("masked-reduction reference facts disagree");
        }
        long[] dataExtents = geometry.data().extents();
        long[] maskExtents = geometry.mask().extents();
        long[] outputExtents = geometry.output().extents();
        long[] dataCoordinates = new long[dataExtents.length];
        long[] maskCoordinates = new long[maskExtents.length];
        long[] outputCoordinates = new long[outputExtents.length];
        int omitted = dataExtents.length - maskExtents.length;
        for (long cell = 0; cell < geometry.outputCount(); cell++) {
            decodeReference(cell, outputExtents, outputCoordinates);
            for (int dataAxis = 0, outputAxis = 0; dataAxis < dataExtents.length; dataAxis++) {
                dataCoordinates[dataAxis] = dataAxis == geometry.axis()
                        ? 0 : outputCoordinates[outputAxis++];
            }
            var selected = new java.util.ArrayList<Object>();
            for (long coordinate = 0; coordinate < geometry.maximumDomainCount(); coordinate++) {
                dataCoordinates[geometry.axis()] = coordinate;
                for (int maskAxis = 0; maskAxis < maskExtents.length; maskAxis++) {
                    int dataAxis = omitted + maskAxis;
                    maskCoordinates[maskAxis] = maskExtents[maskAxis] == 1
                            ? 0 : dataCoordinates[dataAxis];
                }
                byte mask = (byte) load(arguments.get(1), DataType.BOOL,
                        maskedAddress(geometry.mask(), maskCoordinates));
                if (mask == 0) continue;
                selected.add(load(arguments.get(0), ir.dataType(),
                        maskedAddress(geometry.data(), dataCoordinates)));
            }
            Object result = maskedFloatingReference(ir.kind(), ir.dataType(), selected);
            store(arguments.get(2), ir.dataType(),
                    maskedAddress(geometry.output(), outputCoordinates), result);
        }
    }

    private static long maskedAddress(CpuMaskedReductionLowering.Layout layout,
            long[] coordinates) {
        long result = layout.offset(); long[] strides = layout.strides();
        for (int axis = 0; axis < coordinates.length; axis++) result = Math.addExact(result,
                Math.multiplyExact(coordinates[axis], strides[axis]));
        return result;
    }

    private static Object maskedFloatingReference(CpuMaskedReductionIr.Kind kind, DataType type,
            List<Object> selected) {
        int fractionBits = type == DataType.FLOAT64 ? 52 : type == DataType.FLOAT32 ? 23 : 7;
        int bias = type == DataType.FLOAT64 ? 1023 : 127;
        int unitExponent = type == DataType.FLOAT64 ? -1074
                : type == DataType.FLOAT32 ? -149 : -133;
        long exponentMask = type == DataType.FLOAT64 ? 0x7ffL : 0xffL;
        long signMask = type == DataType.FLOAT64 ? Long.MIN_VALUE
                : type == DataType.FLOAT32 ? 1L << 31 : 1L << 15;
        long fractionMask = (1L << fractionBits) - 1;
        BigInteger exact = BigInteger.ZERO;
        boolean nan = false, positiveInfinity = false, negativeInfinity = false;
        boolean positiveZero = false, negativeZero = false, nonzero = false;
        for (Object represented : selected) {
            long bits = type == DataType.FLOAT64 ? Double.doubleToRawLongBits((double) represented)
                    : type == DataType.FLOAT32 ? Integer.toUnsignedLong(
                            Float.floatToRawIntBits((float) represented))
                    : Short.toUnsignedLong((short) represented);
            long fraction = bits & fractionMask;
            long exponentField = bits >>> fractionBits & exponentMask;
            boolean negative = (bits & signMask) != 0;
            if (exponentField == exponentMask) {
                if (fraction != 0) nan = true;
                else if (negative) negativeInfinity = true; else positiveInfinity = true;
                continue;
            }
            if (exponentField == 0 && fraction == 0) {
                if (negative) negativeZero = true; else positiveZero = true;
                continue;
            }
            nonzero = true;
            long significand = exponentField == 0 ? fraction : (1L << fractionBits) | fraction;
            int exponent = exponentField == 0 ? 1 - bias - fractionBits
                    : Math.toIntExact(exponentField) - bias - fractionBits;
            BigInteger coefficient = BigInteger.valueOf(significand)
                    .shiftLeft(exponent - unitExponent);
            exact = exact.add(negative ? coefficient.negate() : coefficient);
        }
        long canonical = type == DataType.FLOAT64 ? 0x7ff8000000000000L
                : type == DataType.FLOAT32 ? 0x7fc00000L : 0x7fc0L;
        long infinity = exponentMask << fractionBits;
        long result;
        if (nan || positiveInfinity && negativeInfinity
                || kind == CpuMaskedReductionIr.Kind.MEAN && selected.isEmpty()) {
            result = canonical;
        } else if (positiveInfinity) result = infinity;
        else if (negativeInfinity) result = signMask | infinity;
        else if (exact.signum() == 0) result = !selected.isEmpty() && negativeZero
                && !positiveZero && !nonzero ? signMask : 0;
        else result = roundRational(exact.abs(), unitExponent,
                kind == CpuMaskedReductionIr.Kind.MEAN
                        ? BigInteger.valueOf(selected.size()) : BigInteger.ONE,
                exact.signum() < 0, fractionBits, bias, signMask);
        return switch (type) {
            case FLOAT64 -> Double.longBitsToDouble(result);
            case FLOAT32 -> Float.intBitsToFloat((int) result);
            case BFLOAT16 -> (short) result;
            default -> throw new AssertionError("masked reduction requires floating data");
        };
    }

    /**
     * Independently evaluates complete arg-extrema output cells in logical coordinate order.
     *
     * @param ir non-null arg-extrema semantics
     * @param geometry non-null matching one-axis layout geometry
     * @param arguments non-null numeric input and writable INT64 output carriers
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if IR, geometry, or carrier facts disagree
     * @throws ArithmeticException if logical-to-physical address arithmetic overflows
     */
    public static void execute(CpuArgExtremaIr ir, CpuArgExtremaLowering.Geometry geometry,
            List<CpuBufferArgument> arguments) {
        Objects.requireNonNull(ir, "ir");
        Objects.requireNonNull(geometry, "geometry");
        arguments = List.copyOf(arguments);
        if (arguments.size() != 2 || ir.kind() != geometry.kind()
                || ir.inputType() != geometry.inputType() || ir.axis() != geometry.axis()
                || ir.keepDimensions() != geometry.keepDimensions()
                || ir.tiePolicy() != geometry.tiePolicy()) {
            throw new IllegalArgumentException("arg-extrema reference facts disagree");
        }
        long[] inputExtents = geometry.input().extents();
        long[] inputStrides = geometry.input().strides();
        long[] outputExtents = geometry.output().extents();
        long[] outputStrides = geometry.output().strides();
        long[] inputCoordinates = new long[inputExtents.length];
        long[] outputCoordinates = new long[outputExtents.length];
        for (long cell = 0; cell < geometry.outputCount(); cell++) {
            long remaining = cell;
            for (int outputAxis = outputExtents.length - 1; outputAxis >= 0; outputAxis--) {
                long coordinate = remaining % outputExtents[outputAxis];
                remaining /= outputExtents[outputAxis];
                outputCoordinates[outputAxis] = coordinate;
                int inputAxis = geometry.keepDimensions() ? outputAxis
                        : outputAxis < geometry.axis() ? outputAxis : outputAxis + 1;
                if (inputAxis != geometry.axis()) inputCoordinates[inputAxis] = coordinate;
            }
            long bestIndex = 0;
            inputCoordinates[geometry.axis()] = 0;
            Object best = load(arguments.getFirst(), ir.inputType(),
                    address(geometry.input().offset(), inputStrides, inputCoordinates));
            for (long candidateIndex = 1; candidateIndex < geometry.axisExtent(); candidateIndex++) {
                inputCoordinates[geometry.axis()] = candidateIndex;
                Object candidate = load(arguments.getFirst(), ir.inputType(),
                        address(geometry.input().offset(), inputStrides, inputCoordinates));
                int comparison = argCompare(ir.inputType(), candidate, best);
                boolean candidateNaN = floatingNaN(ir.inputType(), candidate);
                boolean bestNaN = floatingNaN(ir.inputType(), best);
                boolean better = candidateNaN && !bestNaN
                        || !candidateNaN && !bestNaN
                            && (ir.kind() == CpuArgExtremaIr.Kind.ARG_MIN
                                ? comparison < 0 : comparison > 0)
                        || comparison == 0
                            && ir.tiePolicy() == io.github.pho001.synaptik.model.operation.reduction
                                .ArgExtremaTiePolicy.LAST_INDEX;
                if (better) { best = candidate; bestIndex = candidateIndex; }
            }
            store(arguments.getLast(), DataType.INT64,
                    address(geometry.output().offset(), outputStrides, outputCoordinates), bestIndex);
        }
    }

    private static long address(long offset, long[] strides, long[] coordinates) {
        long result = offset;
        for (int axis = 0; axis < strides.length; axis++) result = Math.addExact(result,
                Math.multiplyExact(strides[axis], coordinates[axis]));
        return result;
    }

    private static int argCompare(DataType type, Object left, Object right) {
        return switch (type) {
            case FLOAT64 -> Double.compare((double) left, (double) right);
            case FLOAT32 -> Float.compare((float) left, (float) right);
            case BFLOAT16 -> Float.compare(bfloat((short) left), bfloat((short) right));
            case INT32 -> Integer.compare((int) left, (int) right);
            case INT64 -> Long.compare((long) left, (long) right);
            case BOOL -> throw new AssertionError();
        };
    }

    private static boolean floatingNaN(DataType type, Object value) {
        return switch (type) {
            case FLOAT64 -> Double.isNaN((double) value);
            case FLOAT32 -> Float.isNaN((float) value);
            case BFLOAT16 -> Float.isNaN(bfloat((short) value));
            case INT32, INT64 -> false;
            case BOOL -> throw new AssertionError();
        };
    }

    /**
     * Independently evaluates a complete cumulative scan using logical coordinates.
     * @param ir non-null scan semantics
     * @param geometry non-null matching layouts and slice geometry
     * @param arguments non-null exact two-element input/output carrier list; read but not retained
     * @throws NullPointerException if {@code ir}, {@code geometry}, or {@code arguments} is null
     * @throws IllegalArgumentException if the IR, geometry, or carrier count disagrees
     */
    public static void execute(CpuScanIr ir, CpuScanLowering.Geometry geometry,
            List<CpuBufferArgument> arguments) {
        Objects.requireNonNull(ir, "ir"); Objects.requireNonNull(geometry, "geometry");
        if (arguments.size() != 2 || ir.kind() != geometry.kind()
                || ir.dataType() != geometry.dataType())
            throw new IllegalArgumentException("scan reference facts disagree");
        long[] extents = geometry.input().extents();
        long[] coordinates = new long[extents.length];
        for (long slice = 0; slice < geometry.sliceCount(); slice++) {
            long remaining = slice;
            for (int axis = extents.length - 1; axis >= 0; axis--) {
                if (axis == geometry.axis()) continue;
                coordinates[axis] = remaining % extents[axis]; remaining /= extents[axis];
            }
            Object accumulator = scanIdentity(ir.kind(), ir.dataType());
            for (long step = 0; step < geometry.axisExtent(); step++) {
                coordinates[geometry.axis()] = geometry.reverse()
                        ? geometry.axisExtent() - 1 - step : step;
                Object value = load(arguments.get(0), ir.dataType(),
                        scanAddress(geometry.input(), coordinates));
                if (!geometry.exclusive()) accumulator = scanApply(ir.kind(), ir.dataType(), accumulator, value);
                store(arguments.get(1), ir.dataType(), scanAddress(geometry.output(), coordinates), accumulator);
                if (geometry.exclusive()) accumulator = scanApply(ir.kind(), ir.dataType(), accumulator, value);
            }
        }
    }

    private static long scanAddress(CpuScanLowering.Layout layout, long[] coordinates) {
        long result = layout.offset(); long[] strides = layout.strides();
        for (int axis = 0; axis < coordinates.length; axis++) result += coordinates[axis] * strides[axis];
        return result;
    }
    private static Object scanIdentity(CpuScanIr.Kind kind, DataType type) {
        boolean sum = kind == CpuScanIr.Kind.CUM_SUM;
        return switch (type) {
            case FLOAT64 -> sum ? 0.0d : 1.0d; case FLOAT32 -> sum ? 0.0f : 1.0f;
            case BFLOAT16 -> (short) (sum ? 0 : 0x3f80); case INT32 -> sum ? 0 : 1;
            case INT64 -> sum ? 0L : 1L; case BOOL -> throw new AssertionError();
        };
    }
    private static Object scanApply(CpuScanIr.Kind kind, DataType type, Object left, Object right) {
        boolean sum = kind == CpuScanIr.Kind.CUM_SUM;
        return switch (type) {
            case FLOAT64 -> sum ? (double) left + (double) right : (double) left * (double) right;
            case FLOAT32 -> sum ? (float) left + (float) right : (float) left * (float) right;
            case BFLOAT16 -> toBfloat(sum ? bfloat((short) left) + bfloat((short) right)
                    : bfloat((short) left) * bfloat((short) right));
            case INT32 -> sum ? (int) left + (int) right : (int) left * (int) right;
            case INT64 -> sum ? (long) left + (long) right : (long) left * (long) right;
            case BOOL -> throw new AssertionError();
        };
    }

    /**
     * Independently evaluates one explicit-state initializer or dropout occurrence.
     *
     * @param ir non-null exact CPU-private random identity
     * @param geometry non-null compatible cold layouts and draw count
     * @param arguments non-null ordered output for initialization or
     *     value/state/output/mask/next-state carriers
     * @throws NullPointerException if a required reference or argument element is null
     * @throws IllegalArgumentException if family or boundary facts disagree
     * @throws ArithmeticException if a heap address does not fit an array index
     * @throws IndexOutOfBoundsException if geometry addresses outside a supplied carrier
     */
    public static void execute(CpuRandomIr ir, CpuRandomLowering.Geometry geometry,
            List<CpuBufferArgument> arguments) {
        Objects.requireNonNull(ir, "ir"); Objects.requireNonNull(geometry, "geometry");
        if (ir.family() != geometry.family() || arguments.size() != geometry.boundaries().size())
            throw new IllegalArgumentException("random reference facts disagree");
        if (ir.family() == CpuRandomIr.Family.INITIAL_STATE) {
            store(arguments.getFirst(), DataType.INT64, randomAddress(
                    geometry.boundaries().getFirst(), 0), ir.keyBits());
            store(arguments.getFirst(), DataType.INT64, randomAddress(
                    geometry.boundaries().getFirst(), 1), ir.counterBits());
            return;
        }
        long key = (long) load(arguments.get(1), DataType.INT64,
                randomAddress(geometry.boundaries().get(1), 0));
        long counter = (long) load(arguments.get(1), DataType.INT64,
                randomAddress(geometry.boundaries().get(1), 1));
        double probability = Double.longBitsToDouble(ir.probabilityBits());
        double denominator = 1.0d - probability;
        long keyOffset = randomMix(key + RANDOM_KEY_BIAS);
        for (long logical = 0; logical < geometry.elementCount(); logical++) {
            long mapped = randomMix(counter + logical + keyOffset);
            boolean keep = (mapped >>> 11) * 0x1.0p-53 >= probability;
            store(arguments.get(3), DataType.BOOL,
                    randomAddress(geometry.boundaries().get(3), logical), (byte) (keep ? 1 : 0));
            if (ir.valueType() == DataType.FLOAT64) {
                double value = keep ? (double) load(arguments.get(0), DataType.FLOAT64,
                        randomAddress(geometry.boundaries().get(0), logical)) / denominator : 0.0d;
                store(arguments.get(2), DataType.FLOAT64,
                        randomAddress(geometry.boundaries().get(2), logical), value);
            } else {
                float input = keep ? (float) load(arguments.get(0), DataType.FLOAT32,
                        randomAddress(geometry.boundaries().get(0), logical)) : 0.0f;
                float value = keep ? (float) (((double) input) / denominator) : 0.0f;
                store(arguments.get(2), DataType.FLOAT32,
                        randomAddress(geometry.boundaries().get(2), logical), value);
            }
        }
        store(arguments.get(4), DataType.INT64,
                randomAddress(geometry.boundaries().get(4), 0), key);
        store(arguments.get(4), DataType.INT64,
                randomAddress(geometry.boundaries().get(4), 1), counter + geometry.elementCount());
    }

    private static long randomMix(long value) {
        value = (value ^ (value >>> 30)) * RANDOM_M1;
        value = (value ^ (value >>> 27)) * RANDOM_M2;
        return value ^ (value >>> 31);
    }

    private static long randomAddress(CpuRandomLowering.Layout layout, long logical) {
        long[] extents = layout.extents(), strides = layout.strides();
        long address = layout.offset();
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            long coordinate = extents[axis] == 0 ? 0 : logical % extents[axis];
            if (extents[axis] != 0) logical /= extents[axis];
            address += coordinate * strides[axis];
        }
        return address;
    }

    /**
     * Independently evaluates one stable ordering occurrence using primitive-index insertion.
     *
     * <p>The oracle writes represented values without conversion, emits zero-based logical-axis
     * INT64 indices, keeps floating NaNs last in both directions, distinguishes signed zero, and
     * uses increasing logical indices for equal values. Unsorted TOP_K reorders the selected set
     * by increasing logical index. The caller owns all carriers; this method allocates only its
     * cold reference index array and does not mutate the input.</p>
     *
     * @param ir non-null structural ordering identity
     * @param geometry non-null compatible complete cold geometry
     * @param arguments non-null ordered input then one or two writable output arguments
     * @throws NullPointerException if a required reference or argument element is null
     * @throws IllegalArgumentException if family or boundary cardinality disagrees
     * @throws ArithmeticException if exact address arithmetic overflows
     * @throws IndexOutOfBoundsException if geometry exceeds a supplied carrier
     * @throws IllegalStateException if a supplied segment is not accessible
     */
    public static void execute(CpuOrderingIr ir, CpuOrderingLowering.Geometry geometry,
            List<CpuBufferArgument> arguments) {
        Objects.requireNonNull(ir, "ir"); Objects.requireNonNull(geometry, "geometry");
        if (arguments.size() != geometry.boundaries().size() || ir.family() != geometry.family())
            throw new IllegalArgumentException("ordering reference facts disagree");
        long axisExtent = geometry.boundaries().getFirst().extents()[geometry.axis()];
        long[] indices = new long[Math.toIntExact(axisExtent)];
        for (long slice = 0; slice < geometry.sliceCount(); slice++) {
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            for (int i = 1; i < indices.length; i++) {
                long selected = indices[i]; int j = i;
                while (j > 0 && orderingCompare(arguments.getFirst(), geometry, slice,
                        indices[j - 1], selected) > 0) indices[j] = indices[--j];
                indices[j] = selected;
            }
            int count = Math.toIntExact(geometry.family() == CpuOrderingIr.Family.TOP_K
                    ? geometry.k() : axisExtent);
            if (geometry.family() == CpuOrderingIr.Family.TOP_K && !geometry.sorted())
                java.util.Arrays.sort(indices, 0, count);
            for (int position = 0; position < count; position++) {
                long index = indices[position];
                if (geometry.family() == CpuOrderingIr.Family.ARGSORT) {
                    store(arguments.get(1), DataType.INT64,
                            orderingAddress(geometry.boundaries().get(1), slice, geometry.axis(), position), index);
                } else {
                    Object value = load(arguments.getFirst(), geometry.dataType(),
                            orderingAddress(geometry.boundaries().getFirst(), slice, geometry.axis(), index));
                    store(arguments.get(1), geometry.dataType(),
                            orderingAddress(geometry.boundaries().get(1), slice, geometry.axis(), position), value);
                    if (geometry.family() == CpuOrderingIr.Family.TOP_K) store(arguments.get(2),
                            DataType.INT64, orderingAddress(geometry.boundaries().get(2), slice,
                                    geometry.axis(), position), index);
                }
            }
        }
    }

    private static int orderingCompare(CpuBufferArgument input, CpuOrderingLowering.Geometry geometry,
            long slice, long leftIndex, long rightIndex) {
        Object left = load(input, geometry.dataType(), orderingAddress(geometry.boundaries().getFirst(),
                slice, geometry.axis(), leftIndex));
        Object right = load(input, geometry.dataType(), orderingAddress(geometry.boundaries().getFirst(),
                slice, geometry.axis(), rightIndex));
        int comparison = switch (geometry.dataType()) {
            case FLOAT64 -> orderedFloating((double) left, (double) right, geometry.descending());
            case FLOAT32 -> orderedFloating((float) left, (float) right, geometry.descending());
            case BFLOAT16 -> orderedFloating(Float.intBitsToFloat(Short.toUnsignedInt((short) left) << 16),
                    Float.intBitsToFloat(Short.toUnsignedInt((short) right) << 16), geometry.descending());
            case INT32 -> Integer.compare((int) left, (int) right);
            case INT64 -> Long.compare((long) left, (long) right);
            case BOOL -> Byte.compare((byte) left, (byte) right);
        };
        return geometry.descending() && geometry.dataType() != DataType.FLOAT64
                && geometry.dataType() != DataType.FLOAT32 && geometry.dataType() != DataType.BFLOAT16
                ? -comparison : comparison;
    }

    private static int orderedFloating(double left, double right, boolean descending) {
        boolean ln = Double.isNaN(left), rn = Double.isNaN(right);
        if (ln || rn) return ln == rn ? 0 : ln ? 1 : -1;
        int result = Double.compare(left, right); return descending ? -result : result;
    }

    private static long orderingAddress(CpuOrderingLowering.Layout layout, long slice, int axis,
            long selected) {
        long[] extents = layout.extents(), strides = layout.strides(); long result = layout.offset();
        for (int current = extents.length - 1; current >= 0; current--) {
            long coordinate;
            if (current == axis) coordinate = selected;
            else { coordinate = extents[current] == 0 ? 0 : slice % extents[current];
                if (extents[current] != 0) slice /= extents[current]; }
            result = Math.addExact(result, Math.multiplyExact(coordinate, strides[current]));
        }
        return result;
    }

    /**
     * Independently evaluates one zero-initialized overlap fold for differential evidence.
     * This cold oracle constructs logical coordinates directly and shares no packed traversal or
     * generated-emitter helper with the portable fold body.
     *
     * @param ir non-null fold structural identity
     * @param geometry non-null matching static fold geometry
     * @param arguments exact read-only input followed by writable output
     * @param start inclusive flattened output ordinal
     * @param end exclusive flattened output ordinal
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if IR, geometry, arguments, or range disagree
     */
    public static void execute(CpuFoldIr ir, CpuFoldLowering.Geometry geometry,
            List<CpuBufferArgument> arguments, long start, long end) {
        Objects.requireNonNull(ir, "ir"); Objects.requireNonNull(geometry, "geometry");
        if (ir.family() != geometry.family() || ir.dataType() != geometry.dataType()
                || arguments.size() != 2) {
            throw new IllegalArgumentException("fold reference facts disagree");
        }
        long outputCount = count(geometry.output().extents());
        if (start < 0 || end < start || end > outputCount) {
            throw new IllegalArgumentException("invalid fold reference bounds");
        }
        long inputCount = count(geometry.input().extents());
        DataType type = geometry.dataType();
        for (long outputOrdinal = start; outputOrdinal < end; outputOrdinal++) {
            long[] outputCoordinate = coordinates(outputOrdinal, geometry.output().extents());
            Object sum = positiveZero(type);
            for (long inputOrdinal = 0; inputOrdinal < inputCount; inputOrdinal++) {
                long[] inputCoordinate = coordinates(inputOrdinal, geometry.input().extents());
                if (!foldMatches(geometry, inputCoordinate, outputCoordinate)) continue;
                Object value = load(arguments.getFirst(), type,
                        foldAddress(geometry.input(), inputCoordinate));
                sum = foldAdd(sum, value, type);
            }
            store(arguments.getLast(), type, foldAddress(geometry.output(), outputCoordinate), sum);
        }
    }

    private static boolean foldMatches(CpuFoldLowering.Geometry geometry, long[] input,
            long[] output) {
        if (geometry.mapping() instanceof CpuFoldLowering.AxisGeometry axis) {
            for (int current = 0; current < output.length; current++) {
                long target = current == axis.axis()
                        ? Math.addExact(Math.multiplyExact(input[current], axis.step()),
                                input[input.length - 1])
                        : input[current];
                if (target != output[current]) return false;
            }
            return true;
        }
        var two = (CpuFoldLowering.TwoDimensionalGeometry) geometry.mapping();
        long kernelArea = Math.multiplyExact(two.kernelHeight(), two.kernelWidth());
        long channel = input[1] / kernelArea;
        long kernel = input[1] % kernelArea;
        long kernelHeight = kernel / two.kernelWidth();
        long kernelWidth = kernel % two.kernelWidth();
        long columnHeight = input[2] / two.outputColumnsWidth();
        long columnWidth = input[2] % two.outputColumnsWidth();
        long height = Math.addExact(Math.subtractExact(
                Math.multiplyExact(columnHeight, two.strideHeight()), two.paddingHeight()),
                Math.multiplyExact(kernelHeight, two.dilationHeight()));
        long width = Math.addExact(Math.subtractExact(
                Math.multiplyExact(columnWidth, two.strideWidth()), two.paddingWidth()),
                Math.multiplyExact(kernelWidth, two.dilationWidth()));
        return input[0] == output[0] && channel == output[1]
                && height == output[2] && width == output[3];
    }

    private static long foldAddress(CpuFoldLowering.Layout layout, long[] coordinate) {
        long result = layout.offset(); long[] strides = layout.strides();
        for (int axis = 0; axis < coordinate.length; axis++) result = Math.addExact(result,
                Math.multiplyExact(coordinate[axis], strides[axis]));
        return result;
    }

    private static Object positiveZero(DataType type) {
        return switch (type) {
            case FLOAT64 -> 0.0d; case FLOAT32 -> 0.0f; case BFLOAT16 -> (short) 0;
            case INT32 -> 0; case INT64 -> 0L;
            case BOOL -> throw new AssertionError("BOOL fold is unsupported");
        };
    }

    private static Object foldAdd(Object left, Object right, DataType type) {
        return switch (type) {
            case FLOAT64 -> (double) left + (double) right;
            case FLOAT32 -> (float) left + (float) right;
            case BFLOAT16 -> toBfloat(bfloat((short) left) + bfloat((short) right));
            case INT32 -> (int) left + (int) right;
            case INT64 -> (long) left + (long) right;
            case BOOL -> throw new AssertionError("BOOL fold is unsupported");
        };
    }

    /**
     * Independently validates and evaluates one functional scatter for differential evidence.
     * This cold reference deliberately uses coordinate arrays, target lists, and {@link
     * java.math.BigInteger} exact products; it shares no generated emitter or primitive-limb
     * arithmetic and is never selected as a Runtime fallback.
     *
     * @param ir non-null scatter structural identity
     * @param geometry non-null matching static scatter geometry
     * @param arguments non-null unique inputs followed by the writable output
     * @param start inclusive output logical ordinal
     * @param end exclusive output logical ordinal
     * @throws IllegalArgumentException if boundaries, range, or NONE uniqueness are invalid
     * @throws IndexOutOfBoundsException if an index is outside its selected data extent
     */
    public static void execute(CpuScatterIr ir, CpuScatterLowering.Geometry geometry,
            List<CpuBufferArgument> arguments, long start, long end) {
        Objects.requireNonNull(ir,"ir");Objects.requireNonNull(geometry,"geometry");
        if(arguments.size()!=geometry.boundaries().size())throw new IllegalArgumentException(
                "scatter reference boundary count is inconsistent");
        int db=geometry.occurrenceToBoundary().get(0),ib=geometry.occurrenceToBoundary().get(1),
                ub=geometry.occurrenceToBoundary().get(2);
        var dl=geometry.boundaries().get(db);var il=geometry.boundaries().get(ib);
        var ul=geometry.boundaries().get(ub);var ol=geometry.boundaries().getLast();
        long indexCount=count(il.extents());
        for(long ordinal=0;ordinal<indexCount;ordinal++){
            long[] coordinate=coordinates(ordinal,il.extents());
            long value=((Number)load(arguments.get(ib),geometry.boundaryTypes().get(ib),
                    address(il,coordinate))).longValue();
            int axis=ir.family()==CpuScatterIr.Family.SCATTER_ND
                    ?geometry.batchDimensions()+(int)(ordinal%geometry.tupleDepth()):geometry.axis();
            long extent=dl.extents()[axis];
            if(value<0||value>=extent)throw new IndexOutOfBoundsException(ir.family().name()
                    +" index at logical position "+ordinal+" for data axis "+axis
                    +" is out of bounds: value="+value+", extent="+extent);
        }
        long updateCount=count(ul.extents());
        var targets=new java.util.ArrayList<long[]>(Math.toIntExact(Math.min(updateCount,Integer.MAX_VALUE)));
        for(long ordinal=0;ordinal<updateCount;ordinal++)targets.add(scatterTarget(ir,geometry,
                arguments.get(ib),geometry.boundaryTypes().get(ib),il,
                coordinates(ordinal,ul.extents()),dl.extents()));
        if(ir.reduction()==ScatterReduction.NONE){
            if(ir.family()==CpuScatterIr.Family.SCATTER_ND){
                long[] tupleExtents=java.util.Arrays.copyOf(il.extents(),il.extents().length-1);
                var tupleTargets=new java.util.ArrayList<long[]>();
                for(long ordinal=0;ordinal<product(tupleExtents);ordinal++){
                    long[] tupleCoordinate=coordinates(ordinal,tupleExtents);
                    long[] updateCoordinate=new long[ul.extents().length];
                    System.arraycopy(tupleCoordinate,0,updateCoordinate,0,tupleCoordinate.length);
                    tupleTargets.add(prefix(scatterTarget(ir,geometry,arguments.get(ib),
                            geometry.boundaryTypes().get(ib),il,updateCoordinate,dl.extents()),
                            dl.extents().length-geometry.batchDimensions()-geometry.tupleDepth()));
                }
                for(int later=1;later<tupleTargets.size();later++)for(int first=0;first<later;first++)
                    if(java.util.Arrays.equals(tupleTargets.get(later),tupleTargets.get(first)))
                        throw new IllegalArgumentException("SCATTER_ND duplicate target tuple at logical tuple position "+later+"; first addressed at logical tuple position "+first);
            }else for(int later=1;later<targets.size();later++)for(int first=0;first<later;first++)
                if(java.util.Arrays.equals(targets.get(later),targets.get(first)))
                    throw new IllegalArgumentException("SCATTER_ELEMENTS duplicate target at logical update position "+later+"; first addressed at logical update position "+first);
        }
        long outputCount=count(ol.extents());
        if(start<0||end<start||end>outputCount)throw new IllegalArgumentException("invalid reference bounds");
        DataType type=geometry.boundaryTypes().get(db);
        for(long ordinal=start;ordinal<end;ordinal++){
            long[] coordinate=coordinates(ordinal,ol.extents());
            Object base=load(arguments.get(db),type,address(dl,coordinate));
            Object result=base;boolean found=false;
            var productBits=new java.util.ArrayList<Long>();
            for(int update=0;update<targets.size();update++)if(java.util.Arrays.equals(coordinate,targets.get(update))){
                Object value=load(arguments.get(ub),type,address(ul,coordinates(update,ul.extents())));
                if(ir.reduction()==ScatterReduction.NONE)result=value;
                else if(ir.reduction()==ScatterReduction.MUL&&floatingType(type)){
                    if(!found)productBits.add(rawBits(base,type));productBits.add(rawBits(value,type));
                }else result=referenceReduce(result,value,type,ir.reduction());
                found=true;
            }
            if(found&&ir.reduction()==ScatterReduction.MUL&&floatingType(type))
                result=fromRawBits(referenceProduct(productBits,type),type);
            store(arguments.getLast(),type,address(ol,coordinate),result);
        }
    }

    private static long[] prefix(long[] value,int removedSuffix){return java.util.Arrays.copyOf(value,value.length-removedSuffix);}
    private static long address(CpuScatterLowering.Geometry.Layout layout,long[] coordinate){long result=layout.offset();long[] strides=layout.strides();for(int i=0;i<coordinate.length;i++)result=Math.addExact(result,Math.multiplyExact(coordinate[i],strides[i]));return result;}
    private static long product(long[] values){long r=1;for(long v:values)r=Math.multiplyExact(r,v);return r;}
    private static long[] scatterTarget(CpuScatterIr ir,CpuScatterLowering.Geometry g,
            CpuBufferArgument indices,DataType indexType,CpuScatterLowering.Geometry.Layout il,
            long[] update,long[] dataExtents){
        long[] target=new long[dataExtents.length];
        if(ir.family()==CpuScatterIr.Family.SCATTER_ELEMENTS){System.arraycopy(update,0,target,0,target.length);target[g.axis()]=((Number)load(indices,indexType,address(il,update))).longValue();return target;}
        if(ir.family()==CpuScatterIr.Family.SCATTER_ADD){int q=il.extents().length;for(int a=0;a<g.axis();a++)target[a]=update[a];long[] index=java.util.Arrays.copyOfRange(update,g.axis(),g.axis()+q);target[g.axis()]=((Number)load(indices,indexType,address(il,index))).longValue();for(int a=g.axis()+1;a<target.length;a++)target[a]=update[g.axis()+q+a-g.axis()-1];return target;}
        int q=il.extents().length;for(int a=0;a<g.batchDimensions();a++)target[a]=update[a];long[] index=new long[q];System.arraycopy(update,0,index,0,q-1);for(int k=0;k<g.tupleDepth();k++){index[q-1]=k;target[g.batchDimensions()+k]=((Number)load(indices,indexType,address(il,index))).longValue();}for(int a=g.batchDimensions()+g.tupleDepth();a<target.length;a++)target[a]=update[q-1+a-g.batchDimensions()-g.tupleDepth()];return target;
    }

    private static boolean floatingType(DataType t){return t==DataType.FLOAT64||t==DataType.FLOAT32||t==DataType.BFLOAT16;}
    private static long rawBits(Object v,DataType t){return switch(t){case FLOAT64->Double.doubleToRawLongBits((double)v);case FLOAT32->Integer.toUnsignedLong(Float.floatToRawIntBits((float)v));case BFLOAT16->Short.toUnsignedLong((short)v);default->((Number)v).longValue();};}
    private static Object fromRawBits(long v,DataType t){return switch(t){case FLOAT64->Double.longBitsToDouble(v);case FLOAT32->Float.intBitsToFloat((int)v);case BFLOAT16->(short)v;default->v;};}
    private static Object referenceReduce(Object a,Object b,DataType t,ScatterReduction r){if(t==DataType.INT32){int x=(int)a,y=(int)b;return switch(r){case ADD->x+y;case MUL->x*y;case MIN->Math.min(x,y);case MAX->Math.max(x,y);default->b;};}if(t==DataType.INT64){long x=(long)a,y=(long)b;return switch(r){case ADD->x+y;case MUL->x*y;case MIN->Math.min(x,y);case MAX->Math.max(x,y);default->b;};}double x=t==DataType.FLOAT64?(double)a:t==DataType.FLOAT32?(float)a:bfloat((short)a),y=t==DataType.FLOAT64?(double)b:t==DataType.FLOAT32?(float)b:bfloat((short)b);double z=r==ScatterReduction.ADD?x+y:Double.isNaN(x)||Double.isNaN(y)?Double.NaN:r==ScatterReduction.MIN?(x==0&&y==0&&(Double.doubleToRawLongBits(x)<0||Double.doubleToRawLongBits(y)<0)?-0.0:Math.min(x,y)):(x==0&&y==0&&(Double.doubleToRawLongBits(x)>=0||Double.doubleToRawLongBits(y)>=0)?0.0:Math.max(x,y));return switch(t){case FLOAT64->z;case FLOAT32->(float)z;case BFLOAT16->toBfloat((float)z);default->throw new AssertionError(t);};}
    private static float bfloat(short v){return Float.intBitsToFloat(Short.toUnsignedInt(v)<<16);}
    private static short toBfloat(float v){int bits=Float.floatToRawIntBits(v);if((bits&0x7f800000)==0x7f800000&&(bits&0x7fffff)!=0)return(short)((bits>>>16)|0x40);int upper=bits>>>16,lower=bits&0xffff;if(lower>0x8000||(lower==0x8000&&(upper&1)!=0))upper++;return(short)upper;}

    private static long referenceProduct(List<Long> factors,DataType type){boolean negative=false,zero=false,infinity=false,nan=false;java.math.BigInteger product=java.math.BigInteger.ONE;long exponent=0;int fractionBits=type==DataType.FLOAT64?52:type==DataType.FLOAT32?23:7,exponentBits=type==DataType.FLOAT64?11:8,bias=type==DataType.FLOAT64?1023:127,total=type==DataType.FLOAT64?64:type==DataType.FLOAT32?32:16;long fractionMask=(1L<<fractionBits)-1,exponentMask=(1L<<exponentBits)-1;for(long bits:factors){if((bits&(1L<<(total-1)))!=0)negative=!negative;long f=bits&fractionMask,e=(bits>>>fractionBits)&exponentMask;if(e==exponentMask){if(f!=0)nan=true;else infinity=true;}else if(e==0&&f==0)zero=true;else{long significand=e==0?f:(1L<<fractionBits)|f;product=product.multiply(java.math.BigInteger.valueOf(significand));exponent+=(e==0?1-bias:e-bias)-fractionBits;}}long sign=negative?1L<<(total-1):0;if(nan||zero&&infinity)return sign|(type==DataType.FLOAT64?0x7ff8000000000000L:type==DataType.FLOAT32?0x7fc00000L:0x7fc0L);if(infinity)return sign|(type==DataType.FLOAT64?0x7ff0000000000000L:type==DataType.FLOAT32?0x7f800000L:0x7f80L);if(zero)return sign;int precision=fractionBits+1,minNormal=1-bias,maxExponent=bias;long unbiased=exponent+product.bitLength()-1;if(unbiased>maxExponent)return sign|(exponentMask<<fractionBits);long shift=unbiased>=minNormal?product.bitLength()-precision:(minNormal-fractionBits)-exponent;java.math.BigInteger q=roundShift(product,shift);if(unbiased>=minNormal){if(q.bitLength()>precision){q=q.shiftRight(1);unbiased++;}if(unbiased>maxExponent)return sign|(exponentMask<<fractionBits);return sign|((unbiased+bias)<<fractionBits)|(q.longValue()&fractionMask);}if(q.signum()==0)return sign;if(q.bitLength()>fractionBits)return sign|(1L<<fractionBits);return sign|q.longValue();}
    private static java.math.BigInteger roundShift(java.math.BigInteger value,long shift){if(shift<=0)return value.shiftLeft(Math.toIntExact(-shift));if(shift>Integer.MAX_VALUE)return java.math.BigInteger.ZERO;int s=(int)shift;java.math.BigInteger q=value.shiftRight(s);if(s==0)return q;boolean guard=value.testBit(s-1),sticky=value.getLowestSetBit()>=0&&value.getLowestSetBit()<s-1;return guard&&(sticky||q.testBit(0))?q.add(java.math.BigInteger.ONE):q;}

    /**
     * Independently validates and evaluates one indexing occurrence for conformance tests.
     * @param ir non-null closed indexing structural form
     * @param geometry non-null compact static geometry matching {@code ir}
     * @param arguments unique inputs followed by the output
     * @param start inclusive output logical bound
     * @param end exclusive output logical bound
     * @throws NullPointerException if a required argument is {@code null}
     * @throws IllegalArgumentException if boundary counts or output bounds disagree
     * @throws IndexOutOfBoundsException if the first logical index value is outside its selected
     *     data-axis extent or one-hot depth
     * @throws ArithmeticException if exact reference address or count arithmetic overflows
     */
    public static void execute(CpuIndexingIr ir, CpuIndexingLowering.Geometry geometry,
            List<CpuBufferArgument> arguments, long start, long end) {
        Objects.requireNonNull(ir, "ir"); Objects.requireNonNull(geometry, "geometry");
        if (arguments.size() != geometry.boundaries().size()) throw new IllegalArgumentException(
                "indexing reference boundary count is inconsistent");
        int indexBoundary = ir.family() == CpuIndexingIr.Family.ONE_HOT
                ? ir.occurrenceToBoundary().getFirst() : ir.occurrenceToBoundary().get(1);
        var indexLayout = geometry.boundaries().get(indexBoundary);
        long indexCount = count(indexLayout.extents());
        long[] indexCoordinate = new long[indexLayout.extents().length];
        int dataBoundary = ir.family() == CpuIndexingIr.Family.ONE_HOT ? -1
                : ir.occurrenceToBoundary().getFirst();
        long[] dataExtents = dataBoundary < 0 ? new long[0]
                : geometry.boundaries().get(dataBoundary).extents();
        int axis = geometry.variant() instanceof CpuIndexingLowering.Geometry.Axis a ? a.axis() : -1;
        int batch = geometry.variant() instanceof CpuIndexingLowering.Geometry.Nd n
                ? n.batchDimensions() : 0;
        int tuple = geometry.variant() instanceof CpuIndexingLowering.Geometry.Nd n
                ? n.tupleDepth() : 0;
        long fixedBound = geometry.variant() instanceof CpuIndexingLowering.Geometry.Hot h
                ? h.depth() : axis >= 0 ? dataExtents[axis] : -1;
        for (long ordinal = 0; ordinal < indexCount; ordinal++) {
            indexCoordinate = coordinates(ordinal, indexLayout.extents());
            long value = ((Number) load(arguments.get(indexBoundary),
                    geometry.boundaryTypes().get(indexBoundary),
                    address(indexLayout, indexCoordinate))).longValue();
            int selectedAxis = axis; long bound = fixedBound;
            if (ir.family() == CpuIndexingIr.Family.GATHER_ND) {
                selectedAxis = batch + (int) (ordinal % tuple); bound = dataExtents[selectedAxis];
            }
            if (value < 0 || value >= bound) throw indexingFailure(ir.family(), ordinal, value,
                    selectedAxis, bound);
        }
        long[] outputExtents = geometry.outputExtents();
        if (start < 0 || end < start || end > count(outputExtents))
            throw new IllegalArgumentException("invalid reference bounds");
        for (long logical = start; logical < end; logical++) {
            long[] out = coordinates(logical, outputExtents);
            long[] index = new long[indexLayout.extents().length];
            long[] data = new long[dataExtents.length];
            if (ir.family() == CpuIndexingIr.Family.GATHER) {
                System.arraycopy(out, axis, index, 0, index.length);
                long selected = ((Number) load(arguments.get(indexBoundary),
                        geometry.boundaryTypes().get(indexBoundary), address(indexLayout,index))).longValue();
                for(int a=0;a<axis;a++)data[a]=out[a]; data[axis]=selected;
                for(int a=axis+1;a<data.length;a++)data[a]=out[a-1+index.length];
            } else if (ir.family() == CpuIndexingIr.Family.GATHER_ELEMENTS) {
                index=out.clone(); data=out.clone();
                data[axis]=((Number)load(arguments.get(indexBoundary),
                        geometry.boundaryTypes().get(indexBoundary),address(indexLayout,index))).longValue();
            } else if (ir.family() == CpuIndexingIr.Family.GATHER_ND) {
                System.arraycopy(out,0,index,0,index.length-1);
                for(int a=0;a<batch;a++)data[a]=out[a];
                for(int k=0;k<tuple;k++){index[index.length-1]=k;data[batch+k]=((Number)load(
                        arguments.get(indexBoundary),geometry.boundaryTypes().get(indexBoundary),
                        address(indexLayout,index))).longValue();}
                for(int a=batch+tuple;a<data.length;a++)data[a]=out[index.length-1+a-batch-tuple];
            } else {
                System.arraycopy(out,0,index,0,index.length);
                long selected=((Number)load(arguments.get(indexBoundary),
                        geometry.boundaryTypes().get(indexBoundary),address(indexLayout,index))).longValue();
                store(arguments.getLast(),DataType.BOOL,address(geometry.boundaries().getLast(),out),
                        (byte)(selected==out[out.length-1]?1:0)); continue;
            }
            Object value=load(arguments.get(dataBoundary),geometry.boundaryTypes().get(dataBoundary),
                    address(geometry.boundaries().get(dataBoundary),data));
            store(arguments.getLast(),geometry.boundaryTypes().getLast(),
                    address(geometry.boundaries().getLast(),out),value);
        }
    }

    private static long count(long[] extents){if(java.util.Arrays.stream(extents).anyMatch(v->v==0))return 0;long n=1;for(long e:extents)n=Math.multiplyExact(n,e);return n;}
    private static long address(CpuIndexingLowering.Geometry.Layout layout,long[] coordinate){long a=layout.offset();long[] s=layout.strides();for(int i=0;i<coordinate.length;i++)a=Math.addExact(a,Math.multiplyExact(coordinate[i],s[i]));return a;}
    private static IndexOutOfBoundsException indexingFailure(CpuIndexingIr.Family family,long ordinal,long value,int axis,long bound){String m=switch(family){case GATHER->"GATHER index at logical position "+ordinal+" for data axis "+axis+" is out of bounds: value="+value+", extent="+bound;case GATHER_ELEMENTS->"GATHER_ELEMENTS index at logical position "+ordinal+" for data axis "+axis+" is out of bounds: value="+value+", extent="+bound;case GATHER_ND->"GATHER_ND index at logical position "+ordinal+" for data axis "+axis+" is out of bounds: value="+value+", extent="+bound;case ONE_HOT->"ONE_HOT index at logical position "+ordinal+" is out of bounds: value="+value+", depth="+bound;};return new IndexOutOfBoundsException(m);}

    /**
     * Evaluates the portable exact/default GELU target in fixed operation order.
     *
     * @param value input value, including IEEE 754 special values
     * @return {@code 0.5 * value * (1 + erf(value / sqrt(2)))} using the shared bounded error-
     *     function approximation, with negative infinity mapped to negative zero and other
     *     documented special-value classifications preserved
     */
    public static double gelu(double value) {
        if (value == Double.NEGATIVE_INFINITY) return -0.0d;
        return 0.5d * value * (1.0d + erf(value / Math.sqrt(2.0d)));
    }

    /**
     * Evaluates logistic sigmoid without avoidable exponential overflow.
     * @param value input value, including IEEE 754 special values
     * @return the stable two-branch sigmoid result; NaN remains NaN
     */
    public static double sigmoid(double value) {
        if (value >= 0.0d) return 1.0d / (1.0d + StrictMath.exp(-value));
        double exponential = StrictMath.exp(value);
        return exponential / (1.0d + exponential);
    }

    /**
     * Evaluates the fixed Model hyperbolic-tangent GELU approximation.
     * @param value input value, including IEEE 754 special values
     * @return the fixed-coefficient approximation, with negative infinity mapped to negative zero
     */
    public static double geluTanhApproximation(double value) {
        if (value == Double.NEGATIVE_INFINITY) return -0.0d;
        double cube = value * value * value;
        return 0.5d * value * (1.0d + StrictMath.tanh(Math.sqrt(2.0d / Math.PI)
                * (value + 0.044715d * cube)));
    }

    /**
     * Evaluates sigmoid linear unit without avoidable exponential overflow.
     * @param value input value, including IEEE 754 special values
     * @return {@code value * sigmoid(value)}, with negative infinity mapped to negative zero
     */
    public static double silu(double value) {
        if (value == Double.NEGATIVE_INFINITY) return -0.0d;
        if (value >= 0.0d) return value / (1.0d + StrictMath.exp(-value));
        double exponential = StrictMath.exp(value);
        return value * exponential / (1.0d + exponential);
    }

    /**
     * Evaluates a portable scalar approximation of the Gaussian error function.
     * The approximation is shared by generated and reference realizations and preserves NaN,
     * infinities, and signed zero classifications.
     *
     * @param value input value, including IEEE 754 special values
     * @return the finite approximation to the Gaussian error function, or the corresponding
     *     preserved NaN, infinity, or signed-zero classification
     */
    public static double erf(double value) {
        if (Double.isNaN(value)) return Double.NaN;
        if (value == 0.0d) return value;
        if (value == Double.POSITIVE_INFINITY) return 1.0d;
        if (value == Double.NEGATIVE_INFINITY) return -1.0d;
        double x = Math.abs(value);
        double result;
        if (x <= 1.0d) {
            double z = x * x;
            result = x * polevl(z, ERF_T) / p1evl(z, ERF_U);
        } else {
            double erfc = Math.exp(-x * x) * (x < 8.0d
                    ? polevl(x, ERFC_P) / p1evl(x, ERFC_Q)
                    : polevl(x, ERFC_R) / p1evl(x, ERFC_S));
            result = 1.0d - erfc;
        }
        return Math.copySign(result, value);
    }

    /**
     * Executes the fused reference calculation over one half-open range.
     *
     * @param a non-null first ADD input; not mutated
     * @param b non-null second ADD input; not mutated
     * @param c non-null MUL input; not mutated
     * @param output non-null destination mutated only in {@code [start, end)}
     * @param start non-negative inclusive element index
     * @param end exclusive element index no greater than any array length
     * @throws NullPointerException if an array is {@code null}
     * @throws IllegalArgumentException if the half-open range is negative, reversed, or exceeds an
     *     input or output array
     */
    public static void execute(double[] a, double[] b, double[] c, double[] output,
            long start, long end) {
        if (start < 0 || end < start || end > a.length || end > b.length || end > c.length
                || end > output.length) throw new IllegalArgumentException("invalid reference bounds");
        for (long index = start; index < end; index++) {
            double sum = a[(int) index] + b[(int) index];
            double activated = gelu(sum);
            output[(int) index] = activated * c[(int) index];
        }
    }

    /**
     * Executes the completed four-boundary FLOAT64 proving topology over the same normalized
     * bindings and direct carrier forms as generated scalar code.
     * This reference path may allocate coordinate arrays and use division/modulo because it is
     * conformance support, not the generated Runtime hot path.
     *
     * @param arguments non-null ordered direct inputs {@code a}, {@code b}, {@code c}, and output
     * @param bindings non-null matching normalized access bindings in the same order
     * @param start non-negative inclusive logical element bound
     * @param end exclusive logical element bound no greater than the first binding's count
     * @throws NullPointerException if an argument, binding, or list is {@code null}
     * @throws IllegalArgumentException if boundary counts or range are invalid
     * @throws ArithmeticException if exact address arithmetic overflows
     */
    public static void execute(List<CpuBufferArgument> arguments,
            List<CpuAccessPlan.Binding> bindings, long start, long end) {
        if (arguments.size() != 4 || bindings.size() != 4) throw new IllegalArgumentException(
                "reference execution requires four ordered boundaries");
        CpuAccessPlan.Binding first = bindings.getFirst();
        if (start < 0 || end < start || end > first.elementCount()) {
            throw new IllegalArgumentException("invalid reference bounds");
        }
        long[] extents = first.extents().stream().mapToLong(Long::longValue).toArray();
        for (long index = start; index < end; index++) {
            long[] coordinate = coordinates(index, extents);
            double sum = load(arguments.get(0), address(bindings.get(0), coordinate))
                    + load(arguments.get(1), address(bindings.get(1), coordinate));
            double result = gelu(sum)
                    * load(arguments.get(2), address(bindings.get(2), coordinate));
            store(arguments.get(3), address(bindings.get(3), coordinate), result);
        }
    }

    /**
     * Executes one already-lowered typed pointwise IR for differential conformance.
     *
     * @param ir non-null typed CPU pointwise IR
     * @param arguments non-null materialized boundary arguments in IR boundary order
     * @param bindings non-null normalized boundary bindings in the same order
     * @param start non-negative inclusive logical bound
     * @param end exclusive logical bound
     * @throws NullPointerException if {@code ir}, a list, argument, or binding is {@code null}
     * @throws IllegalArgumentException if boundary counts or the half-open range are invalid
     * @throws ArithmeticException if exact coordinate or address arithmetic overflows
     */
    public static void execute(CpuKernelIr ir, List<CpuBufferArgument> arguments,
            List<CpuAccessPlan.Binding> bindings, long start, long end) {
        List<CpuKernelIr.Value> boundaries = ir.values().stream()
                .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL).toList();
        if (arguments.size() != boundaries.size() || bindings.size() != boundaries.size()
                || start < 0 || end < start || end > bindings.getFirst().elementCount()) {
            throw new IllegalArgumentException("invalid typed reference boundaries or range");
        }
        long[] extents = bindings.getFirst().extents().stream().mapToLong(Long::longValue).toArray();
        Object[] values = new Object[ir.values().size()];
        for (long index = start; index < end; index++) {
            long[] coordinate = coordinates(index, extents);
            for (int boundary = 0; boundary < boundaries.size(); boundary++) {
                CpuKernelIr.Value value = boundaries.get(boundary);
                if (value.kind() == CpuKernelIr.Value.Kind.INPUT
                        && requiresInputLoad(ir, value.ordinal())) values[value.ordinal()] =
                        load(arguments.get(boundary), value.dataType(),
                                address(bindings.get(boundary), coordinate));
            }
            for (CpuKernelIr.Instruction instruction : ir.instructions()) {
                values[instruction.output()] = evaluate(ir, instruction, values);
            }
            for (CpuKernelIr.Store store : ir.stores()) {
                CpuKernelIr.Value value = ir.values().get(store.value());
                int boundary = 0;
                while (boundaries.get(boundary).ordinal() != value.ordinal()) boundary++;
                store(arguments.get(boundary), value.dataType(),
                        address(bindings.get(boundary), coordinate), values[value.ordinal()]);
            }
        }
    }

    /**
     * Executes one cold-composed represented-bit affine address sequence for differential tests.
     *
     * @param ir non-null affine copy contract supplying the represented data type
     * @param addressPairs non-null alternating source and result element addresses
     * @param arguments non-null ordered source and writable result arguments
     * @param start non-negative inclusive address-pair index
     * @param end exclusive address-pair index no greater than the available pair count
     * @throws NullPointerException if {@code ir} or {@code addressPairs} is {@code null}
     * @throws IllegalArgumentException if the argument count, pair table, or range is invalid
     * @throws ArithmeticException if a requested pair index cannot be represented safely
     */
    public static void execute(CpuAffineCopyIr ir, long[] addressPairs,
            List<CpuBufferArgument> arguments, long start, long end) {
        Objects.requireNonNull(ir, "ir");
        Objects.requireNonNull(addressPairs, "addressPairs");
        if (arguments.size() != 2 || addressPairs.length % 2 != 0 || start < 0 || end < start
                || end > addressPairs.length / 2) {
            throw new IllegalArgumentException("invalid affine reference boundaries or range");
        }
        for (long index = start; index < end; index++) {
            int pair = Math.toIntExact(Math.multiplyExact(index, 2));
            Object represented = load(arguments.get(0), ir.dataType(), addressPairs[pair]);
            store(arguments.get(1), ir.dataType(), addressPairs[pair + 1], represented);
        }
    }

    /**
     * Executes one compact static movement mapping for differential conformance.
     *
     * @param ir non-null represented-bit movement IR, including window extraction and functional
     *     slice update
     * @param geometry non-null matching compact cold occurrence geometry; window forms may have
     *     unequal input and output ranks
     * @param arguments non-null unique input arguments followed by one writable output
     * @param start non-negative inclusive output logical bound
     * @param end exclusive output logical bound
     * @throws NullPointerException if {@code ir}, {@code geometry}, {@code arguments}, or a
     *     required argument is {@code null}
     * @throws IllegalArgumentException if boundary counts or bounds are inconsistent
     * @throws ArithmeticException if the output element count or an address calculation overflows
     */
    public static void execute(CpuDataMovementIr ir,
            CpuNonAffineMovementLowering.Geometry geometry,
            List<CpuBufferArgument> arguments, long start, long end) {
        Objects.requireNonNull(ir, "ir");
        Objects.requireNonNull(geometry, "geometry");
        long[] outputExtents = geometry.outputExtents();
        long count = outputExtents.length == 0 ? 1 : 1;
        for (long extent : outputExtents) count = Math.multiplyExact(count, extent);
        if (arguments.size() != geometry.inputs().size() + 1
                || start < 0 || end < start || end > count) {
            throw new IllegalArgumentException("invalid movement reference boundaries or range");
        }
        long[] outputStrides = geometry.outputStrides();
        int output = arguments.size() - 1;
        for (long logical = start; logical < end; logical++) {
            long[] coordinate = coordinates(logical, outputExtents);
            Object represented;
            if (ir.plan() instanceof CpuDataMovementIr.PadPlan pad) {
                var variant = (CpuNonAffineMovementLowering.Geometry.Pad) geometry.variant();
                long[] before = variant.before(), inputExtents = variant.inputExtents();
                boolean fill = false;
                long[] source = coordinate.clone();
                for (int axis = 0; axis < source.length; axis++) {
                    source[axis] -= before[axis];
                    fill |= source[axis] < 0 || source[axis] >= inputExtents[axis];
                }
                represented = fill ? representedImmediate(ir.dataType(), pad.immediateBits())
                        : load(arguments.getFirst(), ir.dataType(),
                            movementAddress(geometry.inputs().getFirst(), source));
            } else if (ir.plan() instanceof CpuDataMovementIr.TilePlan) {
                var variant = (CpuNonAffineMovementLowering.Geometry.Tile) geometry.variant();
                long[] source = coordinate.clone(), extents = variant.inputExtents();
                for (int axis = 0; axis < source.length; axis++) source[axis] %= extents[axis];
                represented = load(arguments.getFirst(), ir.dataType(),
                        movementAddress(geometry.inputs().getFirst(), source));
            } else if (ir.plan() instanceof CpuDataMovementIr.ConcatPlan concat) {
                var variant = (CpuNonAffineMovementLowering.Geometry.Concat) geometry.variant();
                long[] prefixes = variant.prefixes();
                int occurrence = 0;
                while (coordinate[variant.axis()] >= prefixes[occurrence + 1]) occurrence++;
                long[] source = coordinate.clone();
                source[variant.axis()] -= prefixes[occurrence];
                int boundary = concat.occurrenceToBoundary().get(occurrence);
                represented = load(arguments.get(boundary), ir.dataType(),
                        movementAddress(geometry.inputs().get(boundary), source));
            } else if (ir.plan() instanceof CpuDataMovementIr.StackPlan stack) {
                var variant = (CpuNonAffineMovementLowering.Geometry.Stack) geometry.variant();
                int occurrence = Math.toIntExact(coordinate[variant.axis()]);
                long[] source = new long[coordinate.length - 1];
                for (int outAxis = 0, inAxis = 0; outAxis < coordinate.length; outAxis++) {
                    if (outAxis != variant.axis()) source[inAxis++] = coordinate[outAxis];
                }
                int boundary = stack.occurrenceToBoundary().get(occurrence);
                represented = load(arguments.get(boundary), ir.dataType(),
                        movementAddress(geometry.inputs().get(boundary), source));
            } else if (ir.plan() instanceof CpuDataMovementIr.UnfoldAxisPlan) {
                var variant = (CpuNonAffineMovementLowering.Geometry.UnfoldAxis) geometry.variant();
                long[] source = new long[coordinate.length - 1];
                for (int axis = 0; axis < source.length; axis++) source[axis] = axis == variant.axis()
                        ? Math.addExact(Math.multiplyExact(coordinate[axis], variant.step()),
                            coordinate[coordinate.length - 1])
                        : coordinate[axis];
                represented = load(arguments.getFirst(), ir.dataType(),
                        movementAddress(geometry.inputs().getFirst(), source));
            } else if (ir.plan() instanceof CpuDataMovementIr.SliceUpdatePlan slicePlan) {
                var variant = (CpuNonAffineMovementLowering.Geometry.SliceUpdate) geometry.variant();
                long[] starts = variant.starts(), lengths = variant.lengths(), steps = variant.steps();
                long[] updateCoordinate = new long[coordinate.length];
                boolean selected = true;
                for (int axis = 0; axis < coordinate.length; axis++) {
                    long length = lengths[axis];
                    if (length == 0) { selected = false; break; }
                    long ordinal = -1;
                    for (long candidate = 0; candidate < length; candidate++) {
                        if (Math.addExact(starts[axis], Math.multiplyExact(candidate, steps[axis]))
                                == coordinate[axis]) {
                            ordinal = candidate;
                            break;
                        }
                    }
                    if (ordinal < 0) { selected = false; break; }
                    updateCoordinate[axis] = ordinal;
                }
                int boundary = slicePlan.occurrenceToBoundary().get(selected ? 1 : 0);
                long[] source = selected ? updateCoordinate : coordinate;
                represented = load(arguments.get(boundary), ir.dataType(),
                        movementAddress(geometry.inputs().get(boundary), source));
            } else {
                var plan = (CpuDataMovementIr.Unfold2dPlan) ir.plan();
                var variant = (CpuNonAffineMovementLowering.Geometry.Unfold2d) geometry.variant();
                long q = coordinate[1], p = coordinate[2];
                long kw = q % variant.kernelWidth();
                long kh = q / variant.kernelWidth() % variant.kernelHeight();
                long channel = q / Math.multiplyExact(variant.kernelHeight(), variant.kernelWidth());
                long ow = p % variant.outputWidth(), oh = p / variant.outputWidth();
                long ih = Math.addExact(Math.subtractExact(Math.multiplyExact(oh,
                        variant.strideHeight()), variant.paddingHeight()),
                        Math.multiplyExact(kh, variant.dilationHeight()));
                long iw = Math.addExact(Math.subtractExact(Math.multiplyExact(ow,
                        variant.strideWidth()), variant.paddingWidth()),
                        Math.multiplyExact(kw, variant.dilationWidth()));
                represented = ih < 0 || ih >= variant.height() || iw < 0 || iw >= variant.width()
                        ? representedImmediate(ir.dataType(), plan.immediateBits())
                        : load(arguments.getFirst(), ir.dataType(),
                            movementAddress(geometry.inputs().getFirst(),
                                new long[] {coordinate[0], channel, ih, iw}));
            }
            long outputAddress = geometry.outputOffset();
            for (int axis = 0; axis < coordinate.length; axis++) outputAddress = Math.addExact(
                    outputAddress, Math.multiplyExact(coordinate[axis], outputStrides[axis]));
            store(arguments.get(output), ir.dataType(), outputAddress, represented);
        }
    }

    private static long movementAddress(CpuNonAffineMovementLowering.Geometry.Input input,
            long[] coordinate) {
        long result = input.offset();
        long[] strides = input.strides();
        for (int axis = 0; axis < coordinate.length; axis++) result = Math.addExact(result,
                Math.multiplyExact(coordinate[axis], strides[axis]));
        return result;
    }

    private static Object representedImmediate(DataType type, long bits) {
        return switch (type) {
            case FLOAT64 -> Double.longBitsToDouble(bits);
            case FLOAT32 -> Float.intBitsToFloat((int) bits);
            case BFLOAT16 -> (short) bits;
            case INT32 -> (int) bits;
            case INT64 -> bits;
            case BOOL -> (byte) bits;
        };
    }

    private static Object evaluate(CpuKernelIr ir, CpuKernelIr.Instruction instruction,
            Object[] values) {
        DataType type = ir.values().get(instruction.inputs().getFirst()).dataType();
        Object left = values[instruction.inputs().getFirst()];
        Object right = instruction.inputs().size() > 1 ? values[instruction.inputs().get(1)] : null;
        Object scalar = instruction.scalarImmediate() == null ? null
                : immediate(instruction.scalarImmediate());
        return switch (instruction.opcode()) {
            case ADD -> arithmetic(type, left, right, 0);
            case SUB -> arithmetic(type, left, right, 1);
            case MUL -> arithmetic(type, left, right, 2);
            case DIV -> arithmetic(type, left, right, 3);
            case MIN -> extrema(type, left, right, true);
            case MAX -> extrema(type, left, right, false);
            case POW -> tensorPower(type, left, right);
            case SCALAR_ADD -> arithmetic(type, left, scalar, 0);
            case SCALAR_SUB -> arithmetic(type, left, scalar, 1);
            case SCALAR_MUL -> arithmetic(type, left, scalar, 2);
            case SCALAR_DIV -> arithmetic(type, left, scalar, 3);
            case SCALAR_POW -> power(type, left, instruction.scalarImmediate(),
                    instruction.powerRealization());
            case SCALAR_MIN -> extrema(type, left, scalar, true);
            case SCALAR_MAX -> extrema(type, left, scalar, false);
            case SCALAR_CLAMP -> extrema(type,
                    extrema(type, left, immediate(instruction.clampImmediate().lower()), false),
                    immediate(instruction.clampImmediate().upper()), true);
            case NEG -> { if (type == DataType.FLOAT64) yield Double.valueOf(-(double) left);
                yield Float.valueOf(-(float) left); }
            case ABS, RECIPROCAL, LOG, LOG1P, EXP, EXPM1, ERF, SQRT, RSQRT, FLOOR, CEIL,
                    SIGN, RELU, SIGMOID, TANH, GELU_EXACT, GELU_TANH_APPROXIMATION, SILU ->
                    unary(instruction.opcode(), type, left);
            case IS_FINITE -> (byte) ((type == DataType.FLOAT64
                    ? Double.isFinite((double) left) : Float.isFinite((float) left)) ? 1 : 0);
            case IS_NAN -> (byte) ((type == DataType.FLOAT64
                    ? Double.isNaN((double) left) : Float.isNaN((float) left)) ? 1 : 0);
            case IS_INF -> (byte) ((type == DataType.FLOAT64
                    ? Double.isInfinite((double) left) : Float.isInfinite((float) left)) ? 1 : 0);
            case GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL ->
                    bool(relation(instruction.opcode(), type, left, right));
            case EQUAL -> bool(equal(type, left, right));
            case NOT_EQUAL -> bool(!equal(type, left, right));
            case LOGICAL_AND -> bool((byte) left == 1 && (byte) right == 1);
            case LOGICAL_OR -> bool((byte) left == 1 || (byte) right == 1);
            case LOGICAL_NOT -> bool((byte) left == 0);
            case WHERE -> ((byte) left) == 1 ? values[instruction.inputs().get(1)]
                    : values[instruction.inputs().get(2)];
            case CAST -> left;
        };
    }

    private static Object unary(CpuPointwiseOpcode opcode, DataType type, Object input) {
        double value = type == DataType.FLOAT64 ? (double) input : (double) (float) input;
        double result = switch (opcode) {
            case ABS -> Math.abs(value); case RECIPROCAL -> 1.0d / value;
            case LOG -> StrictMath.log(value); case LOG1P -> StrictMath.log1p(value);
            case EXP -> StrictMath.exp(value); case EXPM1 -> StrictMath.expm1(value);
            case ERF -> erf(value); case SQRT -> StrictMath.sqrt(value);
            case RSQRT -> 1.0d / StrictMath.sqrt(value); case FLOOR -> StrictMath.floor(value);
            case CEIL -> StrictMath.ceil(value); case SIGN -> Math.signum(value);
            case RELU -> Math.max(value, +0.0d); case SIGMOID -> sigmoid(value);
            case TANH -> StrictMath.tanh(value); case GELU_EXACT -> gelu(value);
            case GELU_TANH_APPROXIMATION -> geluTanhApproximation(value); case SILU -> silu(value);
            default -> throw new AssertionError(opcode);
        };
        if (type == DataType.FLOAT64) return Double.valueOf(result);
        return Float.valueOf((float) result);
    }

    private static Object tensorPower(DataType type, Object base, Object exponent) {
        if (type == DataType.FLOAT64) return Double.valueOf(
                StrictMath.pow((double) base, (double) exponent));
        return Float.valueOf((float) StrictMath.pow((double) (float) base,
                (double) (float) exponent));
    }

    private static Object extrema(DataType type, Object left, Object right, boolean minimum) {
        return switch (type) {
            case FLOAT64 -> minimum ? Math.min((double) left, (double) right)
                    : Math.max((double) left, (double) right);
            case FLOAT32 -> minimum ? Math.min((float) left, (float) right)
                    : Math.max((float) left, (float) right);
            case INT32 -> minimum ? Math.min((int) left, (int) right)
                    : Math.max((int) left, (int) right);
            case INT64 -> minimum ? Math.min((long) left, (long) right)
                    : Math.max((long) left, (long) right);
            default -> throw new IllegalArgumentException("unsupported extrema type");
        };
    }

    private static Object arithmetic(DataType type, Object left, Object right, int operation) {
        return switch (type) {
            case FLOAT64 -> { double a = (double) left, b = (double) right;
                yield operation == 0 ? a + b : operation == 1 ? a - b
                        : operation == 2 ? a * b : a / b; }
            case FLOAT32 -> { float a = (float) left, b = (float) right;
                yield operation == 0 ? a + b : operation == 1 ? a - b
                        : operation == 2 ? a * b : a / b; }
            case INT32 -> { int a = (int) left, b = (int) right;
                if (operation == 3) throw new IllegalArgumentException("integral division unsupported");
                yield operation == 0 ? a + b : operation == 1 ? a - b : a * b; }
            case INT64 -> { long a = (long) left, b = (long) right;
                if (operation == 3) throw new IllegalArgumentException("integral division unsupported");
                yield operation == 0 ? a + b : operation == 1 ? a - b : a * b; }
            default -> throw new IllegalArgumentException("unsupported arithmetic type");
        };
    }

    private static Object power(DataType type, Object base, CpuKernelIr.ScalarImmediate exponent,
            CpuKernelIr.PowerRealization realization) {
        if (type == DataType.FLOAT64) {
            double value = base == null ? Double.NaN : (double) base;
            return switch (realization) {
                case DIRECT -> StrictMath.pow(value, Double.longBitsToDouble(exponent.bits()));
                case POSITIVE_ONE -> 1.0d;
                case IDENTITY -> value;
                case SQUARE -> value * value;
                case RECIPROCAL -> 1.0d / value;
            };
        }
        float value = base == null ? Float.NaN : (float) base;
        return switch (realization) {
            case DIRECT -> (float) StrictMath.pow((double) value,
                    (double) Float.intBitsToFloat((int) exponent.bits()));
            case POSITIVE_ONE -> 1.0f;
            case IDENTITY -> value;
            case SQUARE -> value * value;
            case RECIPROCAL -> 1.0f / value;
        };
    }

    private static boolean requiresInputLoad(CpuKernelIr ir, int ordinal) {
        return ir.instructions().stream().anyMatch(instruction -> {
            for (int input = 0; input < instruction.inputs().size(); input++) {
                if (instruction.inputs().get(input) != ordinal) continue;
                if (input == 0 && instruction.opcode()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.SCALAR_POW
                        && instruction.powerRealization()
                            == CpuKernelIr.PowerRealization.POSITIVE_ONE) continue;
                return true;
            }
            return false;
        });
    }

    private static boolean relation(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode opcode,
            DataType type, Object left, Object right) {
        int relation = switch (type) {
            case FLOAT64 -> (double) left > (double) right ? 1 : (double) left < (double) right ? -1
                    : (double) left == (double) right ? 0 : 2;
            case FLOAT32 -> (float) left > (float) right ? 1 : (float) left < (float) right ? -1
                    : (float) left == (float) right ? 0 : 2;
            case INT32 -> Integer.compare((int) left, (int) right);
            case INT64 -> Long.compare((long) left, (long) right);
            default -> throw new IllegalArgumentException("unsupported comparison type");
        };
        return switch (opcode) {
            case GREATER_THAN -> relation == 1; case GREATER_OR_EQUAL -> relation == 1 || relation == 0;
            case LESS_THAN -> relation == -1; case LESS_OR_EQUAL -> relation == -1 || relation == 0;
            default -> throw new AssertionError(opcode);
        };
    }

    private static boolean equal(DataType type, Object left, Object right) {
        return switch (type) {
            case FLOAT64 -> (double) left == (double) right;
            case FLOAT32 -> (float) left == (float) right;
            case INT32 -> (int) left == (int) right;
            case INT64 -> (long) left == (long) right;
            default -> throw new IllegalArgumentException("unsupported comparison type");
        };
    }

    private static byte bool(boolean value) { return (byte) (value ? 1 : 0); }

    private static Object immediate(CpuKernelIr.ScalarImmediate value) {
        return switch (value.dataType()) {
            case FLOAT64 -> Double.longBitsToDouble(value.bits());
            case FLOAT32 -> Float.intBitsToFloat((int) value.bits());
            case INT32 -> (int) value.bits(); case INT64 -> value.bits();
            default -> throw new IllegalArgumentException("unsupported immediate type");
        };
    }

    private static long[] coordinates(long index, long[] extents) {
        long[] result = new long[extents.length];
        for (int axis = extents.length - 1; axis >= 0; axis--) if (extents[axis] != 0) {
            result[axis] = index % extents[axis]; index /= extents[axis];
        }
        return result;
    }

    private static long address(CpuAccessPlan.Binding binding, long[] coordinates) {
        long address = binding.baseElementOffset();
        for (int axis = 0; axis < coordinates.length; axis++) address = Math.addExact(address,
                Math.multiplyExact(coordinates[axis], binding.effectiveStrides().get(axis)));
        return address;
    }

    private static double load(CpuBufferArgument argument, long address) {
        if (argument instanceof CpuBufferArgument.Doubles doubles) return doubles.carrier()[
                Math.toIntExact(doubles.byteOffset() / Double.BYTES + address)];
        return ((CpuBufferArgument.Segment) argument).segment().get(DOUBLE,
                Math.multiplyExact(address, Double.BYTES));
    }

    private static Object load(CpuBufferArgument argument, DataType type, long address) {
        long base = argument.byteOffset() / type.byteWidth() + address;
        if (argument instanceof CpuBufferArgument.Doubles value) return value.carrier()[Math.toIntExact(base)];
        if (argument instanceof CpuBufferArgument.Floats value) return value.carrier()[Math.toIntExact(base)];
        if (argument instanceof CpuBufferArgument.Shorts value) return value.carrier()[Math.toIntExact(base)];
        if (argument instanceof CpuBufferArgument.Ints value) return value.carrier()[Math.toIntExact(base)];
        if (argument instanceof CpuBufferArgument.Longs value) return value.carrier()[Math.toIntExact(base)];
        if (argument instanceof CpuBufferArgument.Bytes value) return value.carrier()[Math.toIntExact(base)];
        var segment = ((CpuBufferArgument.Segment) argument).segment();
        long offset = Math.multiplyExact(address, type.byteWidth());
        return switch (type) {
            case FLOAT64 -> segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset);
            case FLOAT32 -> segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset);
            case BFLOAT16 -> segment.get(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset);
            case INT32 -> segment.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset);
            case INT64 -> segment.get(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset);
            case BOOL -> segment.get(ValueLayout.JAVA_BYTE, offset);
            default -> throw new IllegalArgumentException("unsupported reference type");
        };
    }

    private static void store(CpuBufferArgument argument, long address, double value) {
        if (argument instanceof CpuBufferArgument.Doubles doubles) doubles.carrier()[
                Math.toIntExact(doubles.byteOffset() / Double.BYTES + address)] = value;
        else ((CpuBufferArgument.Segment) argument).segment().set(DOUBLE,
                Math.multiplyExact(address, Double.BYTES), value);
    }

    private static void store(CpuBufferArgument argument, DataType type, long address, Object stored) {
        long base = argument.byteOffset() / type.byteWidth() + address;
        if (argument instanceof CpuBufferArgument.Doubles value) value.carrier()[Math.toIntExact(base)] = (double) stored;
        else if (argument instanceof CpuBufferArgument.Floats value) value.carrier()[Math.toIntExact(base)] = (float) stored;
        else if (argument instanceof CpuBufferArgument.Shorts value) value.carrier()[Math.toIntExact(base)] = (short) stored;
        else if (argument instanceof CpuBufferArgument.Ints value) value.carrier()[Math.toIntExact(base)] = (int) stored;
        else if (argument instanceof CpuBufferArgument.Longs value) value.carrier()[Math.toIntExact(base)] = (long) stored;
        else if (argument instanceof CpuBufferArgument.Bytes value) value.carrier()[Math.toIntExact(base)] = (byte) stored;
        else {
            var segment = ((CpuBufferArgument.Segment) argument).segment();
            long offset = Math.multiplyExact(address, type.byteWidth());
            switch (type) {
                case FLOAT64 -> segment.set(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, (double) stored);
                case FLOAT32 -> segment.set(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, (float) stored);
                case BFLOAT16 -> segment.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, (short) stored);
                case INT32 -> segment.set(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, (int) stored);
                case INT64 -> segment.set(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, (long) stored);
                case BOOL -> segment.set(ValueLayout.JAVA_BYTE, offset, (byte) stored);
                default -> throw new IllegalArgumentException("unsupported reference type");
            }
        }
    }

    private static double polevl(double x, double[] coefficients) {
        double result = coefficients[0];
        for (int i = 1; i < coefficients.length; i++) result = result * x + coefficients[i];
        return result;
    }

    private static double p1evl(double x, double[] coefficients) {
        double result = x + coefficients[0];
        for (int i = 1; i < coefficients.length; i++) result = result * x + coefficients[i];
        return result;
    }
}
